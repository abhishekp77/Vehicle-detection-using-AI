import time
from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------- Config ----------------
APPROACHES = ["N", "E", "S", "W"]  # North, East, South, West

WEIGHTS = {  # impact weights
    "car": 1.0,
    "motorbike": 0.5,
    "truck": 2.5,
    "bus": 2.0,
}

# controller parameters
ALPHA_EMA = 0.5            # smoothing for demand
CYCLE_MIN = 60             # min cycle seconds
CYCLE_MAX = 120            # max cycle seconds
GREEN_MIN = 8              # min green per phase
GREEN_MAX = 60             # max green per phase
YELLOW = 3                 # yellow clearance (s)
ALL_RED = 1                # all-red (s)
FAIRNESS_MAX_SKIP = 3      # ensure no approach is skipped > this many cycles
PREEMPT_MIN_GREEN = 12     # min green during emergency preemption (s)
PREEMPT_MAX_GREEN = 25     # max green during preemption (s)

# ---------------- Data classes ----------------
@dataclass
class Counts:
    car: int = 0
    motorbike: int = 0
    truck: int = 0
    bus: int = 0
    emergency: bool = False  # set True if ambulance/fire seen in this approach

@dataclass
class ApproachState:
    ema_score: float = 0.0
    skip_streak: int = 0

@dataclass
class Plan:
    phase_order: list = field(default_factory=list)  # e.g., ["N", "S", "E", "W"]
    green_times: Dict[str, int] = field(default_factory=dict)
    cycle_time: int = 0
    reason: str = ""

# ---------------- Controller ----------------
class AdaptiveSignalController:
    def __init__(self):
        self.state: Dict[str, ApproachState] = {a: ApproachState() for a in APPROACHES}
        self.last_phase: Optional[str] = None

    def _score(self, counts: Counts) -> float:
        return (WEIGHTS["car"] * counts.car +
                WEIGHTS["motorbike"] * counts.motorbike +
                WEIGHTS["truck"] * counts.truck +
                WEIGHTS["bus"] * counts.bus)

    def _update_ema(self, raw_scores: Dict[str, float]):
        for a in APPROACHES:
            s = self.state[a]
            s.ema_score = ALPHA_EMA * raw_scores[a] + (1 - ALPHA_EMA) * s.ema_score

    def _apply_fairness(self, order: list):
        # Ensure approaches not served recently get moved earlier
        order.sort(key=lambda a: self.state[a].skip_streak, reverse=True)
        return order

    def _normalize_split(self, scores: Dict[str, float], green_budget: int) -> Dict[str, int]:
        # Normalize EMA scores into green seconds with min/max constraints
        total = sum(max(0.0, v) for v in scores.values())
        if total == 0:
            # give equal minimal service
            base_each = max(GREEN_MIN, green_budget // len(scores))
            return {a: min(GREEN_MAX, base_each) for a in scores}

        # initial proportional
        raw = {a: int(green_budget * (max(0.0, s) / total)) for a, s in scores.items()}
        # enforce min
        for a in raw:
            if raw[a] < GREEN_MIN:
                raw[a] = GREEN_MIN
        # enforce max
        for a in raw:
            if raw[a] > GREEN_MAX:
                raw[a] = GREEN_MAX

        # adjust to fit budget
        diff = green_budget - sum(raw.values())
        # distribute remaining seconds (positive or negative) by scores
        if diff != 0:
            # sort by score descending if diff>0 else ascending
            sorted_as = sorted(scores, key=lambda a: scores[a], reverse=(diff > 0))
            i = 0
            while diff != 0 and i < len(sorted_as) * 8:
                a = sorted_as[i % len(sorted_as)]
                if diff > 0 and raw[a] < GREEN_MAX:
                    raw[a] += 1
                    diff -= 1
                elif diff < 0 and raw[a] > GREEN_MIN:
                    raw[a] -= 1
                    diff += 1
                i += 1
        return raw

    def compute_plan(self, snapshot: Dict[str, Counts]) -> Plan:
        # 1) Raw scores, with emergency flags
        raw_scores = {a: self._score(snapshot.get(a, Counts())) for a in APPROACHES}
        emergencies = {a: snapshot.get(a, Counts()).emergency for a in APPROACHES}

        # 2) EMA smoothing
        self._update_ema(raw_scores)
        ema_scores = {a: self.state[a].ema_score for a in APPROACHES}

        # 3) Emergency preemption?
        em_approaches = [a for a, f in emergencies.items() if f]
        if em_approaches:
            # Pick the highest EMA among emergencies
            target = max(em_approaches, key=lambda a: ema_scores[a])
            # update fairness (others skipped)
            for a in APPROACHES:
                if a == target:
                    self.state[a].skip_streak = 0
                else:
                    self.state[a].skip_streak += 1
            return Plan(
                phase_order=[target],
                green_times={target: max(PREEMPT_MIN_GREEN, min(PREEMPT_MAX_GREEN, int(ema_scores[target] // 2 + GREEN_MIN)))},
                cycle_time=YELLOW + ALL_RED + max(PREEMPT_MIN_GREEN, min(PREEMPT_MAX_GREEN, int(ema_scores[target] // 2 + GREEN_MIN))),
                reason=f"Emergency preemption on {target}"
            )

        # 4) Build base order by EMA score (high to low)
        order = sorted(APPROACHES, key=lambda a: ema_scores[a], reverse=True)
        # 5) Apply fairness (move starved approaches earlier)
        order = self._apply_fairness(order)

        # 6) Decide cycle time based on congestion level (sum of EMA)
        congestion = sum(ema_scores.values())
        cycle = int(max(CYCLE_MIN, min(CYCLE_MAX, CYCLE_MIN + (congestion / 50))))  # heuristic

        # 7) Allocate green budget (minus intergreens)
        intergreens = (YELLOW + ALL_RED) * len(order)
        green_budget = max(CYCLE_MIN, cycle) - intergreens
        green_splits = self._normalize_split({a: ema_scores[a] for a in order}, green_budget)

        # 8) Update fairness streaks
        for a in APPROACHES:
            if green_splits.get(a, 0) > 0:
                self.state[a].skip_streak = 0
            else:
                self.state[a].skip_streak += 1
        # clamp skip streak
        for a in APPROACHES:
            if self.state[a].skip_streak > FAIRNESS_MAX_SKIP:
                self.state[a].skip_streak = FAIRNESS_MAX_SKIP

        return Plan(
            phase_order=order,
            green_times=green_splits,
            cycle_time=green_budget + intergreens,
            reason="Adaptive split by EMA demand with fairness"
        )

# ---------------- Demo loop ----------------
def demo():
    """
    Demo with synthetic inputs. Replace get_live_counts() with your detector.
    """
    ctl = AdaptiveSignalController()

    def get_live_counts() -> Dict[str, Counts]:
        # TODO: Replace with real detector outputs.
        # Example: heavier traffic on East, some trucks on South, emergency on N randomly.
        import random
        return {
            "N": Counts(car=random.randint(5, 20), motorbike=random.randint(10, 30), truck=random.randint(0, 2), bus=random.randint(0, 1), emergency=False),
            "E": Counts(car=random.randint(30, 60), motorbike=random.randint(20, 40), truck=random.randint(1, 4), bus=random.randint(0, 2), emergency=False),
            "S": Counts(car=random.randint(5, 15), motorbike=random.randint(5, 15), truck=random.randint(2, 40), bus=random.randint(0, 1), emergency=False),
            "W": Counts(car=random.randint(5, 12), motorbike=random.randint(5, 10), truck=random.randint(0, 1), bus=random.randint(0, 1), emergency=False),
        }

    print("Starting adaptive controller demo. Press Ctrl+C to stop.\n")
    while True:
        snapshot = get_live_counts()  # <- plug your live counts here
        plan = ctl.compute_plan(snapshot)

        # Pretty print
        print(f"[PLAN] Reason: {plan.reason}")
        print(f"       Cycle: {plan.cycle_time}s | Order: {plan.phase_order}")
        for a in plan.phase_order:
            g = plan.green_times.get(a, 0)
            sc = ctl.state[a].ema_score
            print(f"       {a}: green={g:>2}s  | EMA demand={sc:6.1f} | counts={snapshot[a].__dict__}")
        print(f"       Intergreens: yellow={YELLOW}s + all-red={ALL_RED}s between phases\n")

        # Simulate running the plan (in real life, send to controller/PLC here)
        time.sleep(2)  # shorten for demo; in production, you'd actuate real times

if __name__ == "__main__":
    demo()
