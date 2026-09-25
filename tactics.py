"""Tactical layer: the drone's programmable common sense.

Everything a human should review lives at the top of this file: the questions,
the option rubrics, and the thresholds. Nothing else in the project hard-codes a
number that changes how the aircraft reacts to a judgment.

Runs off the control loop in a worker thread. The flight code never blocks on it
and never *needs* it -- it reads whatever judgment is currently cached, and the
safety gate + reflex (safety.py, run.py) own safety regardless of what comes back.

Which model answers is not decided here: see providers.py (Laya local, Jev legacy).
"""
import asyncio, threading, time
import numpy as np

# --- how the flight code reacts to a judgment -------------------------------
THRESHOLDS = {
    "stale_after_s": 1.5,     # ignore a judgment older than this; the world moved on
    "risk_slow_down": 1.45,   # score above which we bleed speed regardless of maneuver
    "call_hz": 3.0,           # upper bound on how often we ask
    "call_budget": 160,      # hard cap per episode, so a bug cannot run up a bill
    "consult_within_m": 4.0,  # only ask when something is actually in the way...
    "consult_lost_s": 1.0,    # ...or the target has been missing this long
    "climb_steps": 150,       # ~3s: long enough to rise AND cross, not just bob up
    "commit_steps": 55,       # ~1.4s at 50Hz: commit to a maneuver instead of chattering
    "override_risk": 1.7,     # ...unless things get this dangerous, then re-decide now
    "really_lost": 0.5,       # Noul above which we stop trusting the remembered bearing
}

# The drone's own capabilities. Without this the model cannot know that "climb"
# is physically available, or what counts as a small height.
AIRCRAFT = {
    "type": "quadrotor, camera-only, no map and no GPS",
    "cruise_altitude_m": 1.6,
    "can_climb_to_m": 3.0,
    "climb_takes_about_s": 1.5,
    "top_speed_mps": 3.6,
    "note": "All distances are from a forward camera. 25 m means nothing was detected.",
}

MISSION = ("Follow the ground rover and keep it in view. Do not hit anything. "
           "Losing the rover briefly is acceptable; hitting an obstacle is not.")

MANEUVERS = {
    "hold_course": (
        "Nothing meaningfully blocks the pursuit line: the path ahead is clear for "
        "several metres. Keep flying straight at the target."),
    "gap_left": (
        "Something blocks the way ahead, and the left sectors show clearly more free "
        "space than the right. Steer around it to the left."),
    "gap_right": (
        "Something blocks the way ahead, and the right sectors show clearly more free "
        "space than the left. Steer around it to the right."),
    "climb": (
        "The obstruction ahead is LOW: free_ahead_above_m is much larger than "
        "free_ahead_level_m, so there is clear air over the top of it. This is the right "
        "answer when every sector is blocked, because that means there is no gap to "
        "steer through, but the thing is short enough to simply fly over."),
    "brake": (
        "Close to something on several sides and no option is clearly better. Bleed off "
        "speed and hold until the picture improves."),
    "reacquire": (
        "The target has been out of sight long enough that the remembered bearing is "
        "stale. Stop chasing it and sweep to find the target again."),
}

# Plain SystemOne wire format, so any provider can take it as-is. Byte-for-byte the
# same JSON the TypeSafe SDK used to send for the original Choice/Score/Noul objects.
QUESTIONS = {
    "maneuver": {
        "type": "choice",
        "instructions": {
            "role": "You are the tactical decision layer of an autonomous quadrotor.",
            "mission": MISSION,
            "ask": "Which single maneuver should the drone commit to right now?",
        },
        "criteria": MANEUVERS,
    },
    "risk": {
        "type": "score",
        "instructions": "How dangerous is the drone's immediate situation?",
        "criteria": ["clear and open", "tight but manageable", "about to hit something"],
    },
    "target_truly_lost": {
        "type": "noul",
        "instructions": (
            "Has the drone genuinely lost the rover? Judge from unseen_for_s: a fraction "
            "of a second behind a pillar is a normal occlusion, several seconds of nothing "
            "in an open scene means the chase line is stale."),
        "criteria": {"true": "Give up the remembered bearing and sweep to search.",
                     "false": "Keep flying the last known bearing; it should reappear."},
    },
}


def decision_needed(scene):
    """Code decides WHEN there is a judgment worth paying for. On an empty corridor
    with the target in view there is nothing to decide, so we do not ask."""
    return (scene["nearest_obstacle_m"] < THRESHOLDS["consult_within_m"]
            or scene["sectors_blocked"] >= 1
            or (scene["target"]["unseen_for_s"] or 0.0) >= THRESHOLDS["consult_lost_s"])


def build_state(scene):
    """Everything the model is allowed to know, and nothing it cannot observe."""
    return {"mission": MISSION, "aircraft": AIRCRAFT, "observed": scene}


DEFAULT = {"maneuver": "hold_course", "risk": 0.0, "confidence": 0.0,
           "target_truly_lost": 0.0, "source": "default", "age_s": 0.0,
           "probabilities": {}, "from_model": False, "decision_id": None}


def to_judgment(resp, source):
    """Typed answers -> the judgment dict guidance reads. Provider-agnostic."""
    a = resp.answers
    return {
        "maneuver": a["maneuver"].choice,
        "confidence": round(a["maneuver"].confidence or 0.0, 3),
        "probabilities": {k: round(v, 3) for k, v in a["maneuver"].probabilities.items()},
        "risk": round(a["risk"].score, 2),
        "target_truly_lost": round(a["target_truly_lost"].noul, 3),
        "source": source,
        "from_model": True,
    }


def _pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if xs else None


class Tactician:
    """Asks a DecisionProvider for a judgment at most `hz` times a second, and only
    when the scene has actually changed enough to be worth a call.

    Timing is honest: a judgment becomes visible to guidance only when the answer
    actually comes back. At most one request is in flight and at most one is waiting;
    a newer scene replaces a waiting one (latest-state-wins), so there is no queue to
    build up. An answer older than `deadline_s` (state -> available, sim time) is
    discarded as late and the previous judgment keeps ageing.

    `now` is the simulation clock, advanced by the episode loop; all per-decision
    timestamps are in sim time, inference latency is wall-clock.
    """

    def __init__(self, provider, hz=THRESHOLDS["call_hz"], budget=THRESHOLDS["call_budget"],
                 deadline_s=None, timeout_s=2.0):
        self.provider = provider
        self.model = f"{provider.name}:{provider.model}"
        self.min_dt = 1.0 / hz
        self.budget = budget                      # None = unlimited
        self.deadline_s = deadline_s
        self.timeout_s = timeout_s
        self.now = 0.0
        self.calls = 0
        self.attempts = 0
        self.skipped = 0
        self.errors = 0
        self.n_offer = 0
        self.n_ratelimited = 0
        self.n_superseded = 0
        self.n_late = 0
        self.last_error = None
        self.tokens = 0
        self.latency = []                         # wall-clock seconds per completed call
        self.records = []                         # one dict per requested decision
        self.veto_reasons = {}
        self._latest = dict(DEFAULT)
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self._pending = None                      # (decision_id, scene, t_state)
        self._last_sent = float("-inf")
        self._last_key = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @staticmethod
    def _key(scene):
        """Coarse fingerprint: only re-ask when the situation is materially new."""
        t = scene["target"]
        return (
            scene["sectors_blocked"],
            min(int(scene["free_ahead_above_m"] / 3.0), 8),
            tuple(min(int(v / 1.5), 6) for v in scene["sector_range_m"].values()),
            min(int(scene["nearest_obstacle_m"] / 1.5), 6),
            t["visible"],
            None if t["bearing_deg"] is None else int(t["bearing_deg"] / 15),
            (t["unseen_for_s"] or 0) > 2.0,
        )

    def offer(self, scene, now):
        """Non-blocking. Hand the latest scene over if it's worth a call."""
        self.n_offer += 1
        if (self.budget is not None and self.attempts >= self.budget) or now - self._last_sent < self.min_dt:
            self.n_ratelimited += 1
            return
        key = self._key(scene)
        if key == self._last_key:
            self.skipped += 1
            return
        with self._lock:
            did = len(self.records)
            self.records.append({"decision_id": did, "timestamp": round(now, 3), "status": "pending",
                                 "safety_veto": False, "safety_veto_reason": None})
        with self._cv:
            if self._pending is not None:         # latest-state-wins
                self.records[self._pending[0]]["status"] = "superseded"
                self.n_superseded += 1
            self._pending = (did, scene, now)
            self._cv.notify()
        self._last_sent, self._last_key = now, key

    def read(self, now):
        with self._lock:
            out = dict(self._latest)
        out["age_s"] = round(now - self._stamp, 2)
        return out

    def mark_consumed(self, decision_id, now):
        """Guidance has acted on this judgment for the first time."""
        if decision_id is None:
            return
        rec = self.records[decision_id]
        if "end_to_end_latency_ms" not in rec:
            rec["end_to_end_latency_ms"] = round(1000 * (now - rec["timestamp"]), 1)

    def mark_veto(self, decision_id, reason):
        if decision_id is None:
            return
        rec = self.records[decision_id]
        if not rec["safety_veto"]:
            rec["safety_veto"], rec["safety_veto_reason"] = True, reason
            self.veto_reasons[reason] = self.veto_reasons.get(reason, 0) + 1

    def _take(self):
        with self._cv:
            while self._pending is None and not self._stop.is_set():
                self._cv.wait(timeout=0.2)
            job, self._pending = self._pending, None
            return job

    def _worker(self):
        loop = asyncio.new_event_loop()
        try:
            while not self._stop.is_set():
                job = self._take()
                if job is None:
                    continue
                did, scene, t_state = job
                rec = self.records[did]
                self.attempts += 1  # counts against the budget whether or not the call succeeds
                t0 = time.perf_counter()
                try:
                    r = loop.run_until_complete(asyncio.wait_for(
                        self.provider.predict(build_state(scene), QUESTIONS), self.timeout_s))
                    lat = time.perf_counter() - t0
                    judgment = to_judgment(r, self.provider.name)
                    judgment["decision_id"] = did
                    age = self.now - t_state
                    self.tokens += r.input_tokens + r.output_tokens
                    self.latency.append(lat)
                    rec.update(inference_latency_ms=round(1000 * lat, 1),
                               state_age_ms=round(1000 * age, 1),
                               maneuver=judgment["maneuver"],
                               maneuver_probability=judgment["probabilities"].get(judgment["maneuver"]),
                               confidence=judgment["confidence"],
                               risk=judgment["risk"],
                               target_lost_probability=judgment["target_truly_lost"])
                    if self.deadline_s is not None and age > self.deadline_s:
                        rec["status"] = "late"             # missed the onboard budget
                        self.n_late += 1
                        continue
                    rec["status"] = "completed"
                    self.calls += 1
                    with self._lock:
                        self._latest, self._stamp = judgment, t_state
                except Exception as e:                    # degrade, never crash the flight
                    self.errors += 1
                    self.last_error = f"{type(e).__name__}: {e}"[:160]
                    rec["status"] = "error"
                    rec["error"] = self.last_error
                    # A failed call replaces the cached judgment with an error placeholder, so the
                    # next offer() for this same scene must not be skipped as "unchanged" - otherwise
                    # a static scene never gets re-asked and never recovers a real judgment.
                    self._last_key = None
                    with self._lock:
                        self._latest = dict(DEFAULT, source=f"error:{type(e).__name__}")
        finally:
            try:
                loop.run_until_complete(self.provider.aclose())
            except Exception:
                pass
            loop.close()

    def close(self):
        self._stop.set()
        with self._cv:
            self._cv.notify()
        self._thread.join(timeout=self.timeout_s + 1.0)

    def stats(self):
        lat_ms = [1000 * x for x in self.latency]
        done = [r for r in self.records if r["status"] == "completed"]
        e2e = [r["end_to_end_latency_ms"] for r in done if "end_to_end_latency_ms" in r]
        age = [r["state_age_ms"] for r in done]
        conf = [r["maneuver_probability"] for r in done if r.get("maneuver_probability") is not None]
        vetoed = sum(r["safety_veto"] for r in self.records)
        return {"provider": self.provider.name, "model": self.provider.model,
                "decisions_requested": len(self.records),
                "decisions_completed": self.calls,
                "decisions_dropped": self.n_superseded + self.n_late + self.errors,
                "dropped_superseded": self.n_superseded, "dropped_late": self.n_late,
                "errors": self.errors, "last_error": self.last_error,
                "skipped_unchanged": self.skipped, "rate_limited": self.n_ratelimited,
                "offers": self.n_offer, "tokens": self.tokens,
                "latency_p50_ms": _pct(lat_ms, 50), "latency_p90_ms": _pct(lat_ms, 90),
                "latency_p99_ms": _pct(lat_ms, 99),
                "state_age_p50_ms": _pct(age, 50), "state_age_p99_ms": _pct(age, 99),
                "end_to_end_p50_ms": _pct(e2e, 50), "end_to_end_p99_ms": _pct(e2e, 99),
                "maneuver_probability_mean": round(float(np.mean(conf)), 3) if conf else None,
                "unsafe_decisions_proposed": vetoed, "unsafe_decisions_vetoed": vetoed,
                "veto_reasons": dict(self.veto_reasons)}
