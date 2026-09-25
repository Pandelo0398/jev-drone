"""Safety gate: the code-owned veto over any model's tactical recommendation.

These are the exact checks that used to live inline in run.Guidance, pulled out so
every veto has a name and can be counted. Thresholds are unchanged. The model can
propose; only what passes here reaches guidance, and the reflex below overrides
everything regardless of what was proposed.

  Laya / Jev  ->  tactical recommendation  ->  SafetyGate  ->  Guidance  ->  Pilot
"""

REFLEX_M = 2.2           # code-owned: below this, the model's opinion is irrelevant
CLIMB_MIN_BLOCKED = 4    # climb only when there is no lateral gap to take instead...
CLIMB_AIR_RATIO = 2.2    # ...and the air above is demonstrably clearer than at level


def judgment_usable(judg, stale_after_s):
    """(ok, reason). A judgment the gate will not even consider."""
    if not judg.get("from_model"):
        return False, "no_model_judgment"
    if judg["age_s"] >= stale_after_s:
        return False, "stale"
    return True, None


def climb_allowed(scene):
    """(ok, reason). Only go over it if there is demonstrably clear air up there.

    free_ahead_above_m is measured from the depth buffer above the aircraft's own
    height; if the obstruction's top is beyond the climb ceiling, that band is
    blocked too and the ratio test fails.
    """
    if scene["sectors_blocked"] < CLIMB_MIN_BLOCKED:
        return False, "climb_rejected:lateral_gap_available"
    if not scene["free_ahead_above_m"] > CLIMB_AIR_RATIO * scene["free_ahead_level_m"]:
        return False, "climb_rejected:no_clear_air_above"
    return True, None


def reflex_engaged(scene):
    """The hard reflex: something is in the flight path, inside REFLEX_M."""
    return scene["path_ahead_m"] < REFLEX_M
