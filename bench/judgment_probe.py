"""Hand-built scenes with a known right answer, asked through the real provider path.

Mirrors the README's "The judgments themselves are good" table, so any decision
provider can be checked against the same seven situations before it flies.

    .venv/bin/python bench/judgment_probe.py --model laya-english --repeat 20
"""
import argparse, asyncio, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from providers import make_provider
from tactics import QUESTIONS, build_state, to_judgment


def scene(sec, blocked, above, level, nearest, vis=True, brg=0.0, rng=6.0, unseen=None):
    names = ("far_left", "left", "center", "right", "far_right")
    s = dict(zip(names, sec))
    return {"sector_range_m": s, "sectors_blocked": blocked, "path_ahead_m": min(s["left"], s["center"], s["right"]),
            "free_ahead_above_m": above, "free_ahead_level_m": level, "free_ahead_below_m": level,
            "room_above_m": 3.0, "room_below_m": 1.2, "room_left_m": 5.0, "room_right_m": 5.0,
            "nearest_obstacle_m": nearest, "nearest_bearing_deg": 0.0,
            "target": ({"visible": True, "bearing_deg": brg, "range_m": rng, "pixels": 120, "unseen_for_s": 0.0} if vis else
                       {"visible": False, "bearing_deg": None, "range_m": None, "pixels": 0, "unseen_for_s": unseen})}


CASES = [  # (label, scene, expected maneuver or None, check)
    ("LOW BARRIER, all 5 blocked, clear above", scene((2.4, 2.3, 2.2, 2.3, 2.4), 5, 25.0, 2.2, 2.2, vis=False, unseen=1.5), "climb", None),
    ("TALL PILLAR ahead, right wide open", scene((2.5, 2.2, 2.0, 12.0, 25.0), 3, 2.0, 2.0, 2.0), "gap_right", None),
    ("TALL PILLAR ahead, left wide open", scene((25.0, 12.0, 2.0, 2.2, 2.5), 3, 2.0, 2.0, 2.0), "gap_left", None),
    ("TARGET GONE 7s, wide open", scene((25.0, 25.0, 25.0, 25.0, 25.0), 0, 25.0, 25.0, 25.0, vis=False, unseen=7.0), "reacquire", ("lost", ">", 0.5)),
    ("BOXED IN, close on all sides, tall", scene((1.4, 1.3, 1.2, 1.3, 1.4), 5, 1.2, 1.2, 1.2), "brake", None),
    ("ALL CLEAR, target dead ahead", scene((25.0, 25.0, 25.0, 25.0, 25.0), 0, 25.0, 25.0, 25.0), "hold_course", None),
    ("BRIEF OCCLUSION 0.6s, path clear", scene((25.0, 25.0, 25.0, 25.0, 25.0), 0, 25.0, 25.0, 25.0, vis=False, unseen=0.6), None, ("lost", "<", 0.5)),
]


async def main(a):
    prov = make_provider(a.provider, **({"model": a.model} if a.model else {}))
    lat, ok, toks = [], 0, []
    for label, sc, want, check in CASES:
        for i in range(a.repeat):
            t0 = time.perf_counter()
            r = await prov.predict(build_state(sc), QUESTIONS)
            lat.append(time.perf_counter() - t0)
        toks.append(r.input_tokens)
        j = to_judgment(r, prov.name)
        good = (want is None or j["maneuver"] == want)
        if check:
            good = good and (j["target_truly_lost"] > check[2] if check[1] == ">" else j["target_truly_lost"] < check[2])
        ok += good
        print(f"{'OK ' if good else 'BAD'} {label:42s} -> {j['maneuver']:<11s} p={j['probabilities'][j['maneuver']]:.2f} "
              f"risk={j['risk']:.2f} lost={j['target_truly_lost']:.2f}   (want {want or 'lost<0.5'})")
    await prov.aclose()
    ms = np.array(lat) * 1000
    print(f"\n{prov.name}:{prov.model}  correct {ok}/{len(CASES)}   latency p50={np.percentile(ms,50):.1f}ms "
          f"p90={np.percentile(ms,90):.1f}ms p99={np.percentile(ms,99):.1f}ms (n={len(ms)})   input_tokens~{max(toks)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="laya")
    p.add_argument("--model", default=None)
    p.add_argument("--repeat", type=int, default=10)
    asyncio.run(main(p.parse_args()))
