"""Aggregate a benchmark directory written by bench/run_benchmark.sh.

    .venv/bin/python bench/summarize.py runs/bench
"""
import glob, json, os, sys
from collections import Counter

import numpy as np


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.startswith("{")]


def ms(xs):
    return f"{np.mean(xs):.2f} ± {np.std(xs):.2f}" if xs else "-"


def flight_table(runs_by_cfg):
    rows = [("runs", lambda r: None),
            ("course completed", lambda r: r["course_completed"]),
            ("course completion (stations/5)", lambda r: r["course_completion"]),
            ("max x (m)", lambda r: r["max_x_m"]),
            ("distance flown (m)", lambda r: r["distance_flown_m"]),
            ("collisions", lambda r: r["collisions"]),
            ("runs with >=1 collision", lambda r: r["collisions"] > 0),
            ("target visible (%)", lambda r: r["target_visible_pct"]),
            ("time in safety reflex (%)", lambda r: r["steps_reflex_pct"]),
            ("model acted (% of guidance ticks)", lambda r: r["steps_jev_acted_pct"]),
            ("crashed", lambda r: r["crashed_at_s"] is not None)]
    names = list(runs_by_cfg)
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for label, f in rows:
        cells = []
        for n in names:
            rs = runs_by_cfg[n]
            if label == "runs":
                cells.append(str(len(rs)))
            elif label in ("course completed", "runs with >=1 collision", "crashed"):
                cells.append(f"{sum(bool(f(r)) for r in rs)}/{len(rs)}")
            else:
                cells.append(ms([float(f(r)) for r in rs]))
        print(f"| {label} | " + " | ".join(cells) + " |")


def per_seed(runs_by_cfg):
    print("\n| seed | " + " | ".join(f"{n} max_x / stations / coll" for n in runs_by_cfg) + " |")
    print("|---|" + "---|" * len(runs_by_cfg))
    seeds = sorted({r["seed"] for rs in runs_by_cfg.values() for r in rs})
    for s in seeds:
        cells = []
        for rs in runs_by_cfg.values():
            r = next((r for r in rs if r["seed"] == s), None)
            cells.append(f"{r['max_x_m']} / {len(r['stations_cleared'])} / {r['collisions']}" if r else "-")
        print(f"| {s} | " + " | ".join(cells) + " |")


def decision_section(out_dir, runs):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "decisions", "*.jsonl"))):
        recs += load(p)
    done = [r for r in recs if r["status"] == "completed"]
    lat = [r["inference_latency_ms"] for r in recs if "inference_latency_ms" in r]
    e2e = [r["end_to_end_latency_ms"] for r in done if "end_to_end_latency_ms" in r]
    age = [r["state_age_ms"] for r in done]
    status = Counter(r["status"] for r in recs)
    mv = Counter(r["maneuver"] for r in done)
    vetoes = Counter(r["safety_veto_reason"] for r in recs if r["safety_veto"])
    pct = lambda xs, q: f"{np.percentile(xs, q):.1f}" if xs else "-"
    print(f"\nDecisions (all seeds pooled): requested {len(recs)}, "
          + ", ".join(f"{k} {v}" for k, v in status.most_common()))
    print(f"inference latency ms  p50 {pct(lat,50)}  p90 {pct(lat,90)}  p99 {pct(lat,99)}  max {max(lat):.1f}" if lat else "")
    print(f"state age at arrival  p50 {pct(age,50)}  p90 {pct(age,90)}  p99 {pct(age,99)}")
    print(f"end-to-end (state -> first used by guidance) ms  p50 {pct(e2e,50)}  p90 {pct(e2e,90)}  p99 {pct(e2e,99)}")
    probs = [r["maneuver_probability"] for r in done]
    print(f"maneuver probability  mean {np.mean(probs):.3f}  p10 {np.percentile(probs,10):.3f}  p90 {np.percentile(probs,90):.3f}")
    print(f"risk  mean {np.mean([r['risk'] for r in done]):.2f}  range [{min(r['risk'] for r in done):.2f}, {max(r['risk'] for r in done):.2f}]")
    lost = [r["target_lost_probability"] for r in done]
    print(f"target_lost  mean {np.mean(lost):.3f}  range [{min(lost):.3f}, {max(lost):.3f}]  >=0.5: {sum(x >= 0.5 for x in lost)}/{len(lost)}")
    print("maneuver distribution: " + ", ".join(f"{k} {v} ({100*v/len(done):.0f}%)" for k, v in mv.most_common()))
    n_done = max(1, len(done))
    print(f"safety vetoes: {sum(vetoes.values())} decisions ({100*sum(vetoes.values())/n_done:.1f}% of completed): {dict(vetoes)}")
    res = [r["resources"] for r in runs if r.get("resources")]
    if res:
        g = lambda k, f=max: f([x[k] for x in res if k in x]) if any(k in x for x in res) else None
        print(f"\nResources (onboard container, across runs): RAM max {g('container_ram_mb_max')} MiB, "
              f"CPU mean {np.mean([x['container_cpu_cores_mean'] for x in res]):.2f} cores "
              f"(p99 max {g('container_cpu_cores_p99')}), GPU mem device max {g('gpu_mem_mb_max')} MiB, "
              f"model share ~{g('gpu_mem_model_mb')} MiB, GPU util mean "
              f"{np.mean([x['gpu_util_pct_mean'] for x in res]):.1f}%")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/bench"
    runs = {n: load(os.path.join(out, f"{n}.jsonl"))
            for n in ("heuristic", "laya", "jev") if os.path.exists(os.path.join(out, f"{n}.jsonl"))}
    flight_table(runs)
    per_seed(runs)
    if runs.get("laya"):
        decision_section(out, runs["laya"])
