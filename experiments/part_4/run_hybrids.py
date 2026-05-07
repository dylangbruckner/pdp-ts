"""Part 4 supplement: run best 2 hybrids across all scenarios."""
import csv, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import make_hybrid_policy, make_ml_policy
from src.decay_policy import make_decay_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_BASE = os.path.join(PROJECT, "traces/workload2")
RESULT_DIR = os.path.join(PART_DIR, "results")

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

CSV_FIELDS = ["policy", "scenario", "train_group", "eval_group",
              "sat_shards", "eval_shards", "hit_rate", "wa",
              "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part4_hybrids.csv")
CHECKPOINT = os.path.join(RESULT_DIR, "checkpoint_hybrids.json")

GROUPS = {
    "c1g1": os.path.join(TRACE_BASE, "cluster_1_group_1"),
    "c1g2": os.path.join(TRACE_BASE, "cluster_1_group_2"),
    "c3g1": os.path.join(TRACE_BASE, "cluster_3_group_1"),
}

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"completed": []}

def save_checkpoint(ck):
    with open(CHECKPOINT, "w") as f:
        json.dump(ck, f, indent=2)

def is_done(ck, key):
    return key in ck["completed"]

def mark_done(ck, key):
    ck["completed"].append(key)
    save_checkpoint(ck)

def append_result(row):
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def get_shards(gk):
    return sorted(os.listdir(GROUPS[gk]))

def load_shard_range(gk, start, end):
    shards = get_shards(gk)
    reqs = []
    for i in range(start, min(end, len(shards))):
        reqs.extend(load_thesios_csv(os.path.join(GROUPS[gk], shards[i])))
    reqs.sort(key=lambda r: r.arrival_time)
    return reqs

def run_eval(name, pol, reqs, warmup_n, scenario, train_group, eval_group, shards_str):
    cfg = EvaluatorConfig(
        FAST_CFG, SLOW_CFG, os.path.join(GROUPS["c1g1"], get_shards("c1g1")[0]),
        "/dev/null", warmup_ops=warmup_n, always_write_slow=True, on_fast_full="spill",
    )
    t0 = time.perf_counter()
    mc = Evaluator(cfg, pol, reqs).run()
    elapsed = time.perf_counter() - t0
    s = mc.global_metrics.summary()
    wa = mc.global_metrics.write_amplification(mc._requested_write_bytes)
    row = {
        "policy": name, "scenario": scenario,
        "train_group": train_group, "eval_group": eval_group,
        "sat_shards": shards_str, "eval_shards": shards_str,
        "hit_rate": round(s["hit_rate"], 6), "wa": round(wa, 4),
        "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
        "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
        "wall_s": round(elapsed, 1),
    }
    print(f"  {name:<35s} [{scenario}] hit={row['hit_rate']:.4f}  ({elapsed:.0f}s)")
    return row

part2 = os.path.join(PROJECT, "experiments/part_2/models")
HYBRIDS = [
    ("Hybrid_ML_Reg_p25", os.path.join(part2, "ML_Reg_t1000_d10_l127.joblib"), 0.25),
    ("Hybrid_OPT_Rank_p10", os.path.join(part2, "OPT_Rank_t100_d4_l15.joblib"), 0.10),
]

SCENARIOS = [
    ("same_group_far", "c1g1", "c1g1", 46, 48),
    ("diff_group", "c1g1", "c1g2", 0, 2),
    ("diff_cluster", "c1g1", "c3g1", 0, 2),
    ("baseline", "c1g1", "c1g1", 4, 6),
]

if __name__ == "__main__":
    ck = load_checkpoint()

    for sc_name, train_grp, eval_grp, start, end in SCENARIOS:
        reqs = load_shard_range(eval_grp, start, end)
        warmup_n = len(reqs) // 2
        for r in reqs[:warmup_n]:
            r.is_warmup = True
        shards_str = f"{start}-{end-1}"
        print(f"\n--- {sc_name}: {eval_grp} shards {shards_str} ({len(reqs)} reqs) ---")

        for name, model_path, pct in HYBRIDS:
            key = f"{sc_name}_{name}"
            if is_done(ck, key):
                print(f"  {name} already done.")
                continue
            if not os.path.exists(model_path):
                print(f"  {name}: model not found")
                continue
            pol = make_hybrid_policy(FAST_CAP, model_path,
                                     protect_percentile=pct, fast_fill_threshold=0.9)
            row = run_eval(name, pol, reqs, warmup_n, sc_name, train_grp, eval_grp, shards_str)
            append_result(row)
            mark_done(ck, key)

    print("\nDone!")
