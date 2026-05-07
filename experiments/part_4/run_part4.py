"""Part 4: Distance testing — how well do models generalize across shard populations.
Models trained on cluster_1_group_1 first shards, evaluated on:
  A) Same group, later shards (within-group distance)
  B) Same cluster, different group (cluster_1_group_2)
  C) Different cluster (cluster_3_group_1)
Also: compare a few policies on each group's native data.
"""
import csv, json, os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import make_lru_policy, make_ml_policy, make_opt_policy
from src.decay_policy import make_decay_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_BASE = os.path.join(PROJECT, "traces/workload2")
RESULT_DIR = os.path.join(PART_DIR, "results")
FIG_DIR = os.path.join(PART_DIR, "figures")
for d in [RESULT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

CSV_FIELDS = ["policy", "scenario", "train_group", "eval_group",
              "sat_shards", "eval_shards", "hit_rate", "wa",
              "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part4_results.csv")
_BATCH_ID = None

GROUPS = {
    "c1g1": os.path.join(TRACE_BASE, "cluster_1_group_1"),
    "c1g2": os.path.join(TRACE_BASE, "cluster_1_group_2"),
    "c3g1": os.path.join(TRACE_BASE, "cluster_3_group_1"),
}

def _ck_path():
    if _BATCH_ID is not None:
        return os.path.join(RESULT_DIR, f"checkpoint_batch{_BATCH_ID}.json")
    return os.path.join(RESULT_DIR, "checkpoint.json")

def _csv_path():
    if _BATCH_ID is not None:
        return os.path.join(RESULT_DIR, f"part4_batch{_BATCH_ID}.csv")
    return RESULTS_CSV

def load_checkpoint():
    p = _ck_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"completed": []}

def save_checkpoint(ck):
    with open(_ck_path(), "w") as f:
        json.dump(ck, f, indent=2)

def is_done(ck, key):
    return key in ck["completed"]

def mark_done(ck, key):
    ck["completed"].append(key)
    save_checkpoint(ck)

def append_result(row):
    path = _csv_path()
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def get_shards(group_key):
    d = GROUPS[group_key]
    return sorted(os.listdir(d))

def shard_path(group_key, idx):
    shards = get_shards(group_key)
    return os.path.join(GROUPS[group_key], shards[idx])

def load_shard_range(group_key, start, end):
    reqs = []
    shards = get_shards(group_key)
    for i in range(start, min(end, len(shards))):
        reqs.extend(load_thesios_csv(os.path.join(GROUPS[group_key], shards[i])))
    reqs.sort(key=lambda r: r.arrival_time)
    return reqs

def load_eval_with_saturation(group_key, sat_start, sat_end, eval_start, eval_end):
    reqs = load_shard_range(group_key, sat_start, eval_end)
    sat_reqs = load_shard_range(group_key, sat_start, sat_end)
    warmup_n = len(sat_reqs)
    for r in reqs[:warmup_n]:
        r.is_warmup = True
    return reqs, warmup_n

def run_eval(name, pol, reqs, warmup_n, scenario="", train_group="", eval_group="",
             sat_shards="", eval_shards_str=""):
    cfg = EvaluatorConfig(
        FAST_CFG, SLOW_CFG, shard_path("c1g1", 0), "/dev/null",
        warmup_ops=warmup_n, always_write_slow=True, on_fast_full="spill",
    )
    t0 = time.perf_counter()
    mc = Evaluator(cfg, pol, reqs).run()
    elapsed = time.perf_counter() - t0
    s = mc.global_metrics.summary()
    wa = mc.global_metrics.write_amplification(mc._requested_write_bytes)
    row = {
        "policy": name, "scenario": scenario,
        "train_group": train_group, "eval_group": eval_group,
        "sat_shards": sat_shards, "eval_shards": eval_shards_str,
        "hit_rate": round(s["hit_rate"], 6), "wa": round(wa, 4),
        "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
        "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
        "wall_s": round(elapsed, 1),
    }
    print(f"  {name:<35s} [{scenario}] hit={row['hit_rate']:.4f}  mean={row['mean_resp_ms']:.3f}ms  ({elapsed:.0f}s)")
    return row

def get_model_path(model_type):
    part2 = os.path.join(PROJECT, "experiments/part_2/models")
    part1 = os.path.join(PROJECT, "experiments/part_1/models")
    paths = {
        "ML_Reg": os.path.join(part2, "ML_Reg_t1000_d10_l127.joblib"),
        "ML_Rank": os.path.join(part2, "ML_Rank_t500_d8_l63.joblib"),
        "OPT_Imit_Reg": os.path.join(part2, "OPT_Reg_t500_d6_l31.joblib"),
        "OPT_Imit_Rank": os.path.join(part2, "OPT_Rank_t100_d4_l15.joblib"),
    }
    p = paths.get(model_type)
    if p and not os.path.exists(p):
        fallbacks = {
            "ML_Reg": os.path.join(part1, "ml_reg_4shards.joblib"),
            "ML_Rank": os.path.join(part1, "ml_rank_4shards.joblib"),
            "OPT_Imit_Reg": os.path.join(part1, "opt_reg_1shard.joblib"),
            "OPT_Imit_Rank": os.path.join(part1, "opt_rank_10k.joblib"),
        }
        p = fallbacks.get(model_type)
    return p

SCENARIOS = [
    {
        "name": "same_group_far",
        "desc": "Same group, far shards",
        "eval_group": "c1g1",
        "sat_start": 46, "sat_end": 48,
        "eval_start": 46, "eval_end": 48,
        "warmup_frac": True,
    },
    {
        "name": "diff_group",
        "desc": "Same cluster, different group",
        "eval_group": "c1g2",
        "sat_start": 0, "sat_end": 2,
        "eval_start": 0, "eval_end": 2,
        "warmup_frac": True,
    },
    {
        "name": "diff_cluster",
        "desc": "Different cluster",
        "eval_group": "c3g1",
        "sat_start": 0, "sat_end": 2,
        "eval_start": 0, "eval_end": 2,
        "warmup_frac": True,
    },
]

POLICIES = ["LRU", "Decay", "ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank"]


def run_scenario(ck, scenario):
    s = scenario
    print(f"\n--- Scenario: {s['desc']} (eval on {s['eval_group']}) ---")
    eg = s["eval_group"]
    n_shards = len(get_shards(eg))
    actual_eval_end = min(s["eval_end"], n_shards)

    if s.get("warmup_frac"):
        reqs = load_shard_range(eg, s["eval_start"], actual_eval_end)
        warmup_n = len(reqs) // 2
        for r in reqs[:warmup_n]:
            r.is_warmup = True
    else:
        actual_sat_end = min(s["sat_end"], n_shards)
        reqs, warmup_n = load_eval_with_saturation(eg, s["sat_start"], actual_sat_end,
                                                     s["eval_start"], actual_eval_end)
    print(f"  {len(reqs)} requests, {warmup_n} warmup")
    sat_str = f"{s.get('sat_start', s['eval_start'])}-{s.get('sat_end', s['eval_end'])-1}"
    eval_str = f"{s['eval_start']}-{actual_eval_end-1}"

    for pol_name in POLICIES:
        key = f"{s['name']}_{pol_name}"
        if is_done(ck, key):
            print(f"  {pol_name} already done.")
            continue

        if pol_name == "LRU":
            pol = make_lru_policy(FAST_CAP)
        elif pol_name == "Decay":
            pol = make_decay_policy(FAST_CAP, alpha=0.5, decay_rate=0.05,
                                     fast_fill_threshold=0.9)
        else:
            mp = get_model_path(pol_name)
            if not mp or not os.path.exists(mp):
                print(f"  {pol_name}: model not found, skipping")
                continue
            pol = make_ml_policy(FAST_CAP, mp, fast_fill_threshold=0.9)

        row = run_eval(pol_name, pol, reqs, warmup_n,
                      scenario=s["name"], train_group="c1g1", eval_group=eg,
                      sat_shards=sat_str, eval_shards_str=eval_str)
        append_result(row)
        mark_done(ck, key)


def make_plots():
    import glob as _glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = []
    seen = set()
    for f in sorted(_glob.glob(os.path.join(RESULT_DIR, "part4_batch*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                key = (r["policy"], r["scenario"])
                if key not in seen:
                    all_rows.append(r)
                    seen.add(key)
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as fh:
            for r in csv.DictReader(fh):
                key = (r["policy"], r["scenario"])
                if key not in seen:
                    all_rows.append(r)
                    seen.add(key)
    with open(RESULTS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Merged {len(all_rows)} results")

    for r in all_rows:
        for k in ["hit_rate", "wa", "mean_resp_ms", "p99_resp_ms"]:
            r[k] = float(r[k])

    scenarios = ["same_group_far", "diff_group", "diff_cluster"]
    scenario_labels = ["Same Group\n(far shards)", "Diff Group\n(same cluster)", "Diff Cluster"]
    policies = POLICIES
    colors = {"LRU": "#333333", "Decay": "#666666",
              "ML_Reg": "#1f77b4", "ML_Rank": "#ff7f0e",
              "OPT_Imit_Reg": "#2ca02c", "OPT_Imit_Rank": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, (metric, title) in enumerate([("hit_rate", "Hit Rate"),
                                                ("mean_resp_ms", "Mean Resp (ms)")]):
        ax = axes[ax_idx]
        x = np.arange(len(scenarios))
        width = 0.12
        for i, pol in enumerate(policies):
            vals = []
            for sc in scenarios:
                match = [r for r in all_rows if r["policy"] == pol and r["scenario"] == sc]
                vals.append(match[0][metric] if match else 0)
            ax.bar(x + i * width, vals, width, label=pol, color=colors.get(pol, "gray"))
        ax.set_xticks(x + width * (len(policies) - 1) / 2)
        ax.set_xticklabels(scenario_labels)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Part 4: Distance Testing — Model Generalization (10GB)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part4_distance_testing.png"), dpi=150)
    print(f"Saved part4_distance_testing.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["same_group_far", "diff_group", "diff_cluster",
                                                "all", "plot"], default="all")
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()

    if args.batch is not None:
        _BATCH_ID = args.batch

    if args.scenario == "plot":
        make_plots()
        sys.exit(0)

    ck = load_checkpoint()

    if args.scenario == "all":
        for s in SCENARIOS:
            run_scenario(ck, s)
    else:
        match = [s for s in SCENARIOS if s["name"] == args.scenario]
        if match:
            run_scenario(ck, match[0])

    print("\nDone!")
