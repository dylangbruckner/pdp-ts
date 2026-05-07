"""Part 6: Configuration toggle effects.
Test how always_write_slow and on_fast_full affect LRU, Decay, OPT Imit Rank, ML Reg.
"""
import csv, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import make_lru_policy, make_ml_policy
from src.decay_policy import make_decay_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))
RESULT_DIR = os.path.join(PART_DIR, "results")
FIG_DIR = os.path.join(PART_DIR, "figures")

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

CSV_FIELDS = ["policy", "always_write_slow", "on_fast_full",
              "hit_rate", "wa", "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part6_results.csv")
CHECKPOINT = os.path.join(RESULT_DIR, "checkpoint.json")

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

def shard_path(idx):
    return os.path.join(TRACE_DIR, SHARDS[idx])

def load_eval_data():
    reqs = []
    for i in range(4, 6):
        reqs.extend(load_thesios_csv(shard_path(i)))
    reqs.sort(key=lambda r: r.arrival_time)
    warmup_n = len(reqs) // 2
    for r in reqs[:warmup_n]:
        r.is_warmup = True
    return reqs, warmup_n

def get_model_path(name):
    part2 = os.path.join(PROJECT, "experiments/part_2/models")
    paths = {
        "OPT_Imit_Rank": os.path.join(part2, "OPT_Rank_t100_d4_l15.joblib"),
        "ML_Reg": os.path.join(part2, "ML_Reg_t1000_d10_l127.joblib"),
    }
    return paths.get(name)

CONFIGS = [
    {"always_write_slow": True,  "on_fast_full": "spill"},  # default
    {"always_write_slow": True,  "on_fast_full": "wait"},
    {"always_write_slow": False, "on_fast_full": "spill"},
    {"always_write_slow": False, "on_fast_full": "wait"},
]

POLICIES = ["LRU", "Decay", "OPT_Imit_Rank", "ML_Reg"]

if __name__ == "__main__":
    ck = load_checkpoint()
    reqs, warmup_n = load_eval_data()
    print(f"Eval: {len(reqs)} requests, {warmup_n} warmup\n")

    for cfg in CONFIGS:
        aws = cfg["always_write_slow"]
        off = cfg["on_fast_full"]
        cfg_label = f"aws={aws}_off={off}"
        print(f"\n--- Config: always_write_slow={aws}, on_fast_full={off} ---")

        for pol_name in POLICIES:
            key = f"{pol_name}_{cfg_label}"
            if is_done(ck, key):
                print(f"  {pol_name} already done.")
                continue

            if pol_name == "LRU":
                pol = make_lru_policy(FAST_CAP)
            elif pol_name == "Decay":
                pol = make_decay_policy(FAST_CAP, alpha=0.5, decay_rate=0.05,
                                         fast_fill_threshold=1.0)
            else:
                mp = get_model_path(pol_name)
                if not mp or not os.path.exists(mp):
                    print(f"  {pol_name}: no model")
                    continue
                pol = make_ml_policy(FAST_CAP, mp, fast_fill_threshold=1.0)

            eval_cfg = EvaluatorConfig(
                FAST_CFG, SLOW_CFG, shard_path(0), "/dev/null",
                warmup_ops=warmup_n, always_write_slow=aws, on_fast_full=off,
            )
            t0 = time.perf_counter()
            mc = Evaluator(eval_cfg, pol, reqs).run()
            elapsed = time.perf_counter() - t0
            s = mc.global_metrics.summary()
            wa = mc.global_metrics.write_amplification(mc._requested_write_bytes)
            row = {
                "policy": pol_name,
                "always_write_slow": str(aws),
                "on_fast_full": off,
                "hit_rate": round(s["hit_rate"], 6),
                "wa": round(wa, 4),
                "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
                "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
                "wall_s": round(elapsed, 1),
            }
            print(f"  {pol_name:<20s} hit={row['hit_rate']:.4f}  WA={row['wa']:.3f}  "
                  f"mean={row['mean_resp_ms']:.3f}ms  ({elapsed:.0f}s)")
            append_result(row)
            mark_done(ck, key)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            for k in ["hit_rate", "wa", "mean_resp_ms", "p99_resp_ms"]:
                r[k] = float(r[k])
            rows.append(r)

    cfg_labels = ["aws=T\nspill", "aws=T\nwait", "aws=F\nspill", "aws=F\nwait"]
    cfg_keys = [("True", "spill"), ("True", "wait"), ("False", "spill"), ("False", "wait")]
    colors = {"LRU": "#333333", "Decay": "#666666",
              "ML_Reg": "#1f77b4", "OPT_Imit_Rank": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [("hit_rate", "Hit Rate"), ("wa", "Write Amplification"),
               ("mean_resp_ms", "Mean Response Time (ms)"),
               ("p99_resp_ms", "P99 Response Time (ms)")]

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes.flatten()[ax_idx]
        x = np.arange(len(cfg_labels))
        width = 0.18
        for i, pol in enumerate(POLICIES):
            vals = []
            for aws, off in cfg_keys:
                match = [r for r in rows if r["policy"] == pol
                         and r["always_write_slow"] == aws and r["on_fast_full"] == off]
                vals.append(match[0][metric] if match else 0)
            ax.bar(x + i * width, vals, width, label=pol, color=colors[pol])
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(cfg_labels, fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Part 6: Configuration Toggle Effects (10GB)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part6_config_toggles.png"), dpi=150)
    print(f"\nSaved part6_config_toggles.png")
    print("\nDone!")
