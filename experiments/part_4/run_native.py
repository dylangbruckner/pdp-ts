"""Part 4 supplement: train models natively on each group and compare to cross-trained.
Also adds baseline scenario (c1g1 shards 4-5) for reference."""
import csv, json, os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import joblib

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.ml_training import generate_training_data, train_model, train_ranking_model
from src.opt_training import generate_imitation_data, train_regressor, train_ranker
from src.policy import make_lru_policy, make_ml_policy
from src.decay_policy import make_decay_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_BASE = os.path.join(PROJECT, "traces/workload2")
MODEL_DIR = os.path.join(PART_DIR, "models")
RESULT_DIR = os.path.join(PART_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

CSV_FIELDS = ["policy", "scenario", "train_group", "eval_group",
              "sat_shards", "eval_shards", "hit_rate", "wa",
              "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part4_native.csv")
CHECKPOINT = os.path.join(RESULT_DIR, "checkpoint_native.json")

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

def run_eval(name, pol, reqs, warmup_n, scenario="", train_group="", eval_group="", shards_str=""):
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


def train_native_models(ck, group_key):
    """Train ML models on first 2 shards of the given group."""
    train_reqs = load_shard_range(group_key, 0, 2)
    prefix = f"{group_key}_native"

    # ML training data
    ml_key = f"ml_data_{prefix}"
    ml_path = os.path.join(MODEL_DIR, f"ml_data_{prefix}.npz")
    if not is_done(ck, ml_key):
        print(f"Generating ML training data for {group_key}...")
        X, y, ts = generate_training_data(train_reqs, fast_capacity=FAST_CAP, return_timestamps=True)
        np.savez_compressed(ml_path, X=X, y=y, ts=ts)
        mark_done(ck, ml_key)
        print(f"  {X.shape[0]} samples")

    ml_data = np.load(ml_path)
    X, y, ts = ml_data["X"], ml_data["y"], ml_data["ts"]

    models = {}
    for mtype, train_fn, params in [
        ("ML_Reg", lambda X, y, p: train_model(X, y, p, params={"n_estimators": 1000, "max_depth": 10, "num_leaves": 127}), None),
        ("ML_Rank", lambda X, y, p: train_ranking_model(X, y, p, timestamps=ts, params={"n_estimators": 500, "max_depth": 8, "num_leaves": 63}), None),
    ]:
        key = f"train_{prefix}_{mtype}"
        path = os.path.join(MODEL_DIR, f"{prefix}_{mtype}.joblib")
        if not is_done(ck, key):
            print(f"  Training {mtype} for {group_key}...")
            train_fn(X, y, path)
            mark_done(ck, key)
        models[mtype] = path

    # OPT imitation
    opt_key = f"opt_data_{prefix}"
    opt_path = os.path.join(MODEL_DIR, f"opt_data_{prefix}.npz")
    if not is_done(ck, opt_key):
        print(f"  Generating OPT imitation data for {group_key}...")
        opt_reqs = load_shard_range(group_key, 0, 1)
        X_opt, y_opt, g_opt = generate_imitation_data(opt_reqs, fast_capacity=FAST_CAP, sample_rate=2)
        if len(y_opt) == 0:
            print(f"    0 samples from 1 shard, trying 2...")
            X_opt, y_opt, g_opt = generate_imitation_data(train_reqs, fast_capacity=FAST_CAP, sample_rate=2)
        np.savez_compressed(opt_path, X=X_opt, y=y_opt, g=g_opt)
        mark_done(ck, opt_key)
        print(f"    {X_opt.shape[0]} samples")

    opt_data = np.load(opt_path)
    X_o, y_o, g_o = opt_data["X"], opt_data["y"], opt_data["g"]

    if len(y_o) > 0:
        for mtype, train_fn in [
            ("OPT_Imit_Reg", lambda X, y, g, p: train_regressor(X, y, p)),
            ("OPT_Imit_Rank", lambda X, y, g, p: train_ranker(X[:10000], y[:10000], g[:10000], p)),
        ]:
            key = f"train_{prefix}_{mtype}"
            path = os.path.join(MODEL_DIR, f"{prefix}_{mtype}.joblib")
            if not is_done(ck, key):
                print(f"  Training {mtype} for {group_key}...")
                train_fn(X_o, y_o, g_o, path)
                mark_done(ck, key)
            models[mtype] = path

    return models


def eval_scenario(ck, scenario_name, group_key, train_group, model_paths, eval_start=0, eval_end=2):
    """Evaluate policies on group_key shards with given models."""
    reqs = load_shard_range(group_key, eval_start, eval_end)
    warmup_n = len(reqs) // 2
    for r in reqs[:warmup_n]:
        r.is_warmup = True
    shards_str = f"{eval_start}-{eval_end-1}"
    print(f"\n--- {scenario_name}: eval on {group_key} shards {shards_str} ---")
    print(f"  {len(reqs)} requests, {warmup_n} warmup")

    for pol_name in ["LRU", "Decay", "ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank"]:
        key = f"{scenario_name}_{pol_name}"
        if is_done(ck, key):
            print(f"  {pol_name} already done.")
            continue
        if pol_name == "LRU":
            pol = make_lru_policy(FAST_CAP)
        elif pol_name == "Decay":
            pol = make_decay_policy(FAST_CAP, alpha=0.5, decay_rate=0.05, fast_fill_threshold=0.9)
        else:
            mp = model_paths.get(pol_name)
            if not mp or not os.path.exists(mp):
                print(f"  {pol_name}: no model, skipping")
                continue
            pol = make_ml_policy(FAST_CAP, mp, fast_fill_threshold=0.9)

        row = run_eval(pol_name, pol, reqs, warmup_n, scenario=scenario_name,
                      train_group=train_group, eval_group=group_key, shards_str=shards_str)
        append_result(row)
        mark_done(ck, key)


def get_cross_model_paths():
    """Models trained on c1g1 (from Part 2 optimal configs)."""
    part2 = os.path.join(PROJECT, "experiments/part_2/models")
    part1 = os.path.join(PROJECT, "experiments/part_1/models")
    return {
        "ML_Reg": os.path.join(part2, "ML_Reg_t1000_d10_l127.joblib"),
        "ML_Rank": os.path.join(part2, "ML_Rank_t500_d8_l63.joblib"),
        "OPT_Imit_Reg": os.path.join(part2, "OPT_Reg_t500_d6_l31.joblib"),
        "OPT_Imit_Rank": os.path.join(part2, "OPT_Rank_t100_d4_l15.joblib"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["baseline", "native_c1g2", "native_c3g1", "all"],
                        default="all")
    args = parser.parse_args()

    ck = load_checkpoint()
    cross_models = get_cross_model_paths()

    if args.task in ("baseline", "all"):
        eval_scenario(ck, "baseline", "c1g1", "c1g1", cross_models, eval_start=4, eval_end=6)

    if args.task in ("native_c1g2", "all"):
        native_models = train_native_models(ck, "c1g2")
        eval_scenario(ck, "native_c1g2", "c1g2", "c1g2", native_models, eval_start=4, eval_end=6)

    if args.task in ("native_c3g1", "all"):
        native_models = train_native_models(ck, "c3g1")
        eval_scenario(ck, "native_c3g1", "c3g1", "c3g1", native_models, eval_start=4, eval_end=6)

    print("\nDone!")
