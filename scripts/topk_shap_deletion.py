#!/usr/bin/env python3
"""Top-k SHAP deletion test for the Level-2 XGBoost model (camera-ready, Reviewer 3).

Reviewer 3: "If the claim is that attribution allocation and deletion response are
distinct, the direct test is to rank by SHAP and delete the top k."

For each official MatBench fold this script
  1. trains the full Level-2 model with the main-pipeline settings (8,000-round cap,
     200-round early stopping on a random 20% of the training fold, seed 42) and, if
     --reference-pattern is given, checks that its test predictions equal the frozen
     Level-2 predictions (so the ranking comes from the same model as the paper);
  2. ranks the columns of the deletion pool by mean |SHAP| on 3,000 training rows,
     sampled exactly as in the paper's SHAP analysis (per fold, training rows only);
  3. deletes the top-k columns of the pool and retrains (config "top");
  4. as a control, deletes k randomly chosen pool columns and retrains ("random").

The pool is the 150 structural columns by default (--pool structural); --pool all uses
all 283 columns. Use a separate --out-dir for each pool.

Input: ../topk_inputs/l2_matrices.pkl written by notebooks/mp_gap_v2_export_matrices.ipynb,
i.e. the exact Level-2 DataFrames that the main notebook passes to XGBoost (283 columns,
already one-hot encoded, aligned and imputed) -> run with --impute none (the default).

Every fit is appended to the results CSV straight away, so an interrupted run resumes
where it stopped. Run with --dry-run first: it only lists the pool.

Example (pilot on fold 0):
    python topk_shap_deletion.py --features-cache ../topk_inputs/l2_matrices.pkl --impute none \
        --reference-pattern '../matbench_outputs_v2_run0709/predictions_xgb/pred_fold_{fold}.npy' \
        --out-dir ../topk_shap_deletion_structural --folds 0
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "eta": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 1e-3,
    "tree_method": "hist",
}
RESULT_COLS = ["fold", "config", "k", "rep", "n_features", "best_iteration", "n_rounds",
               "early_stopped", "mae", "rmse", "shap_share_deleted", "seconds", "deleted"]
EXPECTED_POOL = {"structural": 150, "all": 283}


# ----------------------------------------------------------------------------- data
def load_targets(fold):
    """Train and test targets for one official fold, in the same row order as the cache."""
    from matbench.bench import MatbenchBenchmark
    if not hasattr(load_targets, "task"):          # same loading as the main notebook
        mb = MatbenchBenchmark(autoload=False)
        task = next(t for t in mb.tasks if t.dataset_name == "matbench_mp_gap")
        task.load()
        load_targets.task = task
    task = load_targets.task
    target = task.metadata["target"]
    tr = task.get_train_and_val_data(task.folds[fold], as_type="df").rename(columns=lambda c: c.strip())
    te = task.get_test_data(task.folds[fold], include_target=True, as_type="df").rename(columns=lambda c: c.strip())
    return tr[target].to_numpy(float), te[target].to_numpy(float)


def load_fold(cache, fold, impute):
    X_tr = cache[f"fold_{fold}_train"]
    X_te = cache[f"fold_{fold}_test"]
    X_tr, X_te = pd.DataFrame(X_tr), pd.DataFrame(X_te)
    X_tr.columns = [str(c) for c in X_tr.columns]
    X_te.columns = [str(c) for c in X_te.columns]
    if list(X_tr.columns) != list(X_te.columns):
        raise SystemExit(f"fold {fold}: train and test columns differ (names or order)")
    bad = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    if bad:
        raise SystemExit(f"Non-numeric columns in the cache (encode or drop them first): {bad[:10]}")
    if impute == "train_mean":            # fit on train only
        X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
        X_te = X_te.replace([np.inf, -np.inf], np.nan)
        means = X_tr.mean(numeric_only=True)
        X_tr = X_tr.fillna(means).fillna(0)
        X_te = X_te.fillna(means).fillna(0)
    else:                                 # matrices are used exactly as stored
        n_nan = int(X_tr.isna().sum().sum() + X_te.isna().sum().sum())
        if n_nan:
            print(f"  WARNING fold {fold}: {n_nan} missing values left with --impute none")
    return X_tr, X_te


def build_pool(columns, kind, composition_prefix, exclude_regex):
    if kind == "all":
        return list(columns)
    pat = re.compile(exclude_regex) if exclude_regex else None
    return [c for c in columns
            if not str(c).startswith(composition_prefix) and not (pat and pat.search(str(c)))]


def load_reference(pattern, fold):
    path = Path(pattern.format(fold=fold))
    if path.suffix == ".npz":
        z = np.load(path)
        return np.asarray(z[z.files[0]], float).ravel(), path
    if path.suffix == ".npy":
        return np.asarray(np.load(path), float).ravel(), path
    df = pd.read_csv(path, float_precision="round_trip")
    col = [c for c in df.columns if c.lower() not in ("id", "index", "fold", "y_true", "target")][-1]
    return df[col].to_numpy(float), path


# ----------------------------------------------------------------------------- model
def fit_predict(X_tr, y_tr, X_te, seed, max_rounds, n_jobs):
    """Main-pipeline XGBoost fit. Returns booster, clipped test prediction and run info."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    # DataFrames in, exactly as the main notebook (xgb_train_and_predict) passes them
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, shuffle=True)
    params = dict(XGB_PARAMS, seed=seed)
    if n_jobs:
        params["nthread"] = n_jobs
    booster = xgb.train(params, xgb.DMatrix(Xa, label=ya), num_boost_round=max_rounds,
                        evals=[(xgb.DMatrix(Xv, label=yv), "validation")],
                        early_stopping_rounds=200, verbose_eval=False)
    best = int(booster.best_iteration)
    pred = booster.predict(xgb.DMatrix(X_te), iteration_range=(0, best + 1))
    n_rounds = int(booster.num_boosted_rounds())
    return booster, np.maximum(pred, 0.0), dict(best_iteration=best, n_rounds=n_rounds,
                                               early_stopped=n_rounds < max_rounds)


def shap_ranking(booster, X_tr, seed, n_shap=3000):
    """Mean |SHAP| per column on n_shap training rows (same sampling as the paper)."""
    import shap
    n = min(n_shap, len(X_tr))
    idx = np.random.RandomState(seed).choice(len(X_tr), n, replace=False)
    sv = shap.TreeExplainer(booster).shap_values(X_tr.iloc[idx], check_additivity=False)
    return pd.Series(np.abs(sv).mean(axis=0), index=X_tr.columns, name="mean_abs_shap")


# ----------------------------------------------------------------------------- bookkeeping
def done_keys(path):
    if not path.exists():
        return set(), pd.DataFrame(columns=RESULT_COLS)
    df = pd.read_csv(path)
    return {(int(r.fold), r.config, int(r.k), int(r.rep)) for r in df.itertuples()}, df


def append_row(path, row):
    pd.DataFrame([row], columns=RESULT_COLS).to_csv(path, mode="a", header=not path.exists(), index=False)


def check_run_meta(path, meta):
    """Refuse to resume a results directory that was started with different settings."""
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{path.parent} was started with other settings {diff}; use a new --out-dir")
    else:
        path.write_text(json.dumps(meta, indent=1))


def summarise(results_path, out_path):
    df = pd.read_csv(results_path)
    full = df[df.config == "full"].set_index("fold").mae
    rows = []
    for (cfg, k), g in df[df.config != "full"].groupby(["config", "k"]):
        g = g[g.fold.isin(full.index)]
        if g.empty:
            continue
        d = g.mae.to_numpy() - full.loc[g.fold].to_numpy()       # paired by fold
        rows.append(dict(config=cfg, k=int(k), n_runs=len(g), folds=len(set(g.fold)),
                         mean_shap_share_deleted=g.shap_share_deleted.mean(),
                         mean_mae=g.mae.mean(), mean_delta_mae=d.mean(),
                         min_delta=d.min(), max_delta=d.max(),
                         all_runs_early_stopped=bool(g.early_stopped.all())))
    summ = pd.DataFrame(rows).sort_values(["k", "config"])

    # top minus random, paired by fold (random reps averaged within fold first)
    top = df[df.config == "top"].set_index(["fold", "k"]).mae
    rnd = df[df.config == "random"].groupby(["fold", "k"]).mae.mean()
    both = pd.concat([top.rename("top"), rnd.rename("random")], axis=1).dropna()
    if not both.empty:
        both["d"] = both.top - both.random
        tvr = both.groupby(level="k").d.agg(top_minus_random="mean", tvr_min="min", tvr_max="max",
                                            tvr_folds="size").reset_index()
        summ = summ.merge(tvr, on="k", how="left")
    summ.to_csv(out_path, index=False)

    print(f"\nFull Level-2 model: mean MAE {full.mean():.4f} eV over {len(full)} fold(s)")
    fmt = {c: "{:+.4f}".format for c in ["mean_delta_mae", "min_delta", "max_delta",
                                         "top_minus_random", "tvr_min", "tvr_max"] if c in summ}
    fmt["mean_mae"] = "{:.4f}".format
    fmt["mean_shap_share_deleted"] = "{:.1%}".format
    print(summ.to_string(index=False, formatters=fmt))
    return summ


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-cache", required=True, type=Path,
                    help="pickle with fold_{i}_train / fold_{i}_test Level-2 matrices")
    ap.add_argument("--out-dir", type=Path, default=Path("topk_shap_deletion"))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--random-reps", type=int, default=3, help="random-k controls per k and fold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rounds", type=int, default=8000)
    ap.add_argument("--n-jobs", type=int, default=0, help="XGBoost threads (0 = XGBoost default)")
    ap.add_argument("--impute", choices=["train_mean", "none"], default="none",
                    help="'none' if the matrices were exported already imputed (recommended)")
    ap.add_argument("--pool", choices=["structural", "all"], default="structural")
    ap.add_argument("--composition-prefix", default="MagpieData",
                    help="columns starting with this are composition features, not structural")
    ap.add_argument("--exclude-regex", default=r"^band center$",
                    help="non-structural columns to keep out of the pool (BandCenter by default)")
    ap.add_argument("--reference-pattern", default=None,
                    help="frozen Level-2 clipped test predictions, with {fold} in the path; "
                         "the full model must reproduce them")
    ap.add_argument("--reference-tol", type=float, default=0.0)
    ap.add_argument("--allow-reference-mismatch", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the pool and exit")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = pd.read_pickle(args.features_cache)
    pools, size_ok = {}, True        # one-hot columns are built per fold, so the pool is per fold
    for fold in args.folds:
        cols = [str(c) for c in pd.DataFrame(cache[f"fold_{fold}_train"]).columns]
        pools[fold] = build_pool(cols, args.pool, args.composition_prefix, args.exclude_regex)
        n_comp = sum(str(c).startswith(args.composition_prefix) for c in cols)
        ok = len(pools[fold]) == EXPECTED_POOL[args.pool]
        size_ok &= ok
        print(f"fold {fold}: {len(cols)} columns, {n_comp} composition, "
              f"{len(pools[fold])} in the {args.pool} pool {'' if ok else '<-- expected ' + str(EXPECTED_POOL[args.pool])}")
    (args.out_dir / f"pool_{args.pool}.json").write_text(json.dumps(pools, indent=1))
    if args.dry_run:
        print("Pools written to", args.out_dir / f"pool_{args.pool}.json")
        print("First 15 of fold", args.folds[0], ":", pools[args.folds[0]][:15])
        return
    if not size_ok:
        raise SystemExit("Pool size differs from the paper's column count; check the input matrices")
    if max(args.k) >= min(len(p) for p in pools.values()):
        raise SystemExit("k must be smaller than the pool")

    check_run_meta(args.out_dir / "run_meta.json",
                   dict(pool=args.pool, seed=args.seed, max_rounds=args.max_rounds,
                        impute=args.impute,
                        features_cache=args.features_cache.name))
    results = args.out_dir / "topk_results.csv"
    done, _ = done_keys(results)
    n_fits = len(args.folds) * (1 + len(args.k) * (1 + args.random_reps))
    print(f"{n_fits} fits planned, {len(done)} already in {results.name}")

    for fold in args.folds:
        X_tr, X_te = load_fold(cache, fold, args.impute)
        pool = pools[fold]
        y_tr, y_te = load_targets(fold)
        if len(X_tr) != len(y_tr) or len(X_te) != len(y_te):
            raise SystemExit(f"fold {fold}: feature rows and targets do not match")
        rank_path = args.out_dir / f"shap_ranking_fold{fold}.csv"
        rk = pd.read_csv(rank_path, index_col="column") if rank_path.exists() else None

        def run(config, k, rep, drop):
            if (fold, config, k, rep) in done:
                return None, None
            t0 = time.time()
            keep = [c for c in X_tr.columns if c not in set(drop)]
            booster, pred, info = fit_predict(X_tr[keep], y_tr, X_te[keep], args.seed,
                                              args.max_rounds, args.n_jobs)
            err = pred - y_te
            share = (float(rk.loc[list(drop), "mean_abs_shap"].sum() / rk.mean_abs_shap.sum())
                     if rk is not None and drop else 0.0)
            row = dict(fold=fold, config=config, k=k, rep=rep, n_features=len(keep), **info,
                       mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                       shap_share_deleted=share,
                       seconds=round(time.time() - t0, 1), deleted=";".join(map(str, drop)))
            append_row(results, row)
            done.add((fold, config, k, rep))
            print(f"  fold {fold} {config:<6} k={k:<3} rep={rep}  MAE {row['mae']:.4f}  "
                  f"SHAP share deleted {share:.1%}  rounds {info['n_rounds']}  "
                  f"{row['seconds'] / 60:.1f} min", flush=True)
            return booster, pred

        # 1. full model + reference check + SHAP ranking (re-done if the ranking file is missing)
        if rk is None:
            done.discard((fold, "full", 0, 0))
            if results.exists():                      # drop a stale full row before re-running
                df = pd.read_csv(results)
                df[~((df.fold == fold) & (df.config == "full"))].to_csv(results, index=False)
            booster, pred = run("full", 0, 0, [])
            if args.reference_pattern:
                ref, ref_path = load_reference(args.reference_pattern, fold)
                diff = float(np.max(np.abs(pred - ref))) if len(ref) == len(pred) else float("inf")
                (args.out_dir / f"reference_check_fold{fold}.json").write_text(json.dumps(
                    dict(reference=ref_path.name, n=len(pred), max_abs_diff=diff,
                         tol=args.reference_tol), indent=1))
                print(f"  fold {fold}: max |full - frozen| = {diff:.3g}")
                if diff > args.reference_tol and not args.allow_reference_mismatch:
                    raise SystemExit(f"fold {fold}: full model does not reproduce the frozen predictions; "
                                     "check the input matrices, column order and thread settings")
            imp = shap_ranking(booster, X_tr, args.seed)
            rk = imp.rename("mean_abs_shap").to_frame()
            rk["in_pool"] = rk.index.isin(pool)
            rk["rank_in_pool"] = rk.loc[rk.in_pool, "mean_abs_shap"].rank(ascending=False, method="first")
            rk.sort_values("mean_abs_shap", ascending=False).to_csv(rank_path, index_label="column")
            rk = pd.read_csv(rank_path, index_col="column")
        ranked_pool = rk[rk.in_pool].sort_values("rank_in_pool").index.tolist()

        # 2. top-k deletions and random-k controls
        for k in args.k:
            run("top", k, 0, ranked_pool[:k])
            for rep in range(args.random_reps):
                rng = np.random.default_rng([args.seed, fold, k, rep])   # independent stream per run
                run("random", k, rep, [pool[i] for i in rng.choice(len(pool), size=k, replace=False)])

    summarise(results, args.out_dir / "topk_summary.csv")


if __name__ == "__main__":
    main()
