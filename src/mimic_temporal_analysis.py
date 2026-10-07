"""Frozen, censoring-adjusted temporal MIMIC-IV CNB analysis.

Only source X/A/U/Delta and target X-only are read by the estimator.  All
model constants are written before point estimation; the old case EIF and
its censored-as-alive stacking loss are deliberately not reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from threadpoolctl import threadpool_limits

# The public repository is self-contained.  Environment variables may be used
# to point to a credentialed local MIMIC-IV workspace without editing code.
ROOT = Path(os.environ.get("CNB_REPO_ROOT", Path(__file__).resolve().parents[1]))
DESIGN = ROOT / os.environ.get("MIMIC_TEMPORAL_DESIGN", ".")
DATA = DESIGN / os.environ.get("MIMIC_TEMPORAL_DATA_DIR", "data/mimic_temporal")
OUT = DESIGN / os.environ.get("MIMIC_TEMPORAL_RESULTS_DIR", "results/mimic_temporal")
SEED, TAU, K, KFLOOR = int(os.environ.get("MIMIC_TEMPORAL_SEED", "20260918")), 14.0, 5, 0.02
GRID_EXPONENT = float(os.environ.get("MIMIC_TEMPORAL_GRID_EXPONENT", "0.75"))
MIN_GRID_INTERVALS = int(os.environ.get("MIMIC_TEMPORAL_MIN_GRID_INTERVALS", "128"))
CANDIDATES = ("linear_cox", "rsf", "gbsa")
CHECK_TIMES = np.array([3.0, 7.0, 10.0, 14.0])
DATA_MANIFEST = json.loads((DATA / "00_data_lock_manifest.json").read_text(encoding="utf-8"))
FEATURES = DATA_MANIFEST["base_features"] + DATA_MANIFEST["missingness_indicator_features"]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def load_data():
    sp, tp = DATA / "source_X_A_U_Delta.parquet", DATA / "target_X_only.parquet"
    assert digest(sp) == DATA_MANIFEST["source_file_sha256"]
    assert digest(tp) == DATA_MANIFEST["target_file_sha256"]
    s, t = pd.read_parquet(sp), pd.read_parquet(tp)
    assert not set(t.columns) & {"A", "U_days", "Delta", "drug", "deathtime", "dischtime"}
    assert len(s) == DATA_MANIFEST["n_source"] and len(t) == DATA_MANIFEST["n_target"]
    s["fold"] = grouped_folds(s, K, SEED, stratify=True)
    t["fold"] = grouped_folds(t, K, SEED + 1, stratify=False)
    return s, t


def grouped_folds(z, k, seed, stratify):
    unique = z.drop_duplicates("stay_id").sort_values("stay_id")
    splitter = StratifiedKFold(k, shuffle=True, random_state=seed) if stratify else KFold(k, shuffle=True, random_state=seed)
    strata = unique.A.to_numpy() if stratify else None
    mapping = {}
    for fold, (_, val) in enumerate(splitter.split(unique, strata)):
        mapping.update({int(x): fold for x in unique.iloc[val].stay_id})
    return z.stay_id.map(mapping).to_numpy(int)


def preprocessor():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())


def integration_grid(u, d):
    """Union of observed event times and a refining deterministic grid.

    With a source-training size n, the deterministic component has
    ceil(n**GRID_EXPONENT) intervals.  Its mesh is therefore
    O(n^{-GRID_EXPONENT}) = o(n^{-1/2}), while the event-time component
    retains the native jump times of the fitted survival estimators.
    """
    n_intervals = max(MIN_GRID_INTERVALS, int(np.ceil(len(u) ** GRID_EXPONENT)))
    regular = np.linspace(TAU / n_intervals, TAU, n_intervals)
    event_times = np.asarray(u)[(np.asarray(d) == 1) & (np.asarray(u) > 0) & (np.asarray(u) <= TAU)]
    return np.sort(np.unique(np.r_[regular, event_times]))


def fit_candidate(name, x, a, u, d, seed, forest_jobs=1):
    models = {}
    for arm in (0, 1):
        m = a == arm
        y = Surv.from_arrays(d[m].astype(bool), u[m])
        leaf = max(8, min(15, int(m.sum() // 12)))
        if name == "linear_cox":
            model = CoxPHSurvivalAnalysis(alpha=1.0, ties="breslow", n_iter=300, tol=1e-8)
        elif name == "rsf":
            model = RandomSurvivalForest(n_estimators=250, min_samples_leaf=leaf,
                min_samples_split=2 * leaf, max_features="sqrt", n_jobs=forest_jobs,
                random_state=seed + 101 * arm, low_memory=False)
        else:
            model = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.04,
                max_depth=2, min_samples_leaf=leaf, random_state=seed + 211 * arm)
        models[arm] = model.fit(x[m], y)
    return {"kind": name, "models": models, "grid": integration_grid(u, d)}


def survival(fit, x, arm, times):
    if fit["kind"] == "super_learner":
        return sum(w * survival(fit["candidates"][n], x, arm, times)
                   for w, n in zip(fit["weights"][arm], CANDIDATES))
    if fit["kind"] == "oracle":
        return np.exp(-fit["rates"][arm] * np.asarray(times)[None, :]) * np.ones((len(x), 1))
    blocks = []
    for first in range(0, len(x), 500):
        funcs = fit["models"][arm].predict_survival_function(x[first:first + 500], return_array=False)
        block = np.ones((len(funcs), len(times)))
        for j, fn in enumerate(funcs):
            ix = np.searchsorted(fn.x, times, side="right") - 1
            mask = ix >= 0
            block[j, mask] = fn.y[ix[mask]]
        blocks.append(block)
    return np.minimum.accumulate(np.clip(np.vstack(blocks), 1e-12, 1.0), axis=1)


def fit_censor(x, a, u, d, a_only=False):
    z = a[:, None].astype(float) if a_only else np.column_stack([a, x])
    # Only live discharge is a censoring event. Day-14 administrative
    # truncations must not be fitted as random-discharge events.
    dc = (d == 0) & (u < TAU)
    model = CoxPHSurvivalAnalysis(alpha=1.0, ties="breslow", n_iter=300, tol=1e-8).fit(z, Surv.from_arrays(dc, u))
    return {"model": model, "a_only": a_only}


def censor_survival(censor, x, a, times, rowwise=False):
    if "oracle_rate" in censor:
        values = np.exp(-censor["oracle_rate"] * np.asarray(times))
        return values if rowwise else np.ones((len(x), 1)) * values[None, :]
    z = a[:, None].astype(float) if censor["a_only"] else np.column_stack([a, x])
    model = censor["model"]
    baseline = model.cum_baseline_hazard_
    # K(t-) is used to avoid counting an administrative horizon or a tied
    # censoring event as having occurred before an event at t.
    ix = np.searchsorted(baseline.x, times, side="left") - 1
    h0 = np.zeros(len(times))
    m = ix >= 0
    h0[m] = baseline.y[ix[m]]
    risk = np.exp(np.clip(z @ model.coef_, -700, 700))
    h = risk * h0 if rowwise else risk[:, None] * h0[None, :]
    return np.exp(-np.minimum(h, 700))


def hazard_components(fit, x, grid):
    s0, s1 = survival(fit, x, 0, grid), survival(fit, x, 1, grid)
    d0 = np.maximum(np.diff(np.column_stack([np.zeros(len(x)), -np.log(s0)]), axis=1), 0)
    d1 = np.maximum(np.diff(np.column_stack([np.zeros(len(x)), -np.log(s1)]), axis=1), 0)
    b0 = np.column_stack([np.ones(len(x)), s0[:, :-1]])
    b1 = np.column_stack([np.ones(len(x)), s1[:, :-1]])
    return d0, d1, b0, b1


def plugin(fit, x):
    d0, d1, b0, b1 = hazard_components(fit, x, fit["grid"])
    return np.sum(b0 * b1 * (d0 - d1), axis=1)


def conditional_eif(fit, censor, x, a, u, d, p):
    grid = np.sort(np.unique(np.r_[fit["grid"], u[(d == 1) & (u <= TAU)]]))
    d0, d1, b0, b1 = hazard_components(fit, x, grid)
    increments = b0 * b1 * (d0 - d1)
    future = np.cumsum(increments[:, ::-1], axis=1)[:, ::-1] - increments
    h0, h1 = b1 - future / np.maximum(b0, 1e-12), -b0 - future / np.maximum(b1, 1e-12)
    lam, h = np.where(a[:, None] == 1, d1, d0), np.where(a[:, None] == 1, h1, h0)
    at_risk = grid[None, :] <= np.minimum(u, TAU)[:, None]
    kg_raw = censor_survival(censor, x, a, grid)
    kg = np.maximum(kg_raw, KFLOOR)
    raw = -np.sum(np.where(at_risk, h * lam / kg, 0), axis=1)
    event = (d == 1) & (u <= TAU)
    if event.any():
        rows = np.where(event)[0]
        cols = np.searchsorted(grid, u[event])
        ku = np.maximum(censor_survival(censor, x[event], a[event], u[event], rowwise=True), KFLOOR)
        raw[event] += h[rows, cols] / ku
    pia = np.where(a == 1, p, 1 - p)
    raw /= np.maximum(pia, 1e-6)
    return raw, {"integration_grid_points": int(len(grid)),
        "integration_grid_max_width_days": float(np.diff(np.r_[0.0, grid, TAU]).max()),
        "K_min_at_risk": float(kg_raw[at_risk].min()),
        "K_floor_hit_count": int(((kg_raw < KFLOOR) & at_risk).sum()),
        "risk_grid_count": int(at_risk.sum()),
        "K_floor_rate_at_risk": float(((kg_raw < KFLOOR) & at_risk).sum() / max(at_risk.sum(), 1))}


def fit_stack(train, seed, forest_jobs):
    folds = grouped_folds(train, 2, seed + 17, stratify=True)
    pred = np.full((len(train), len(CHECK_TIMES), 3), np.nan)
    weight_alive = np.zeros((len(train), len(CHECK_TIMES)))
    weight_dead = np.zeros_like(weight_alive)
    for k in range(2):
        it, iv = folds != k, folds == k
        pre = preprocessor().fit(train.loc[it, FEATURES])
        xt, xv = pre.transform(train.loc[it, FEATURES]), pre.transform(train.loc[iv, FEATURES])
        a, u, d = (train.loc[it, c].to_numpy() for c in ("A", "U_days", "Delta"))
        av, uv, dv = (train.loc[iv, c].to_numpy() for c in ("A", "U_days", "Delta"))
        censor = fit_censor(xt, a, u, d)
        kt = np.maximum(censor_survival(censor, xv, av, CHECK_TIMES), KFLOOR)
        ku = np.maximum(censor_survival(censor, xv, av, uv, rowwise=True), KFLOOR)
        alive = (uv[:, None] > CHECK_TIMES) | ((uv[:, None] == CHECK_TIMES) & (dv[:, None] == 0) & (CHECK_TIMES == TAU))
        dead = (uv[:, None] <= CHECK_TIMES) & (dv[:, None] == 1)
        weight_alive[iv] = alive / kt
        weight_dead[iv] = dead / ku[:, None]
        for j, name in enumerate(CANDIDATES):
            fitted = fit_candidate(name, xt, a, u, d, seed + 1000 * k + j, forest_jobs)
            for arm in (0, 1):
                m = av == arm
                pred[np.where(iv)[0][m], :, j] = survival(fitted, xv[m], arm, CHECK_TIMES)
            del fitted
    weights, losses = {}, {}
    for arm in (0, 1):
        m = train.A.to_numpy() == arm
        def loss(w):
            sv = pred[m] @ w
            return np.mean(weight_alive[m] * (1 - sv)**2 + weight_dead[m] * sv**2)
        opt = minimize(loss, np.ones(3)/3, method="SLSQP", bounds=[(0, 1)]*3,
            constraints={"type": "eq", "fun": lambda w: w.sum()-1}, options={"maxiter": 300, "ftol": 1e-11})
        if not opt.success:
            raise RuntimeError(f"Super Learner optimization: {opt.message}")
        weights[arm] = np.clip(opt.x, 0, 1) / opt.x.sum()
        losses[arm] = {name: float(loss(np.eye(3)[j])) for j, name in enumerate(CANDIDATES)}
    pre = preprocessor().fit(train[FEATURES])
    xt = pre.transform(train[FEATURES])
    a, u, d = (train[c].to_numpy() for c in ("A", "U_days", "Delta"))
    full = {name: fit_candidate(name, xt, a, u, d, seed + 9000 + j, forest_jobs) for j, name in enumerate(CANDIDATES)}
    fit = {"kind": "super_learner", "candidates": full, "weights": weights,
        "losses": losses, "grid": np.sort(np.unique(u[(d == 1) & (u <= TAU)]))}
    fit["grid"] = integration_grid(u, d)
    return pre, fit


def fit_ratio(st, tt, sv):
    both = pd.concat([st[FEATURES], tt[FEATURES]], ignore_index=True)
    pre = preprocessor().fit(both)
    model = LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs").fit(pre.transform(both), np.r_[np.zeros(len(st)), np.ones(len(tt))])
    def odds(z):
        p = np.clip(model.predict_proba(pre.transform(z[FEATURES]))[:, 1], 1e-6, 1-1e-6)
        return p/(1-p) * len(st)/len(tt)
    normalizer = odds(st).mean()
    return odds(sv)/normalizer


def estimate(s, t, forest_jobs=1, all_variants=False, checkpoint=None, verbose=False):
    names = ["primary_super_learner"]
    if all_variants:
        names += ["censor_A_only", *CANDIDATES]
    results = {n: {"target_plugin": np.full(len(t), np.nan), "source_plugin": np.full(len(s), np.nan), "raw_eif": np.full(len(s), np.nan)} for n in names}
    ratio, prop = np.full(len(s), np.nan), np.full(len(s), np.nan)
    diagnostics, stack_rows = [], []
    for k in range(K):
        tick = time.time()
        sm, tm = s.fold.to_numpy() == k, t.fold.to_numpy() == k
        st, sv = s.loc[~sm], s.loc[sm]
        tt, tv = t.loc[~tm], t.loc[tm]
        cache = Path(checkpoint) / f"fold_{k}.npz" if checkpoint else None
        if cache and cache.exists():
            z = np.load(cache, allow_pickle=False)
            ratio[sm], prop[sm] = z["ratio"], z["prop"]
            for n in names:
                for key in ("target_plugin", "source_plugin", "raw_eif"):
                    results[n][key][tm if key == "target_plugin" else sm] = z[f"{n}__{key}"]
            payload = json.loads(str(z["metadata"]))
            diagnostics.extend(payload["diagnostics"]); stack_rows.extend(payload["stack_rows"])
            continue
        pre, fit = fit_stack(st.reset_index(drop=True), SEED + 10000*k, forest_jobs)
        xst, xsv, xtv = (pre.transform(z[FEATURES]) for z in (st, sv, tv))
        a, u, d = (st[c].to_numpy() for c in ("A", "U_days", "Delta"))
        av, uv, dv = (sv[c].to_numpy() for c in ("A", "U_days", "Delta"))
        pmod = LogisticRegression(C=0.01, max_iter=3000, solver="lbfgs").fit(xst, a)
        p = np.clip(pmod.predict_proba(xsv)[:, 1], 1e-6, 1-1e-6)
        r = fit_ratio(st, tt, sv)
        ratio[sm], prop[sm] = r, p
        cfull = fit_censor(xst, a, u, d)
        caonly = fit_censor(xst, a, u, d, a_only=True) if all_variants else None
        fold_diags, fold_stacks = [], []
        for n in names:
            ef = fit["candidates"][n] if n in CANDIDATES else fit
            cf = caonly if n == "censor_A_only" else cfull
            raw, kd = conditional_eif(ef, cf, xsv, av, uv, dv, p)
            results[n]["target_plugin"][tm] = plugin(ef, xtv)
            results[n]["source_plugin"][sm] = plugin(ef, xsv)
            results[n]["raw_eif"][sm] = raw
            row = {"variant": n, "fold": k, "source_train_n": len(st), "source_valid_n": len(sv), "target_train_n": len(tt), "target_valid_n": len(tv),
                "source_train_events": int(d.sum()), "source_valid_events": int(dv.sum()),
                "ratio_valid_ess": float(r.sum()**2/(r@r)), "ratio_valid_q99": float(np.quantile(r, .99)), "ratio_valid_max": float(r.max()),
                "propensity_min": float(p.min()), "propensity_max": float(p.max()), **kd}
            fold_diags.append(row)
        for arm in (0, 1):
            fold_stacks.append({"fold": k, "arm": arm,
                **{f"weight_{n}": float(w) for n, w in zip(CANDIDATES, fit["weights"][arm])},
                **{f"ipcw_brier_{n}": v for n, v in fit["losses"][arm].items()}})
        diagnostics.extend(fold_diags); stack_rows.extend(fold_stacks)
        if cache:
            cache.parent.mkdir(exist_ok=True, parents=True)
            arrays = {"ratio": r, "prop": p, "metadata": json.dumps({"diagnostics": fold_diags, "stack_rows": fold_stacks})}
            for n in names:
                for key in ("target_plugin", "source_plugin", "raw_eif"):
                    arrays[f"{n}__{key}"] = results[n][key][tm if key == "target_plugin" else sm]
            np.savez_compressed(cache, **arrays)
        if verbose:
            print(f"Completed fold {k+1}/{K}, elapsed {time.time()-tick:.1f}s", flush=True)
        del fit
    summaries = []
    ss = np.array([len(s)/(K*int((s.fold==f).sum())) for f in s.fold])
    ts = np.array([len(t)/(K*int((t.fold==f).sum())) for f in t.fold])
    for n in names:
        z = results[n]
        tp, sc = ts*z["target_plugin"], ss*ratio*z["raw_eif"]
        one = float(tp.mean()+sc.mean())
        se = float(np.sqrt(tp.var(ddof=1)/len(t)+sc.var(ddof=1)/len(s)))
        source_one = ss * (z["source_plugin"]+z["raw_eif"])
        source_se = source_one.std(ddof=1)/np.sqrt(len(s))
        assert np.isfinite(tp).all() and np.isfinite(sc).all()
        summaries.append({"variant": n, "n_source": len(s), "n_target": len(t), "events_source": int(s.Delta.sum()),
            "target_plugin": float(tp.mean()), "source_eif_correction": float(sc.mean()), "transported_cnb": one,
            "two_sample_eif_se": se, "wald_ci_low": one-1.959963984540054*se, "wald_ci_high": one+1.959963984540054*se,
            "source_only_cnb": float(source_one.mean()), "source_only_eif_se": float(source_se),
            "source_only_ci_low": float(source_one.mean()-1.959963984540054*source_se), "source_only_ci_high": float(source_one.mean()+1.959963984540054*source_se),
            "ratio_ess_fraction": float(ratio.sum()**2/(ratio@ratio)/len(s)), "ratio_q99": float(np.quantile(ratio,.99)), "ratio_max": float(ratio.max())})
    contributions = pd.concat([
        pd.DataFrame({"role": "source", "stay_id": s.stay_id, "fold": s.fold, "ratio": ratio, "propensity": prop, "raw_eif": results[names[0]]["raw_eif"], "correction": ratio*results[names[0]]["raw_eif"], "source_plugin": results[names[0]]["source_plugin"]}),
        pd.DataFrame({"role": "target", "stay_id": t.stay_id, "fold": t.fold, "target_plugin": results[names[0]]["target_plugin"]})
    ], ignore_index=True)
    return pd.DataFrame(summaries), pd.DataFrame(diagnostics), pd.DataFrame(stack_rows), contributions


def validation():
    rng = np.random.default_rng(SEED)
    d0, d1 = rng.uniform(.001, .03, (2, 9))
    def fun(v0, v1):
        s0, s1 = np.exp(-np.r_[0, np.cumsum(v0[:-1])]), np.exp(-np.r_[0, np.cumsum(v1[:-1])])
        return np.sum(s0*s1*(v0-v1))
    b0, b1 = np.exp(-np.r_[0, np.cumsum(d0[:-1])]), np.exp(-np.r_[0, np.cumsum(d1[:-1])])
    inc = b0*b1*(d0-d1)
    future = np.cumsum(inc[::-1])[::-1]-inc
    analytical = (b0*b1-future, -b0*b1-future)
    errors = []
    for arm in (0, 1):
        for j in range(9):
            plus, minus = [d0.copy(), d1.copy()], [d0.copy(), d1.copy()]
            plus[arm][j] += 1e-6; minus[arm][j] -= 1e-6
            numeric = (fun(*plus)-fun(*minus))/2e-6
            errors.append(abs(numeric-analytical[arm][j]))
    assert max(errors) < 1e-8
    # Known exponential event and censor laws: right-censor EIF must be
    # centered under the true nuisance functions, with no target outcome.
    n = 30000
    a = rng.binomial(1, .37, n)
    rates = (.06, .035)
    event = rng.exponential(1/np.where(a==1,rates[1],rates[0]))
    cens = rng.exponential(1/.12, n)
    u = np.minimum(np.minimum(event,cens),TAU)
    delta = (event<=cens)&(event<=TAU)
    fit = {"kind": "oracle", "rates": rates, "grid": np.arange(.02, TAU+.001, .02)}
    pieces = []
    for first in range(0,n,500):
        ix = slice(first,first+500)
        raw,_ = conditional_eif(fit,{"oracle_rate": .12},np.zeros((len(a[ix]),1)),a[ix],u[ix],delta[ix],np.full(len(a[ix]),.37))
        pieces.append(raw)
    v = np.r_[tuple(pieces)]
    se = v.std(ddof=1)/np.sqrt(n)
    assert abs(v.mean()) < 5*se+.002
    result = {"finite_difference_max_error": max(errors), "oracle_censored_eif_mean": float(v.mean()), "oracle_censored_eif_mc_se": float(se), "tests_pass": True}
    OUT.mkdir(exist_ok=True)
    (OUT/"implementation_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


def bootstrap_job(b):
    threadpool_limits(1)
    tick = time.time()
    s,t = load_data()
    rng = np.random.default_rng(SEED+1000000+b)
    si = np.r_[tuple(rng.choice(s.index[s.A.eq(a)],int(s.A.eq(a).sum()),replace=True) for a in (0,1))]
    ti = rng.choice(t.index,len(t),replace=True)
    # Copies of each original patient inherit one outer fold and are also
    # grouped together in the inner CV; no bootstrap-copy leakage is allowed.
    sb,tb = s.loc[si].reset_index(drop=True),t.loc[ti].reset_index(drop=True)
    try:
        summary,*_ = estimate(sb,tb,forest_jobs=1,all_variants=False)
        z = summary.iloc[0].to_dict()
        z.update({"bootstrap": b,"status":"success","elapsed_seconds":time.time()-tick})
    except Exception as e:
        z={"bootstrap":b,"status":"failed","error":str(e),"traceback":traceback.format_exc(),"elapsed_seconds":time.time()-tick}
    path=OUT/"bootstrap_checkpoints"/f"replicate_{b:04d}.json"
    path.write_text(json.dumps(z,indent=2),encoding="utf-8")
    return z


def bootstrap(reps,workers):
    folder=OUT/"bootstrap_checkpoints";folder.mkdir(exist_ok=True,parents=True)
    todo=[b for b in range(reps) if not (folder/f"replicate_{b:04d}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(bootstrap_job,b) for b in todo]
        for f in as_completed(futures):
            z=f.result();done=len(list(folder.glob("replicate_*.json")))
            print(f"Bootstrap {done}/{reps}: {z['status']}; last elapsed {z['elapsed_seconds']:.1f}s",flush=True)
    rows=[json.loads((folder/f"replicate_{b:04d}.json").read_text(encoding="utf-8")) for b in range(reps)]
    raw=pd.DataFrame(rows);raw.to_csv(OUT/"bootstrap_full_refit_raw.csv",index=False)
    good=raw.loc[raw.status.eq("success"),"transported_cnb"]
    pd.DataFrame([{"requested_replicates":reps,"successful_replicates":len(good),"failed_replicates":reps-len(good),"bootstrap_sd":good.std(ddof=1),"percentile_ci_low":good.quantile(.025),"percentile_ci_high":good.quantile(.975)}]).to_csv(OUT/"bootstrap_full_refit_interval.csv",index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["validate","point","bootstrap"],default="point")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--reps",type=int,default=500)
    args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    threadpool_limits(1)
    if args.stage=="validate":validation();return
    if args.stage=="bootstrap":bootstrap(args.reps,args.workers);return
    s,t=load_data()
    pd.concat([s[["stay_id","fold"]].assign(role="source"),t[["stay_id","fold"]].assign(role="target")]).to_csv(OUT/"outer_fold_assignments.csv",index=False)
    import importlib.metadata as meta
    manifest={"script_sha256":digest(__file__),"seed":SEED,"outer_folds":K,"inner_folds":2,"censor_survival_floor":KFLOOR,
        "integration_grid":{"event_times_included":True,"deterministic_intervals":"ceil(n_train**0.75)","minimum_intervals":MIN_GRID_INTERVALS},
        "cox_alpha":1.0,"rsf_trees":250,"gbsa_trees":150,"ipcw_brier_times":CHECK_TIMES.tolist(),
        "bootstrap_replicates":500,"target_outcome_read":False,"data_manifest":DATA_MANIFEST,
        "packages":{n:meta.version(n) for n in ("numpy","pandas","scikit-learn","scikit-survival","scipy")}}
    (OUT/"execution_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    summary,diags,stack,contrib=estimate(s,t,forest_jobs=args.workers,all_variants=True,checkpoint=OUT/"point_checkpoints",verbose=True)
    summary.to_csv(OUT/"primary_and_model_sensitivity_results.csv",index=False)
    diags.to_csv(OUT/"crossfit_fold_diagnostics.csv",index=False)
    stack.to_csv(OUT/"super_learner_ipcw_weights.csv",index=False)
    contrib.to_parquet(OUT/"primary_two_sample_contributions.parquet",index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":main()
