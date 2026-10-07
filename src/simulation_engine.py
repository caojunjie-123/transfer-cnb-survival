from __future__ import annotations

"""Light model-combination screen under the common-superpopulation design.

The target superpopulation has p=5 Toeplitz covariates.  The target sample is
drawn from that superpopulation, while the source sample is selected through
I_S with a covariate-dependent inclusion probability.  Event-time and
censoring mechanisms are generated conditionally on X and treatment.  The
linear and nonlinear scenarios satisfy conditional survival transportability;
the outcome_drift scenario deliberately violates it while keeping covariate
shift unchanged.
The one-step correction evaluates D_10^c-D_01^c on each fold's pooled
continuous-time event-jump grid.
"""

from pathlib import Path
import itertools
import importlib.util
import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import SplineTransformer
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv


ROOT = Path(__file__).resolve().parent
SEED = 20260608
P = 5
TAU = 12.0
NONPH_EDGES = np.linspace(0.0, TAU, 5)
NONPH_WIDTHS = np.diff(NONPH_EDGES)
MU_T = np.array([0.4, -0.4, 0.3, 0.2, -0.2])
MU_STAR = MU_T.copy()
RHO = 0.6
SIGMA = RHO ** np.abs(np.subtract.outer(np.arange(P), np.arange(P)))
CHOL = np.linalg.cholesky(SIGMA)
SOURCE_SELECTION_INTERCEPT = -1.0
SOURCE_SELECTION_COEF = np.linalg.solve(SIGMA, -MU_STAR)
SOURCE_SELECTION_STRENGTH = 1.0

# Light screening settings, not the notebook's formal n_rep=500 run.
N_S = 1000
N_T = 200
N_REP = 20
N_FOLDS = 5
# The estimator uses a fixed grid, while the primary truth is evaluated on the
# continuous-time Weibull DGP so grid approximation is not hidden in bias.
# Use one fixed grid for the event model, censoring model, EIF risk sets, and
# the simulation truth.  A fold-specific quantile grid changes the parameter
# across folds unless every EIF and truth calculation uses that same grid.
N_INTERVALS = 32
INTEGRATION_GRID_EXPONENT = 0.75
MIN_INTEGRATION_INTERVALS = 128
# This grid is used only to verify that numerical evaluation of the continuous
# target has converged. It is deliberately separate from nuisance-model grids.
TRUTH_DIAGNOSTIC_INTERVALS = 512
TRUTH_DIAGNOSTIC_SAMPLE_SIZE = 20000
WEIGHT_CAP = 8.0
CALIBRATION_BASIS_KIND = "linear"
CALIBRATION_RIDGE = 0.05
CALIBRATION_RIDGE_RATE = "fixed"
CALIBRATION_GUIDE_PENALTY_FACTOR = 0.20
CALIBRATION_GUIDE_TIMES = np.array([3.0, 6.0, 9.0, 12.0])
CALIBRATION_SPLINE_KNOTS = 4
# Each fold must use the same nuisance fit for its target plug-in and source
# correction so the empirical one-step expansion is evaluated at one law.
TARGET_PLUGIN_MODE = "cross_fitted"
# Numerical stabilization for finite samples with sparse late follow-up. These
# are reported diagnostics, not part of the population DGP.
EVENT_HAZARD_INCREMENT_CAP = 1.0
CENSOR_HAZARD_INCREMENT_CAP = 1.0
CENSOR_SURVIVAL_FLOOR = 0.05
EVENT_SURVIVAL_FLOOR = 0.05
Z95 = 1.959963984540054
MULTIPLIER_REPS = 200
CONTROL_VARIATE_RIDGE = 1e-4
ENABLE_CONTROL_VARIATE = False
OUT = ROOT

# Reuse the project's discrete-time hazard learner.  It estimates one hazard
# probability per interval with a censored-survival likelihood and makes no
# proportional-hazards assumption.
# The two optional neural-survival utilities are distributed beside this
# module so the repository has no dependence on the original workstation
# directory layout.
SURVIVAL_ML_PATH = ROOT / "dl_nuisance_onestep_comparison.py"
_spec = importlib.util.spec_from_file_location("survival_ml", SURVIVAL_ML_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load discrete-time survival module: {SURVIVAL_ML_PATH}")
survival_ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(survival_ml)
# Recalibrate each interval's baseline logit after neural fitting. This keeps
# the flexible covariate effect while preventing systematic survival drift.
survival_ml.DISCRETE_HAZARD_USE_OFFSET_CALIBRATION = True

# MTLR/PMF is the active non-Cox event and censoring learner. It estimates a
# normalized event-time mass function and converts it to interval hazard
# increments, so the predicted survival curve is monotone by construction.
MTLR_PATH = ROOT / "neural_survival_candidates.py"
_mtlr_spec = importlib.util.spec_from_file_location("mtlr_survival", MTLR_PATH)
if _mtlr_spec is None or _mtlr_spec.loader is None:
    raise ImportError(f"Cannot load MTLR survival module: {MTLR_PATH}")
mtlr_survival = importlib.util.module_from_spec(_mtlr_spec)
_mtlr_spec.loader.exec_module(mtlr_survival)

# Training settings. The legacy variable names are retained so existing CLI
# and bootstrap entry points continue to control the active MTLR learner.
DISCRETE_HAZARD_EPOCHS = 250
DISCRETE_HAZARD_PATIENCE = 25
DISCRETE_HAZARD_LR = 2e-3
# Selected by the MTLR design screen.
MTLR_SHARED = True
MTLR_GRID = "uniform"
MTLR_HIDDEN = (32, 16)
MTLR_WEIGHT_DECAY = 5e-4
MTLR_GRAD_CLIP = 5.0
MTLR_ENSEMBLE = 3
CENSOR_LEARNER_OVERRIDE = None
P_LEARNER_OVERRIDE = None
RATIO_LEARNERS_OVERRIDE = None
FOLD_SEED_OFFSET = 0
SOURCE_MODE = "shifted"
RANDOMIZED_TREATMENT = False
RSF_TREES = 300
RSF_MIN_LEAF = 15
RSF_MAX_FEATURES = "sqrt"
RSF_N_JOBS = 1
RSF_PREDICTION_MODE = "survival"
GBSA_TREES = 150
GBSA_LEARNING_RATE = 0.04
GBSA_MAX_DEPTH = 2
GBSA_MIN_LEAF = 15
SIEVE_COX_RIDGE = 1.0
SIEVE_N_KNOTS = 4
STACK_VALIDATION_FRACTION = 0.20
STACK_INNER_FOLDS = 3
PROPENSITY_INNER_FOLDS = 3
PROPENSITY_RIDGE_GRID = (0.001, 0.01, 0.1, 1.0, 5.0, 10.0)

# The event model is the comparison target.  The censoring DGP is correctly
# represented by the Cox censoring features, which is substantially more
# stable than fitting a second high-variance neural survival model in the EIF.
# The ratio learner is compared against its oracle.
SCENARIO_NUISANCE = {
    "linear": ("cox", "logistic", ("guided_calibration", "oracle")),
    "nonlinear": ("cox", "logistic", ("guided_calibration", "oracle")),
    "strong_nonlinear": ("cox", "logistic", ("guided_calibration", "oracle")),
    "nonproportional": ("cox", "logistic", ("guided_calibration", "oracle")),
    "outcome_drift": ("cox", "logistic", ("guided_calibration", "oracle")),
}


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def softmax(z):
    q = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(q, -35.0, 35.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def solve(h, g):
    try:
        return np.linalg.solve(h, g)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(h, g, rcond=None)[0]


def ridge_hessian(h, ridge):
    h = h.copy()
    h.flat[:: h.shape[0] + 1] += ridge
    h[0, 0] -= ridge
    return h


def fit_logistic(z, y, ridge=1e-3, max_iter=40):
    beta = np.zeros(z.shape[1])
    for _ in range(max_iter):
        p = expit(z @ beta)
        penalty_beta = beta.copy()
        penalty_beta[0] = 0.0
        grad = z.T @ (p - y) + ridge * penalty_beta
        h = (z.T * (p * (1.0 - p))) @ z
        step = solve(ridge_hessian(h, ridge), grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def fit_propensity(x, y, learner):
    """Choose logistic ridge by inner OOF treatment log loss."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    z_raw = features_p(x, learner)
    rng = np.random.default_rng(SEED + 17000 + len(x) + z_raw.shape[1])
    folds = make_folds(len(x), PROPENSITY_INNER_FOLDS, rng)
    losses = {}
    for ridge in PROPENSITY_RIDGE_GRID:
        fold_losses = []
        for fold in range(PROPENSITY_INNER_FOLDS):
            train, valid = folds != fold, folds == fold
            center = z_raw[train].mean(axis=0)
            scale = np.maximum(z_raw[train].std(axis=0, ddof=1), 1e-6)
            center[0], scale[0] = 0.0, 1.0
            z_train = (z_raw[train] - center) / scale
            z_valid = (z_raw[valid] - center) / scale
            beta = fit_logistic(z_train, y[train], ridge=float(ridge))
            probability = np.clip(expit(z_valid @ beta), 1e-6, 1.0 - 1e-6)
            fold_losses.append(float(-np.mean(
                y[valid] * np.log(probability)
                + (1 - y[valid]) * np.log(1.0 - probability)
            )))
        losses[float(ridge)] = float(np.mean(fold_losses))
    selected = min(losses, key=losses.get)
    center = z_raw.mean(axis=0)
    scale = np.maximum(z_raw.std(axis=0, ddof=1), 1e-6)
    center[0], scale[0] = 0.0, 1.0
    z = (z_raw - center) / scale
    return {
        "beta": fit_logistic(z, y, ridge=selected),
        "center": center,
        "scale": scale,
        "ridge": float(selected),
        "cv_loss": float(losses[selected]),
        "learner": learner,
    }


def fit_cox(z, u, event, ridge=1e-3, max_iter=35):
    order = np.argsort(-u)
    z, event = z[order], event[order].astype(float)
    beta = np.zeros(z.shape[1])
    for _ in range(max_iter):
        eta = np.clip(z @ beta, -25.0, 25.0)
        w = np.exp(eta)
        risk = np.maximum(np.cumsum(w), 1e-12)
        wz = np.cumsum(w[:, None] * z, axis=0)
        grad = np.sum(event[:, None] * (wz / risk[:, None] - z), axis=0) + ridge * beta
        # Breslow risk-set second moment for a Newton step.
        wzz = np.cumsum(w[:, None, None] * z[:, :, None] * z[:, None, :], axis=0)
        mean = wz / risk[:, None]
        h = np.zeros((z.shape[1], z.shape[1]))
        for i in range(len(z)):
            if event[i] > 0:
                h += wzz[i] / risk[i] - np.outer(mean[i], mean[i])
        step = solve(ridge_hessian(h, ridge), grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


def breslow(z, u, event, beta):
    eta = np.clip(z @ beta, -30.0, 30.0)
    w = np.exp(eta)
    times = np.sort(np.unique(u[event == 1]))
    inc = np.array([
        np.sum((u == t) & (event == 1)) / max(np.sum(w[u >= t]), 1e-12)
        for t in times
    ])
    return times, inc


def sample_x(n, mu, rng):
    return np.asarray(mu) + rng.normal(size=(n, P)) @ CHOL.T


def source_inclusion_prob(x):
    return np.clip(
        expit(
            SOURCE_SELECTION_INTERCEPT
            + SOURCE_SELECTION_STRENGTH * np.asarray(x) @ SOURCE_SELECTION_COEF
        ),
        1e-6,
        1.0 - 1e-6,
    )


def sample_source_x(n, rng):
    """Rejection-sample source covariates from P*(X | I_S=1)."""
    accepted = []
    total = 0
    while total < n:
        m = max(4 * (n - total), 1000)
        x_pool = sample_x(m, MU_STAR, rng)
        keep = rng.uniform(size=m) < source_inclusion_prob(x_pool)
        if np.any(keep):
            x_keep = x_pool[keep]
            accepted.append(x_keep)
            total += len(x_keep)
    return np.vstack(accepted)[:n]


def propensity_true(x):
    return np.clip(expit(0.5 + 0.8 * x[:, 0] - 0.5 * x[:, 1]), 0.05, 0.95)


def propensity_dgp(x):
    if RANDOMIZED_TREATMENT:
        return np.full(len(x), 0.5, dtype=float)
    return propensity_true(x)


def eta_nonlinear(x, a):
    if a == 0:
        return (0.65 * x[:, 0] - 0.55 * x[:, 1]
                + 0.45 * np.sqrt(np.abs(x[:, 0] * x[:, 1]) + 1e-8)
                - 0.35 * np.sqrt(np.abs(x[:, 4]) + 1e-8)
                + 0.40 * np.cos(x[:, 2])
                - 0.35 * np.cos(x[:, 2]) * np.cos(x[:, 3]))
    return (0.10 * x[:, 0] - 0.15 * x[:, 1]
            + 0.75 * x[:, 2] * x[:, 4]
            - 0.55 * x[:, 1] ** 2
            + 0.45 * np.cos(x[:, 3])
            + 0.35 * np.sqrt(np.abs(x[:, 0] * x[:, 4]) + 1e-8))


def eta_strong_nonlinear(x, a):
    """Stronger nonlinear PH DGP used to stress-test flexible learners."""
    x = np.asarray(x, dtype=float)
    if a == 0:
        eta = (1.10 * np.sin(1.2 * x[:, 0]) - 0.90 * np.tanh(x[:, 1])
               + 0.70 * x[:, 0] * x[:, 1] - 0.55 * x[:, 2] ** 2
               + 0.45 * x[:, 3] * x[:, 4] - 0.35 * np.cos(1.5 * x[:, 2]))
    else:
        eta = (-0.80 * np.sin(0.9 * x[:, 0]) + 0.75 * np.tanh(x[:, 1])
               + 0.85 * x[:, 2] * x[:, 4] - 0.70 * x[:, 1] ** 2
               + 0.50 * np.cos(1.3 * x[:, 3]) + 0.40 * x[:, 0] * x[:, 3])
    return np.clip(eta, -5.0, 5.0)


def nonph_eta_by_interval(x, a):
    """Interval-specific log hazard effects for a non-proportional DGP."""
    x = np.asarray(x, dtype=float)
    if a == 0:
        terms = [
            0.85 * x[:, 0] - 0.55 * x[:, 1] + 0.55 * x[:, 2] * x[:, 4],
            -0.75 * x[:, 0] + 0.65 * x[:, 1] ** 2 - 0.45 * x[:, 3],
            0.45 * np.sin(x[:, 2]) + 0.70 * x[:, 0] * x[:, 3] - 0.35 * x[:, 4],
            -0.60 * np.tanh(x[:, 1]) + 0.55 * x[:, 2] - 0.50 * x[:, 3] * x[:, 4],
        ]
    else:
        terms = [
            -0.65 * x[:, 0] + 0.45 * x[:, 1] + 0.70 * x[:, 2] * x[:, 4],
            0.80 * x[:, 0] - 0.55 * x[:, 1] ** 2 + 0.40 * x[:, 3],
            -0.50 * np.sin(x[:, 2]) + 0.60 * x[:, 1] * x[:, 4] + 0.35 * x[:, 0],
            0.55 * np.tanh(x[:, 1]) - 0.60 * x[:, 2] + 0.45 * x[:, 3] * x[:, 4],
        ]
    return np.clip(np.column_stack(terms), -4.5, 4.5)


def nonph_hazard_rates(x, a):
    base = np.array([0.025, 0.035, 0.030, 0.040]) if a == 0 else np.array([0.030, 0.025, 0.040, 0.030])
    return np.exp(nonph_eta_by_interval(x, a)) * base[None, :]


def event_hazard_grid_true(x, arm, scenario, n_intervals=None):
    """True integrated event hazard on a deterministic diagnostic grid."""
    x = np.asarray(x, dtype=float)
    n_intervals = N_INTERVALS if n_intervals is None else int(n_intervals)
    edges = np.linspace(0.0, TAU, n_intervals + 1)
    if scenario == "nonproportional":
        rates = nonph_hazard_rates(x, arm)
        idx = np.minimum(np.searchsorted(NONPH_EDGES[1:], (edges[:-1] + edges[1:]) / 2.0), len(NONPH_WIDTHS) - 1)
        return rates[:, idx] * np.diff(edges)[None, :]
    scale, shape = ((16.0, 1.35) if arm == 0 else (18.0, 1.25))
    if scenario == "strong_nonlinear":
        eta = eta_strong_nonlinear(x, arm)
    else:
        eta = eta_dgp(x, arm, scenario, population="target")
    return np.exp(np.clip(eta, -8, 8))[:, None] * np.diff((edges / scale) ** shape)[None, :]


def sample_nonph_event(x, a, rng):
    out = np.empty(len(x))
    for arm in (0, 1):
        mask = a == arm
        if not np.any(mask):
            continue
        rates = nonph_hazard_rates(x[mask], arm)
        cumulative = np.cumsum(rates * NONPH_WIDTHS[None, :], axis=1)
        e = -np.log(np.clip(rng.uniform(size=mask.sum()), 1e-12, 1.0))
        j = np.minimum(np.sum(cumulative < e[:, None], axis=1), len(NONPH_WIDTHS) - 1)
        previous = np.where(j == 0, 0.0, cumulative[np.arange(len(j)), j - 1])
        out[mask] = NONPH_EDGES[j] + (e - previous) / rates[np.arange(len(j)), j]
    return out


def eta_linear(x, a):
    """A correctly specified proportional-hazards DGP for Linear Cox."""
    if a == 0:
        return 0.65 * x[:, 0] - 0.55 * x[:, 1] + 0.20 * x[:, 2] - 0.25 * x[:, 3] - 0.35 * x[:, 4]
    return 0.10 * x[:, 0] - 0.15 * x[:, 1] + 0.75 * x[:, 2] - 0.55 * x[:, 3] + 0.35 * x[:, 4]


def eta_dgp(x, a, scenario, population="target"):
    if scenario == "linear":
        return eta_linear(x, a)
    if scenario == "nonlinear":
        return eta_nonlinear(x, a)
    if scenario == "strong_nonlinear":
        return eta_strong_nonlinear(x, a)
    if scenario == "nonproportional":
        # The actual non-proportional mechanism is interval-specific.  This
        # fallback is used only by diagnostics that require a scalar eta.
        return nonph_eta_by_interval(x, a).mean(axis=1)
    if scenario == "outcome_drift":
        # The target follows the linear baseline mechanism.  The source has a
        # treatment-specific conditional outcome drift, so calibration of X
        # alone cannot recover the target survival function.
        base = eta_linear(x, a)
        if population == "source":
            drift = (0.55 if a == 0 else -0.35) + 0.25 * x[:, 2]
            return base + drift
        return base
    raise ValueError(f"Unknown DGP scenario: {scenario}")


def eta_censor_dgp(x, a, scenario):
    if scenario == "linear":
        return 0.25 * a + 0.30 * x[:, 0] - 0.18 * x[:, 1] + 0.12 * x[:, 3]
    return 0.25 * a + 0.30 * x[:, 0] - 0.18 * x[:, 1] ** 2 + 0.12 * x[:, 3]


def sample_weibull(scale, shape, eta, rng):
    v = np.clip(rng.uniform(size=len(eta)), 1e-12, 1.0)
    return scale * ((-np.log(v)) / np.exp(np.clip(eta, -8, 8))) ** (1.0 / shape)


def generate_source(n, rng, scenario):
    if SOURCE_MODE == "target":
        x = sample_x(n, MU_T, rng)
    elif SOURCE_MODE == "shifted":
        x = sample_source_x(n, rng)
    else:
        raise ValueError(f"Unknown source mode: {SOURCE_MODE}")
    a = rng.binomial(1, propensity_dgp(x))
    if scenario == "nonproportional":
        t = sample_nonph_event(x, a, rng)
    else:
        t0 = sample_weibull(16.0, 1.35, eta_dgp(x, 0, scenario, population="source"), rng)
        t1 = sample_weibull(18.0, 1.25, eta_dgp(x, 1, scenario, population="source"), rng)
        t = np.where(a == 1, t1, t0)
    eta_c = eta_censor_dgp(x, a, scenario)
    c = sample_weibull(22.0, 1.10, eta_c, rng)
    u = np.minimum(t, c)
    delta = (t <= c).astype(int)
    return x, a, u, delta


def true_cnb_grid(x, scenario, chunk=4000, n_intervals=None):
    """High-resolution approximation used only to audit numerical convergence."""
    n_intervals = (
        TRUTH_DIAGNOSTIC_INTERVALS if n_intervals is None else int(n_intervals)
    )
    out = np.empty(len(x))
    for lo in range(0, len(x), chunk):
        xx = x[lo:lo + chunk]
        d0 = event_hazard_grid_true(xx, 0, scenario, n_intervals=n_intervals)
        d1 = event_hazard_grid_true(xx, 1, scenario, n_intervals=n_intervals)
        s0 = np.column_stack([np.ones(len(xx)), np.exp(-np.cumsum(d0, axis=1)[:, :-1])])
        s1 = np.column_stack([np.ones(len(xx)), np.exp(-np.cumsum(d1, axis=1)[:, :-1])])
        out[lo:lo + len(xx)] = np.sum(s0 * s1 * (d0 - d1), axis=1)
    return out


def true_cnb(x, scenario, chunk=2000, quadrature_order=8):
    """Continuous-time CNB truth under the Weibull event-time DGP.

    Gauss-Legendre quadrature integrates S0(t|x)S1(t|x){lambda0(t|x)-
    lambda1(t|x)} over [0, TAU], matching the independent-counterfactual CNB
    definition rather than the estimator's fixed-grid approximation.
    """
    x = np.asarray(x, dtype=float)
    if scenario == "nonproportional":
        edges = np.linspace(0.0, TAU, N_INTERVALS + 1)
        times = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        out = np.empty(len(x))
        for lo in range(0, len(x), chunk):
            xx = x[lo:lo + chunk]
            h0 = nonph_hazard_rates(xx, 0)
            h1 = nonph_hazard_rates(xx, 1)
            idx = np.minimum(np.searchsorted(NONPH_EDGES[1:], times), len(NONPH_WIDTHS) - 1)
            d0 = h0[:, idx]
            d1 = h1[:, idx]
            s0 = np.exp(-(np.cumsum(d0 * widths[None, :], axis=1) - 0.5 * d0 * widths[None, :]))
            s1 = np.exp(-(np.cumsum(d1 * widths[None, :], axis=1) - 0.5 * d1 * widths[None, :]))
            out[lo:lo + len(xx)] = np.sum(s0 * s1 * (d0 - d1) * widths[None, :], axis=1)
        return out
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    edges = np.linspace(0.0, TAU, N_INTERVALS + 1)
    widths = np.diff(edges)
    times = (edges[:-1, None] + edges[1:, None]) / 2.0 + widths[:, None] * nodes[None, :] / 2.0
    quad_weights = widths[:, None] * weights[None, :] / 2.0
    times = times.reshape(-1)
    quad_weights = quad_weights.reshape(-1)
    out = np.empty(len(x))
    for lo in range(0, len(x), chunk):
        xx = x[lo:lo + chunk]
        e0 = np.exp(np.clip(eta_dgp(xx, 0, scenario, population="target"), -8, 8))[:, None]
        e1 = np.exp(np.clip(eta_dgp(xx, 1, scenario, population="target"), -8, 8))[:, None]
        t = times[None, :]
        h0 = e0 * (1.35 / 16.0) * np.maximum(t / 16.0, 0.0) ** 0.35
        h1 = e1 * (1.25 / 18.0) * np.maximum(t / 18.0, 0.0) ** 0.25
        H0 = e0 * (np.maximum(t / 16.0, 0.0) ** 1.35)
        H1 = e1 * (np.maximum(t / 18.0, 0.0) ** 1.25)
        integrand = np.exp(-H0 - H1) * (h0 - h1)
        out[lo:lo + len(xx)] = np.sum(integrand * quad_weights[None, :], axis=1)
    return out


def true_survival_grid(x, arm, scenario):
    """True survival at interval left endpoints used by event_plugin."""
    x = np.asarray(x, float)
    if scenario == "nonproportional":
        d = event_hazard_grid_true(x, arm, scenario)
        return np.column_stack([np.ones(len(x)), np.exp(-np.cumsum(d, axis=1)[:, :-1])])
    edges = np.linspace(0.0, TAU, N_INTERVALS + 1)[:-1]
    scale, shape = ((16.0, 1.35) if arm == 0 else (18.0, 1.25))
    eta = eta_strong_nonlinear(x, arm) if scenario == "strong_nonlinear" else eta_dgp(x, arm, scenario, population="target")
    return np.exp(-((edges[None, :] / scale) ** shape) * np.exp(np.clip(eta[:, None], -8, 8)))


def features_linear(x):
    return np.asarray(x, float)


def features_flexible(x):
    x = np.asarray(x, float)
    return np.column_stack([x, x ** 2, x[:, 0] * x[:, 1], x[:, 2] * x[:, 4]])


def features_p(x, learner):
    return np.column_stack([np.ones(len(x)), features_linear(x) if learner == "logistic" else features_flexible(x)])


def features_censor(x, a, learner):
    if learner == "cox":
        # Nest both censoring DGPs: the linear scenario uses X2, whereas the
        # nonlinear scenarios use X2^2.
        return np.column_stack([a, x[:, 0], x[:, 1], x[:, 1] ** 2, x[:, 3]])
    return np.column_stack([a, x, x ** 2, a[:, None] * x])


def make_folds(n, k, rng):
    fold = np.empty(n, int)
    fold[rng.permutation(n)] = np.arange(n) % k
    return fold


def mtlr_time_grid(u, delta):
    """Construct the MTLR time grid used by both fitting and prediction."""
    if MTLR_GRID == "uniform":
        return np.linspace(0.0, TAU, N_INTERVALS + 1)
    if MTLR_GRID != "event_quantile":
        raise ValueError(f"Unknown MTLR grid: {MTLR_GRID}")
    event_times = np.asarray(u, dtype=float)[(np.asarray(delta) == 1) & (np.asarray(u) > 0.0) & (np.asarray(u) <= TAU)]
    if len(event_times) < max(20, N_INTERVALS):
        return np.linspace(0.0, TAU, N_INTERVALS + 1)
    grid = np.quantile(event_times, np.linspace(0.0, 1.0, N_INTERVALS + 1))
    grid[0], grid[-1] = 0.0, TAU
    grid = np.maximum.accumulate(grid)
    # Duplicate empirical quantiles create zero-width bins. Fall back to the
    # regular grid when too many ties would make the PMF poorly conditioned.
    if np.sum(np.diff(grid) > 1e-8) < max(8, N_INTERVALS // 2):
        return np.linspace(0.0, TAU, N_INTERVALS + 1)
    return grid


def aggregate_baseline(times, inc, edges):
    out = np.zeros(len(edges) - 1)
    for t, h in zip(times, inc):
        if t <= TAU:
            j = min(np.searchsorted(edges[1:], t, side="left"), len(out) - 1)
            out[j] += h
    return out


def continuous_integration_grid(u, delta):
    """Observed event times augmented by a deterministic refining grid."""
    n_intervals = max(
        MIN_INTEGRATION_INTERVALS,
        int(np.ceil(len(u) ** INTEGRATION_GRID_EXPONENT)),
    )
    regular = np.linspace(TAU / n_intervals, TAU, n_intervals)
    event_times = np.asarray(u)[
        (np.asarray(delta) == 1)
        & (np.asarray(u) > 0.0)
        & (np.asarray(u) <= TAU)
    ]
    return np.sort(np.unique(np.r_[regular, event_times]))


def fit_event_cox(x, a, u, delta):
    models = {}
    for arm in (0, 1):
        m = a == arm
        z = features_linear(x[m])
        beta = fit_cox(z, u[m], delta[m])
        t, h = breslow(z, u[m], delta[m], beta)
        keep = (t > 0.0) & (t <= TAU)
        models[arm] = {
            "beta": beta,
            "times": np.asarray(t[keep], dtype=float),
            "increments": np.asarray(h[keep], dtype=float),
        }
    event_grid = continuous_integration_grid(u, delta)
    return {"kind": "cox", "models": models, "time_grid": event_grid}


def fit_event_rsf(x, a, u, delta):
    """Fit arm-specific continuous-time random survival forests."""
    models = {}
    for arm in (0, 1):
        mask = a == arm
        y = Surv.from_arrays(delta[mask].astype(bool), u[mask].astype(float))
        model = RandomSurvivalForest(
            n_estimators=RSF_TREES,
            min_samples_split=2 * RSF_MIN_LEAF,
            min_samples_leaf=RSF_MIN_LEAF,
            max_features=RSF_MAX_FEATURES,
            n_jobs=RSF_N_JOBS,
            random_state=SEED + 5000 * arm + len(x),
            low_memory=False,
        )
        models[arm] = model.fit(np.asarray(x[mask], dtype=float), y)
    event_grid = continuous_integration_grid(u, delta)
    return {
        "kind": "rsf",
        "models": models,
        "time_grid": event_grid,
        "prediction_mode": RSF_PREDICTION_MODE,
    }


def fit_event_gbsa(x, a, u, delta):
    """Fit arm-specific nonlinear gradient-boosted survival models."""
    models = {}
    for arm in (0, 1):
        mask = a == arm
        y = Surv.from_arrays(delta[mask].astype(bool), u[mask].astype(float))
        model = GradientBoostingSurvivalAnalysis(
            loss="coxph",
            learning_rate=GBSA_LEARNING_RATE,
            n_estimators=GBSA_TREES,
            max_depth=GBSA_MAX_DEPTH,
            min_samples_leaf=GBSA_MIN_LEAF,
            random_state=SEED + 7000 * arm + len(x),
        )
        models[arm] = model.fit(np.asarray(x[mask], dtype=float), y)
    event_grid = continuous_integration_grid(u, delta)
    return {"kind": "gbsa", "models": models, "time_grid": event_grid}


def _sieve_features(x, transformer, center, scale):
    x = np.asarray(x, dtype=float)
    splines = transformer.transform(x)
    interactions = np.column_stack([
        x[:, i] * x[:, j]
        for i in range(x.shape[1])
        for j in range(i + 1, x.shape[1])
    ])
    design = np.column_stack([x, splines, interactions])
    return (design - center) / scale


def fit_event_sieve_cox(x, a, u, delta):
    """Penalized spline-sieve Cox nuisance model with pairwise interactions."""
    x = np.asarray(x, dtype=float)
    transformer = SplineTransformer(
        n_knots=SIEVE_N_KNOTS,
        degree=3,
        include_bias=False,
        extrapolation="linear",
    ).fit(x)
    raw_splines = transformer.transform(x)
    interactions = np.column_stack([
        x[:, i] * x[:, j]
        for i in range(x.shape[1])
        for j in range(i + 1, x.shape[1])
    ])
    raw_design = np.column_stack([x, raw_splines, interactions])
    center = raw_design.mean(axis=0)
    scale = np.maximum(raw_design.std(axis=0, ddof=1), 1e-6)
    z = (raw_design - center) / scale
    models = {}
    for arm in (0, 1):
        mask = a == arm
        y = Surv.from_arrays(
            delta[mask].astype(bool), u[mask].astype(float)
        )
        models[arm] = CoxPHSurvivalAnalysis(
            alpha=SIEVE_COX_RIDGE,
            ties="breslow",
            n_iter=200,
            tol=1e-8,
        ).fit(z[mask], y)
    event_grid = continuous_integration_grid(u, delta)
    return {
        "kind": "sieve_cox",
        "models": models,
        "transformer": transformer,
        "center": center,
        "scale": scale,
        "time_grid": event_grid,
    }


def predict_sieve_cumhazard(x, fitted, arm, times):
    z = _sieve_features(
        x, fitted["transformer"], fitted["center"], fitted["scale"]
    )
    functions = fitted["models"][arm].predict_cumulative_hazard_function(
        z, return_array=False
    )
    return _step_function_matrix(functions, times)


def _step_function_matrix(step_functions, times, initial=0.0):
    """Evaluate right-continuous step functions at arbitrary query times."""
    times = np.asarray(times, dtype=float)
    out = np.full((len(step_functions), len(times)), float(initial), dtype=float)
    for i, fn in enumerate(step_functions):
        idx = np.searchsorted(np.asarray(fn.x, dtype=float), times, side="right") - 1
        valid = idx >= 0
        if np.any(valid):
            # sksurv StepFunction objects represent f(t) = a * y(t) + b.
            # In particular, Cox-type GBSA predictions store the subject-specific
            # relative-risk multiplier in ``a`` while sharing the baseline values
            # in ``y``.  Reading ``y`` alone therefore collapses all subjects to
            # the same cumulative hazard.
            multiplier = float(getattr(fn, "a", 1.0))
            offset = float(getattr(fn, "b", 0.0))
            out[i, valid] = (
                multiplier * np.asarray(fn.y, dtype=float)[idx[valid]] + offset
            )
    return out


def predict_rsf_survival(x, fitted, arm, times):
    functions = fitted["models"][arm].predict_survival_function(
        np.asarray(x, dtype=float),
        return_array=False,
    )
    survival = _step_function_matrix(functions, times, initial=1.0)
    survival = np.minimum.accumulate(np.clip(survival, 1e-8, 1.0), axis=1)
    return survival


def predict_rsf_cumhazard(x, fitted, arm, times):
    if fitted.get("prediction_mode", "cumhaz") == "survival":
        return -np.log(predict_rsf_survival(x, fitted, arm, times))
    functions = fitted["models"][arm].predict_cumulative_hazard_function(
        np.asarray(x, dtype=float),
        return_array=False,
    )
    return _step_function_matrix(functions, times, initial=0.0)


def predict_gbsa_cumhazard(x, fitted, arm, times):
    functions = fitted["models"][arm].predict_cumulative_hazard_function(
        np.asarray(x, dtype=float),
        return_array=False,
    )
    return _step_function_matrix(functions, times)


def _km_censor_survival(u, delta, times):
    """Marginal Kaplan-Meier estimate of censoring survival for IPCW scoring."""
    u = np.asarray(u, dtype=float)
    censor = 1 - np.asarray(delta, dtype=int)
    unique = np.sort(np.unique(u[censor == 1]))
    survival = []
    current = 1.0
    for t in unique:
        at_risk = np.sum(u >= t)
        if at_risk:
            current *= 1.0 - np.sum((u == t) & (censor == 1)) / at_risk
        survival.append(current)
    unique = np.asarray(unique, dtype=float)
    survival = np.asarray(survival, dtype=float)
    query = np.asarray(times, dtype=float)
    idx = np.searchsorted(unique, query, side="right") - 1
    out = np.ones(len(query), dtype=float)
    valid = idx >= 0
    if np.any(valid):
        out[valid] = survival[idx[valid]]
    return np.maximum(out, CENSOR_SURVIVAL_FLOOR)


def _ipcw_brier_loss(
    pred_survival, u, delta, times, km_u, km_t, sample_weight=None
):
    """Integrated IPCW Brier loss on a validation set."""
    sample_weight = (
        np.ones(len(u), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    losses = []
    for j, t in enumerate(times):
        survived = u > t
        failed = (u <= t) & (delta == 1)
        contribution = np.zeros(len(u), dtype=float)
        g_t = km_t[survived, j] if np.ndim(km_t) == 2 else km_t[j]
        contribution[survived] = (
            (1.0 - pred_survival[survived, j]) ** 2 / g_t
        )
        contribution[failed] = (
            pred_survival[failed, j] ** 2 / km_u[failed]
        )
        losses.append(float(
            np.sum(sample_weight * contribution)
            / max(np.sum(sample_weight), 1e-12)
        ))
    return float(np.mean(losses)) if losses else np.inf


def fit_event_stack(
    x, a, u, delta, x_target=None, refit_selected_only=False,
    prefitted_candidates=None,
):
    """Outer-training-only survival Super Learner with inner OOF predictions."""
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=int)
    u = np.asarray(u, dtype=float)
    delta = np.asarray(delta, dtype=int)
    candidate_names = ("linear_cox", "rsf", "gbsa")
    weights = {}
    validation_losses = {}
    rng = np.random.default_rng(SEED + 11000 + len(x))
    inner_fold = np.empty(len(x), dtype=int)
    for arm in (0, 1):
        arm_idx = np.flatnonzero(a == arm)
        shuffled = arm_idx[rng.permutation(len(arm_idx))]
        inner_fold[shuffled] = np.arange(len(shuffled)) % STACK_INNER_FOLDS
    times = np.linspace(max(TAU / 12.0, 0.25), TAU, 12)
    inner_cache = []
    for fold in range(STACK_INNER_FOLDS):
        train = inner_fold != fold
        cached = {
            "candidates": {
                "linear_cox": fit_event_cox(
                    x[train], a[train], u[train], delta[train]
                ),
                "rsf": fit_event_rsf(
                    x[train], a[train], u[train], delta[train]
                ),
                "gbsa": fit_event_gbsa(
                    x[train], a[train], u[train], delta[train]
                ),
            },
            "censor": fit_censor(
                x[train], a[train], u[train], delta[train], "cox"
            ),
            "ratio": (
                None
                if x_target is None
                else fit_ratio_calibration(x[train], x_target)
            ),
        }
        inner_cache.append(cached)

    for arm in (0, 1):
        arm_idx = np.flatnonzero(a == arm)
        predictions = np.empty((len(arm_idx), len(times), len(candidate_names)))
        g_u = np.empty(len(arm_idx), dtype=float)
        g_t = np.empty((len(arm_idx), len(times)), dtype=float)
        validation_weight = np.empty(len(arm_idx), dtype=float)
        position = {index: pos for pos, index in enumerate(arm_idx)}
        for fold in range(STACK_INNER_FOLDS):
            val_idx = np.flatnonzero((a == arm) & (inner_fold == fold))
            cached = inner_cache[fold]
            candidates = cached["candidates"]
            val_pos = np.asarray([position[index] for index in val_idx], dtype=int)
            for index, name in enumerate(candidate_names):
                predictions[val_pos, :, index] = event_survival_on_grid(
                    x[val_idx], candidates[name], arm, times
                )
            censor = cached["censor"]
            g_t[val_pos] = np.maximum(
                np.exp(-censor_cumhazard_matrix(
                    x[val_idx], a[val_idx], censor, times
                )),
                CENSOR_SURVIVAL_FLOOR,
            )
            g_u_matrix = censor_cumhazard_matrix(
                x[val_idx], a[val_idx], censor, u[val_idx]
            )
            g_u[val_pos] = np.maximum(
                np.exp(-np.diag(g_u_matrix)), CENSOR_SURVIVAL_FLOOR
            )
            if x_target is None:
                validation_weight[val_pos] = 1.0
            else:
                ratio_model = cached["ratio"]
                validation_weight[val_pos] = normalize_capped_weights(
                    predict_ratio(x[val_idx], ratio_model), WEIGHT_CAP
                )

        def objective(theta):
            w = np.exp(theta - logsumexp(theta))
            pred = np.sum(predictions * w[None, None, :], axis=2)
            return _ipcw_brier_loss(
                pred, u[arm_idx], delta[arm_idx], times, g_u, g_t,
                sample_weight=validation_weight,
            )

        fitted = minimize(
            objective,
            np.zeros(len(candidate_names), dtype=float),
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-10},
        )
        weights[arm] = np.exp(fitted.x - logsumexp(fitted.x))
        validation_losses[arm] = {
            name: _ipcw_brier_loss(
                predictions[:, :, index],
                u[arm_idx], delta[arm_idx], times, g_u, g_t,
                sample_weight=validation_weight,
            )
            for index, name in enumerate(candidate_names)
        }
    selected_names = {
        arm: min(validation_losses[arm], key=validation_losses[arm].get)
        for arm in (0, 1)
    }
    required_candidates = (
        set(selected_names.values()) if refit_selected_only else set(candidate_names)
    )
    prefitted_candidates = prefitted_candidates or {}
    full_candidates = {}
    if "linear_cox" in required_candidates:
        full_candidates["linear_cox"] = prefitted_candidates.get(
            "linear_cox"
        ) or fit_event_cox(x, a, u, delta)
    if "rsf" in required_candidates:
        full_candidates["rsf"] = prefitted_candidates.get(
            "rsf"
        ) or fit_event_rsf(x, a, u, delta)
    if "gbsa" in required_candidates:
        full_candidates["gbsa"] = prefitted_candidates.get(
            "gbsa"
        ) or fit_event_gbsa(x, a, u, delta)
    event_grid = continuous_integration_grid(u, delta)
    return {
        "kind": "survival_stack",
        "candidates": full_candidates,
        "weights": weights,
        "candidate_names": candidate_names,
        "validation_losses": validation_losses,
        "selected_names": selected_names,
        "selection_method": "inner_oof_target_weighted_ipcw_brier",
        "inner_folds": STACK_INNER_FOLDS,
        "time_grid": event_grid,
    }


def fit_event_selector(x, a, u, delta, x_target=None):
    """Select Cox or RSF by the same inner-OOF target-weighted IPCW loss."""
    fitted = fit_event_stack(
        x, a, u, delta, x_target=x_target, refit_selected_only=True
    )
    fitted["kind"] = "survival_selector"
    return fitted


def pmf_design(x):
    z = features_flexible(x).astype(float)
    center, scale = z.mean(axis=0), z.std(axis=0)
    return (z - center) / np.maximum(scale, 1e-6), center, scale


def fit_event_discrete_hazard(x, a, u, delta):
    """Fit one non-Cox discrete-time hazard network per treatment arm."""
    models = {}
    for arm in (0, 1):
        mask = a == arm
        models[arm] = survival_ml.fit_discrete_hazard(
            x[mask], u[mask], delta[mask],
            seed=SEED + 1000 * int(arm) + len(x),
            n_intervals=N_INTERVALS,
            epochs=DISCRETE_HAZARD_EPOCHS,
            lr=DISCRETE_HAZARD_LR,
            patience=DISCRETE_HAZARD_PATIENCE,
        )
    return {"kind": "discrete_hazard", "models": models}


def fit_event_mtlr(x, a, u, delta):
    """Fit a separate or shared normalized PMF/MTLR event model."""
    if MTLR_SHARED:
        z = np.column_stack([x, a.astype(float)])
        fitted = mtlr_survival.fit_mtlr(
            z, u, delta,
            seed=SEED + 2100 + len(x),
            tau=TAU,
            n_intervals=N_INTERVALS,
            epochs=DISCRETE_HAZARD_EPOCHS,
            patience=DISCRETE_HAZARD_PATIENCE,
            hidden_dims=MTLR_HIDDEN,
            weight_decay=MTLR_WEIGHT_DECAY,
            grad_clip=MTLR_GRAD_CLIP,
            ensemble_seeds=MTLR_ENSEMBLE,
            time_grid=mtlr_time_grid(u, delta),
        )
        return {"kind": "mtlr_shared", "model": fitted}
    models = {}
    for arm in (0, 1):
        mask = a == arm
        models[arm] = mtlr_survival.fit_mtlr(
            x[mask], u[mask], delta[mask],
            seed=SEED + 2000 * int(arm) + len(x),
            tau=TAU,
            n_intervals=N_INTERVALS,
            epochs=DISCRETE_HAZARD_EPOCHS,
            patience=DISCRETE_HAZARD_PATIENCE,
            hidden_dims=MTLR_HIDDEN,
            weight_decay=MTLR_WEIGHT_DECAY,
            grad_clip=MTLR_GRAD_CLIP,
            ensemble_seeds=MTLR_ENSEMBLE,
            time_grid=mtlr_time_grid(u[mask], delta[mask]),
        )
    return {"kind": "mtlr", "models": models}


def pmf_hazard(x, model):
    z = (features_flexible(x) - model["center"]) / np.maximum(model["scale"], 1e-6)
    p = softmax(z @ model["beta"])
    mass = p[:, :-1]
    surv_before = np.column_stack([np.ones(len(x)), 1.0 - np.cumsum(mass, axis=1)[:, :-1]])
    hazard = np.clip(mass / np.maximum(surv_before, 1e-6), 1e-6, 0.98)
    return -np.log(1.0 - hazard)


def event_hazard_matrix(x, model, arm):
    if model["kind"] == "oracle":
        return event_hazard_grid_true(x, arm, model["scenario"])
    if model["kind"] == "oracle_scaled":
        return (
            float(model["scale"][arm])
            * event_hazard_grid_true(x, arm, model["scenario"])
        )
    if model["kind"] == "hybrid":
        return event_hazard_matrix(x, model["arm_models"][arm], arm)
    if model["kind"] == "survival_selector":
        selected = model["selected_names"][arm]
        return event_hazard_matrix(x, model["candidates"][selected], arm)
    if model["kind"] == "discrete_hazard":
        return np.minimum(
            survival_ml.predict_dlambda_matrix(x, model["models"][arm]),
            EVENT_HAZARD_INCREMENT_CAP,
        )
    if model["kind"] == "mtlr":
        return np.minimum(
            mtlr_survival.predict_dlambda(x, model["models"][arm]),
            EVENT_HAZARD_INCREMENT_CAP,
        )
    if model["kind"] == "mtlr_shared":
        z = np.column_stack([x, np.full(len(x), float(arm))])
        return np.minimum(
            mtlr_survival.predict_dlambda(z, model["model"]),
            EVENT_HAZARD_INCREMENT_CAP,
        )
    if model["kind"] == "rsf":
        cumulative = predict_rsf_cumhazard(x, model, arm, model["time_grid"])
        increments = np.diff(np.column_stack([np.zeros(len(x)), cumulative]), axis=1)
        return np.minimum(np.maximum(increments, 0.0), EVENT_HAZARD_INCREMENT_CAP)
    if model["kind"] == "gbsa":
        cumulative = predict_gbsa_cumhazard(x, model, arm, model["time_grid"])
        increments = np.diff(np.column_stack([np.zeros(len(x)), cumulative]), axis=1)
        return np.minimum(np.maximum(increments, 0.0), EVENT_HAZARD_INCREMENT_CAP)
    if model["kind"] == "survival_stack":
        survival = np.zeros((len(x), len(model["time_grid"])), dtype=float)
        for weight, name in zip(model["weights"][arm], model["candidate_names"]):
            survival += weight * event_survival_on_grid(
                x, model["candidates"][name], arm, model["time_grid"]
            )
        cumulative = -np.log(np.clip(survival, 1e-8, 1.0))
        increments = np.diff(np.column_stack([np.zeros(len(x)), cumulative]), axis=1)
        return np.minimum(np.maximum(increments, 0.0), EVENT_HAZARD_INCREMENT_CAP)
    if model["kind"] == "sieve_cox":
        cumulative = predict_sieve_cumhazard(
            x, model, arm, model["time_grid"]
        )
        increments = np.diff(
            np.column_stack([np.zeros(len(x)), cumulative]), axis=1
        )
        return np.minimum(
            np.maximum(increments, 0.0), EVENT_HAZARD_INCREMENT_CAP
        )
    if model["kind"] == "pmf":
        return pmf_hazard(x, model["models"][arm])
    arm_model = model["models"][arm]
    grid = np.asarray(model["time_grid"], dtype=float)
    base = np.zeros(len(grid), dtype=float)
    if len(arm_model["times"]):
        idx = np.searchsorted(grid, arm_model["times"])
        np.add.at(base, idx, arm_model["increments"])
    return np.minimum(
        np.exp(np.clip(features_linear(x) @ arm_model["beta"], -8, 8))[:, None] * base[None, :],
        EVENT_HAZARD_INCREMENT_CAP,
    )


def event_plugin(x, model):
    d0, d1 = event_hazard_matrix(x, model, 0), event_hazard_matrix(x, model, 1)
    s0 = np.column_stack([np.ones(len(x)), np.exp(-np.cumsum(d0, axis=1)[:, :-1])])
    s1 = np.column_stack([np.ones(len(x)), np.exp(-np.cumsum(d1, axis=1)[:, :-1])])
    return np.sum(s0 * s1 * (d0 - d1), axis=1)


def event_time_grid(model):
    if model["kind"] in {"oracle", "oracle_scaled", "hybrid"}:
        return np.linspace(0.0, TAU, N_INTERVALS + 1)[1:]
    if "time_grid" in model:
        return np.asarray(model["time_grid"], dtype=float)
    return np.linspace(0.0, TAU, N_INTERVALS + 1)[1:]


def event_survival_on_grid(x, model, arm, times):
    """Evaluate predicted event survival at arbitrary times for diagnostics."""
    times = np.asarray(times, dtype=float)
    if model["kind"] == "rsf":
        if model.get("prediction_mode", "cumhaz") == "survival":
            return predict_rsf_survival(x, model, arm, times)
        return np.exp(-predict_rsf_cumhazard(x, model, arm, times))
    if model["kind"] == "gbsa":
        return np.exp(-predict_gbsa_cumhazard(x, model, arm, times))
    if model["kind"] == "survival_stack":
        out = np.zeros((len(x), len(times)), dtype=float)
        for weight, name in zip(model["weights"][arm], model["candidate_names"]):
            out += weight * event_survival_on_grid(
                x, model["candidates"][name], arm, times
            )
        return np.clip(out, 0.0, 1.0)
    if model["kind"] == "survival_selector":
        selected = model["selected_names"][arm]
        return event_survival_on_grid(
            x, model["candidates"][selected], arm, times
        )
    d_lambda = event_hazard_matrix(x, model, arm)
    source_grid = event_time_grid(model)
    cumulative = np.cumsum(d_lambda, axis=1)
    idx = np.searchsorted(source_grid, times, side="right") - 1
    out = np.ones((len(x), len(times)), dtype=float)
    valid = idx >= 0
    if np.any(valid):
        out[:, valid] = np.exp(-cumulative[:, idx[valid]])
    return out


def fit_censor(x, a, u, delta, learner):
    if learner == "rsf":
        fitted = fit_event_rsf(x, a, u, 1 - delta)
        return {"kind": "rsf", "model": fitted, "learner": learner}
    if learner == "mtlr":
        z = features_censor(x, a, "flexible")
        fitted = mtlr_survival.fit_mtlr(
            z, u, 1 - delta,
            seed=SEED + 3000 + len(x),
            tau=TAU,
            n_intervals=N_INTERVALS,
            epochs=DISCRETE_HAZARD_EPOCHS,
            patience=DISCRETE_HAZARD_PATIENCE,
            hidden_dims=MTLR_HIDDEN,
            weight_decay=MTLR_WEIGHT_DECAY,
            grad_clip=MTLR_GRAD_CLIP,
            ensemble_seeds=MTLR_ENSEMBLE,
            time_grid=mtlr_time_grid(u, 1 - delta),
        )
        return {"kind": "mtlr", "model": fitted, "learner": learner}
    if learner == "discrete_hazard":
        z = features_censor(x, a, "flexible")
        fitted = survival_ml.fit_discrete_hazard(
            z, u, 1 - delta,
            seed=SEED + 3000 + len(x),
            n_intervals=N_INTERVALS,
            epochs=DISCRETE_HAZARD_EPOCHS,
            lr=DISCRETE_HAZARD_LR,
            patience=DISCRETE_HAZARD_PATIENCE,
        )
        return {"kind": "discrete_hazard", "model": fitted, "learner": learner}
    z = features_censor(x, a, learner)
    beta = fit_cox(z, u, 1 - delta)
    t, h = breslow(z, u, 1 - delta, beta)
    return {"beta": beta, "times": t, "increments": h, "learner": learner}


def censor_eta(x, a, model):
    return features_censor(x, a, model["learner"]) @ model["beta"]


def censor_dlambda_matrix(x, a, model):
    z = features_censor(x, a, "flexible")
    if model.get("kind") == "mtlr":
        return np.minimum(
            mtlr_survival.predict_dlambda(z, model["model"]),
            CENSOR_HAZARD_INCREMENT_CAP,
        )
    return np.minimum(
        survival_ml.predict_dlambda_matrix(z, model["model"]),
        CENSOR_HAZARD_INCREMENT_CAP,
    )


def censor_cumhazard_matrix(x, a, model, times):
    """Evaluate conditional censoring cumulative hazards on an arbitrary grid."""
    times = np.asarray(times, dtype=float)
    if model.get("kind") == "rsf":
        out = np.zeros((len(x), len(times)), dtype=float)
        for arm in (0, 1):
            mask = np.asarray(a) == arm
            if np.any(mask):
                out[mask] = predict_rsf_cumhazard(
                    np.asarray(x)[mask], model["model"], arm, times
                )
        return out
    if model.get("kind") in ("discrete_hazard", "mtlr"):
        d_lambda = censor_dlambda_matrix(x, a, model)
        source_grid = np.linspace(0.0, TAU, d_lambda.shape[1] + 1)[1:]
        cumulative = np.cumsum(d_lambda, axis=1)
        idx = np.searchsorted(source_grid, times, side="right") - 1
        out = np.zeros((len(x), len(times)), dtype=float)
        valid = idx >= 0
        if np.any(valid):
            out[:, valid] = cumulative[:, idx[valid]]
        return out
    if model.get("kind") == "oracle":
        eta = eta_censor_dgp(x, a, model["scenario"])
        return np.exp(np.clip(eta, -10, 10))[:, None] * (times[None, :] / 22.0) ** 1.10
    base_times = np.asarray(model["times"], dtype=float)
    base_cumulative = np.cumsum(np.asarray(model["increments"], dtype=float))
    idx = np.searchsorted(base_times, times, side="right") - 1
    h0 = np.zeros(len(times), dtype=float)
    valid = idx >= 0
    if np.any(valid):
        h0[valid] = base_cumulative[idx[valid]]
    return np.exp(np.clip(censor_eta(x, a, model), -10, 10))[:, None] * h0[None, :]


def calibration_basis(x, basis_model=None, basis_kind=None):
    """Calibration basis used for the source-to-target density ratio."""
    x = np.asarray(x, dtype=float)
    basis_kind = CALIBRATION_BASIS_KIND if basis_kind is None else basis_kind
    if basis_kind == "linear":
        return x
    if basis_kind == "quadratic":
        interactions = np.column_stack([
            x[:, i] * x[:, j]
            for i in range(x.shape[1])
            for j in range(i + 1, x.shape[1])
        ])
        return np.column_stack([x, x ** 2, interactions])
    if basis_kind == "spline":
        if basis_model is None:
            raise ValueError("Spline calibration requires a fitted basis model.")
        return basis_model.transform(x)
    raise ValueError(f"Unknown calibration basis: {basis_kind}")


def calibration_design(
    x,
    guide_model=None,
    basis_model=None,
    basis_kind=None,
    guide_times=None,
):
    basis = calibration_basis(
        x,
        basis_model=basis_model,
        basis_kind=basis_kind,
    )
    if guide_model is None:
        return basis
    x = np.asarray(x, dtype=float)
    times = np.asarray(
        CALIBRATION_GUIDE_TIMES if guide_times is None else guide_times,
        dtype=float,
    )
    return np.column_stack([
        basis,
        event_plugin(x, guide_model),
        event_survival_on_grid(x, guide_model, 0, times),
        event_survival_on_grid(x, guide_model, 1, times),
    ])


def _effective_calibration_ridge(n_source):
    if CALIBRATION_RIDGE_RATE == "fixed":
        return float(CALIBRATION_RIDGE)
    if CALIBRATION_RIDGE_RATE == "sqrt_n":
        return float(CALIBRATION_RIDGE) / np.sqrt(max(int(n_source), 1))
    raise ValueError(f"Unknown calibration ridge rate: {CALIBRATION_RIDGE_RATE}")


def fit_ratio_calibration(x_s, x_t, guide_model=None):
    basis_kind = CALIBRATION_BASIS_KIND
    basis_model = None
    if basis_kind == "spline":
        basis_model = SplineTransformer(
            n_knots=CALIBRATION_SPLINE_KNOTS,
            degree=2,
            include_bias=False,
            extrapolation="linear",
        ).fit(np.vstack([x_s, x_t]))
    guide_times = CALIBRATION_GUIDE_TIMES.copy() if guide_model is not None else None
    bs = calibration_design(
        x_s, guide_model, basis_model, basis_kind, guide_times=guide_times
    )
    bt = calibration_design(
        x_t, guide_model, basis_model, basis_kind, guide_times=guide_times
    )
    center, scale = bs.mean(0), np.maximum(bs.std(0, ddof=1), 1e-6)
    bs, bt = (bs - center) / scale, (bt - center) / scale
    target = bt.mean(0)
    ridge = _effective_calibration_ridge(len(x_s))
    penalty = np.full(bs.shape[1], ridge, dtype=float)
    if guide_model is not None:
        guide_feature_count = 1 + 2 * len(guide_times)
        penalty[-guide_feature_count:] *= CALIBRATION_GUIDE_PENALTY_FACTOR
    else:
        guide_feature_count = 0

    def objective_and_gradient(lam):
        score = bs @ lam
        log_normalizer = logsumexp(score) - np.log(len(bs))
        probability = np.exp(score - logsumexp(score))
        objective = (
            log_normalizer - target @ lam
            + 0.5 * np.sum(penalty * lam ** 2)
        )
        gradient = probability @ bs - target + penalty * lam
        return float(objective), np.asarray(gradient, dtype=float)

    fitted = minimize(
        objective_and_gradient,
        np.zeros(bs.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success and np.linalg.norm(fitted.jac, ord=np.inf) > 1e-5:
        raise RuntimeError(f"Soft calibration failed: {fitted.message}")
    lam = np.asarray(fitted.x, dtype=float)
    normalizer = float(np.exp(logsumexp(bs @ lam) - np.log(len(bs))))
    return {"kind": "calibration", "lam": lam, "center": center,
            "scale": scale, "normalizer": normalizer,
             "ridge": ridge, "penalty": penalty,
             "basis_model": basis_model,
             "basis_kind": basis_kind,
             "guide_model": guide_model,
             "guide_times": guide_times,
             "guide_feature_count": guide_feature_count,
            "guided": guide_model is not None,
            "optimizer_success": bool(fitted.success)}


def fit_ratio_oracle():
    """Return the exact source-to-target ratio under the known selection DGP."""
    # E_*[pi_S(X)] has no closed form for the logistic-normal selection model;
    # use a large deterministic Monte Carlo approximation only for benchmarking.
    rng = np.random.default_rng(SEED + 700)
    x = sample_x(250000, MU_STAR, rng)
    return {"kind": "oracle", "selection_mean": float(source_inclusion_prob(x).mean())}


def predict_ratio(x, model):
    if model["kind"] == "identity":
        return np.ones(len(x), dtype=float)
    if model["kind"] == "oracle":
        return model["selection_mean"] / source_inclusion_prob(np.asarray(x, dtype=float))
    if model["kind"] == "logistic":
        return np.exp(np.clip(features_p(x, "logistic") @ model["beta"], -10, 10)) * model["prior"]
    z = (
        calibration_design(
            x,
            model.get("guide_model"),
            model.get("basis_model"),
            model.get("basis_kind"),
            guide_times=model.get("guide_times"),
        )
        - model["center"]
    ) / model["scale"]
    w = np.exp(np.clip(z @ model["lam"], -20, 20))
    return w / max(model["normalizer"], 1e-12)


def normalize_capped_weights(raw, cap):
    """Normalize positive weights to mean one while respecting a final cap."""
    raw = np.maximum(np.asarray(raw, dtype=float), 0.0)
    if not np.any(raw > 0):
        return np.ones_like(raw)
    if cap is None or not np.isfinite(cap):
        return raw / max(float(raw.mean()), 1e-12)
    cap = float(cap)
    if cap < 1.0:
        raise ValueError("Weight cap must be at least one after normalization.")
    low, high = 0.0, 1.0 / max(float(raw.mean()), 1e-12)
    while np.mean(np.minimum(high * raw, cap)) < 1.0:
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if np.mean(np.minimum(mid * raw, cap)) < 1.0:
            low = mid
        else:
            high = mid
    return np.minimum(high * raw, cap)


def cnB_eif_grid(x, a, u, delta, event_model, censor_model, p_beta, p_learner,
                 return_parts=False):
    if event_model["kind"] in {"mtlr", "mtlr_shared", "pmf", "discrete_hazard", "hazard_net"}:
        raise ValueError(
            "Discrete-time event models require a discrete-time CNB EIF and cannot "
            "be inserted into the continuous-time counting-process correction."
        )
    grid = event_time_grid(event_model)
    n, k = len(x), len(grid)
    d0, d1 = event_hazard_matrix(x, event_model, 0), event_hazard_matrix(x, event_model, 1)
    censor_cumulative = censor_cumhazard_matrix(x, a, censor_model, grid)
    if p_learner == "oracle":
        p = propensity_dgp(x)
    else:
        if isinstance(p_beta, dict):
            design = (
                features_p(x, p_learner) - p_beta["center"]
            ) / p_beta["scale"]
            beta = p_beta["beta"]
        else:
            design = features_p(x, p_learner)
            beta = p_beta
        p = np.clip(expit(design @ beta), 0.02, 0.98)
    out = np.zeros(n)
    first10_all = np.zeros(n)
    second10_all = np.zeros(n)
    first01_all = np.zeros(n)
    second01_all = np.zeros(n)
    event_survival_floor_hits = 0
    censor_survival_floor_hits = 0
    risk_grid_count = 0
    for i in range(n):
        dL0, dL1 = d0[i], d1[i]
        s0_raw = np.r_[1.0, np.exp(-np.cumsum(dL0)[:-1])]
        s1_raw = np.r_[1.0, np.exp(-np.cumsum(dL1)[:-1])]
        event_survival_floor_hits += int(np.sum(s0_raw < EVENT_SURVIVAL_FLOOR))
        event_survival_floor_hits += int(np.sum(s1_raw < EVENT_SURVIVAL_FLOOR))
        # Keep raw survival in the EIF residual; stabilize only inverse
        # survival factors below so truncation does not alter the plugin term.
        s0 = s0_raw
        s1 = s1_raw
        s0_inv = np.maximum(s0_raw, EVENT_SURVIVAL_FLOOR)
        s1_inv = np.maximum(s1_raw, EVENT_SURVIVAL_FLOOR)
        lam_a = dL1 if a[i] else dL0
        s_a = s1 if a[i] else s0
        s_a_inv = s1_inv if a[i] else s0_inv
        kgrid_raw = np.exp(-censor_cumulative[i])
        censor_survival_floor_hits += int(np.sum(kgrid_raw < CENSOR_SURVIVAL_FLOOR))
        kgrid = np.maximum(kgrid_raw, CENSOR_SURVIVAL_FLOOR)
        risk_grid_count += len(kgrid)
        at_risk = grid <= min(u[i], TAU)
        pi_a = p[i] if a[i] else 1.0 - p[i]
        g_a = 1.0 / max(pi_a, 1e-4)
        mt_inc = np.zeros(k)
        mt_inc[at_risk] = -lam_a[at_risk] / np.maximum(kgrid[at_risk] * s_a_inv[at_risk], 1e-8)
        mt = np.cumsum(mt_inc)
        event_window = delta[i] == 1 and u[i] <= TAU
        if event_window:
            j = min(np.searchsorted(grid, u[i], side="left"), k - 1)
            h_c_u = censor_cumhazard_matrix(
                x[i:i + 1],
                np.asarray([a[i]]),
                censor_model,
                np.asarray([u[i]]),
            )[0, 0]
            ku = max(np.exp(-h_c_u), CENSOR_SURVIVAL_FLOOR)
            sa_u = max(s_a[j], EVENT_SURVIVAL_FLOOR)
            mt[j:] += 1.0 / (ku * sa_u)
        common = s0 * s1 * mt
        first10, first01 = -g_a * np.sum(common * dL0), -g_a * np.sum(common * dL1)
        second10 = second01 = 0.0
        if a[i] == 0:
            mart = -np.sum(s1[at_risk] * dL0[at_risk] / np.maximum(kgrid[at_risk], 1e-6))
            if event_window: mart += s1[j] / ku
            second10 = mart / max(1.0 - p[i], 1e-4)
        else:
            mart = -np.sum(s0[at_risk] * dL1[at_risk] / np.maximum(kgrid[at_risk], 1e-6))
            if event_window: mart += s0[j] / ku
            second01 = mart / max(p[i], 1e-4)
        first10_all[i] = first10
        second10_all[i] = second10
        first01_all[i] = first01
        second01_all[i] = second01
        out[i] = (first10 + second10) - (first01 + second01)
    if return_parts:
        return {
            "d": out,
            "d10": first10_all + second10_all,
            "d01": first01_all + second01_all,
            "first10": first10_all,
            "second10": second10_all,
            "first01": first01_all,
            "second01": second01_all,
            "event_survival_floor_rate": event_survival_floor_hits / max(2 * n * k, 1),
            "censor_survival_floor_rate": censor_survival_floor_hits / max(risk_grid_count, 1),
        }
    return out


def multiplier_bootstrap(plugin_values, source_correction_values, estimate, seed, reps=MULTIPLIER_REPS):
    """Multiplier bootstrap using the two independent cross-fitted EIF parts."""
    if reps <= 0:
        return np.nan, np.nan, np.nan
    target_if = np.asarray(plugin_values, dtype=float) - np.mean(plugin_values)
    source_if = np.asarray(source_correction_values, dtype=float) - np.mean(source_correction_values)
    rng = np.random.default_rng(seed)
    xi_t = rng.standard_normal((reps, len(target_if)))
    xi_s = rng.standard_normal((reps, len(source_if)))
    draws = float(estimate) + xi_t @ target_if / len(target_if) + xi_s @ source_if / len(source_if)
    centered = draws - float(estimate)
    q_low, q_high = np.quantile(centered, [0.025, 0.975])
    return float(np.std(draws, ddof=1)), float(estimate - q_high), float(estimate - q_low)


def calibration_control_variate(plugin, d, r, x_s, x_t, fs, ft):
    """Cross-fitted generalized-regression adjustment for transport moments.

    For any fixed balance function h(X), E_T[h(X)] = E_S[r(X)h(X)].
    The adjustment therefore moves a fitted linear component of the source EIF
    to the target contribution without changing the population estimand when
    the transport ratio is valid. Coefficients are learned outside each fold.
    """
    plugin = np.asarray(plugin, dtype=float)
    d = np.asarray(d, dtype=float)
    r = np.asarray(r, dtype=float)
    h_s_raw = features_linear(x_s)
    h_t_raw = features_linear(x_t)
    target_add = np.zeros(len(x_t), dtype=float)
    source_adjusted = np.zeros(len(x_s), dtype=float)
    beta_norms = []

    def covariance(x, y=None):
        x = np.asarray(x, dtype=float)
        xc = x - x.mean(axis=0, keepdims=True)
        if y is None:
            return xc.T @ xc / max(len(x) - 1, 1)
        y = np.asarray(y, dtype=float)
        yc = y - y.mean()
        return xc.T @ yc / max(len(x) - 1, 1)

    for fold in range(N_FOLDS):
        s_train, s_eval = fs != fold, fs == fold
        t_train, t_eval = ft != fold, ft == fold
        center = h_s_raw[s_train].mean(axis=0)
        scale = np.maximum(h_s_raw[s_train].std(axis=0, ddof=1), 1e-6)
        h_s = (h_s_raw - center) / scale
        h_t = (h_t_raw - center) / scale
        z_s = r[s_train, None] * h_s[s_train]
        y_s = r[s_train] * d[s_train]

        lhs = covariance(z_s) / max(s_train.sum(), 1)
        lhs += covariance(h_t[t_train]) / max(t_train.sum(), 1)
        rhs = covariance(z_s, y_s) / max(s_train.sum(), 1)
        rhs -= covariance(h_t[t_train], plugin[t_train]) / max(t_train.sum(), 1)
        ridge = CONTROL_VARIATE_RIDGE * max(
            float(np.trace(lhs)) / max(lhs.shape[0], 1), 1e-8
        )
        beta = np.linalg.solve(lhs + ridge * np.eye(lhs.shape[0]), rhs)
        beta_norms.append(float(np.linalg.norm(beta)))
        target_add[t_eval] = h_t[t_eval] @ beta
        source_adjusted[s_eval] = r[s_eval] * (
            d[s_eval] - h_s[s_eval] @ beta
        )

    target_adjusted = plugin + target_add
    estimate = float(target_adjusted.mean() + source_adjusted.mean())
    se = float(np.sqrt(
        np.var(target_adjusted, ddof=1) / len(target_adjusted)
        + np.var(source_adjusted, ddof=1) / len(source_adjusted)
    ))
    return {
        "estimate": estimate,
        "se": se,
        "target_values": target_adjusted,
        "source_values": source_adjusted,
        "target_add_mean": float(target_add.mean()),
        "source_mean": float(source_adjusted.mean()),
        "source_sd": float(np.std(source_adjusted, ddof=1)),
        "target_sd": float(np.std(target_adjusted, ddof=1)),
        "mean_beta_norm": float(np.mean(beta_norms)),
    }


def one_combo(x_s, a, u, delta, x_t, fs, ft, combo, fold_cache,
              return_contributions=False, multiplier_seed=0):
    s_learner, g_learner, p_learner, pi_learner = combo
    plugin = np.empty(len(x_t)); source_plugin = np.empty(len(x_s)); d = np.empty(len(x_s)); r = np.empty(len(x_s))
    surv_pred0 = np.empty((len(x_t), N_INTERVALS)); surv_pred1 = np.empty((len(x_t), N_INTERVALS))
    d10 = np.empty(len(x_s)); d01 = np.empty(len(x_s))
    first10 = np.empty(len(x_s)); second10 = np.empty(len(x_s))
    first01 = np.empty(len(x_s)); second01 = np.empty(len(x_s))
    balance_l2 = []
    best_epochs = []
    best_val_losses = []
    full_refit_losses = []
    event_survival_floor_rates = []
    censor_survival_floor_rates = []
    propensity_ridges = []
    propensity_cv_losses = []
    stack_weights = {"linear_cox": [], "rsf": [], "gbsa": []}
    diagnostic_times = np.linspace(0.0, TAU, N_INTERVALS + 1)[:-1]
    for fold in range(N_FOLDS):
        st, se = fs != fold, fs == fold; tt, te = ft != fold, ft == fold
        cached = fold_cache[fold]
        event = cached["event"][s_learner]
        if event.get("kind") in {"survival_stack", "survival_selector"}:
            for arm in (0, 1):
                if event.get("kind") == "survival_selector":
                    selected = event["selected_names"][arm]
                    for name in event["candidate_names"]:
                        stack_weights[name].append(float(name == selected))
                else:
                    for name, weight in zip(
                        event["candidate_names"], event["weights"][arm]
                    ):
                        stack_weights[name].append(float(weight))
        target_event = event
        censor = cached["censor"][g_learner]
        p_beta = cached["p"][p_learner]
        if isinstance(p_beta, dict):
            propensity_ridges.append(float(p_beta["ridge"]))
            propensity_cv_losses.append(float(p_beta["cv_loss"]))
        ratio_key = (
            f"guided_calibration::{s_learner}"
            if pi_learner == "guided_calibration"
            else pi_learner
        )
        ratio = cached["ratio"][ratio_key]
        plugin[te] = event_plugin(x_t[te], target_event)
        source_plugin[se] = event_plugin(x_s[se], event)
        surv_pred0[te] = event_survival_on_grid(
            x_t[te], target_event, 0, diagnostic_times
        )
        surv_pred1[te] = event_survival_on_grid(
            x_t[te], target_event, 1, diagnostic_times
        )
        if event["kind"] == "mtlr_shared":
            model_iter = (event["model"],)
        elif event["kind"] in ("discrete_hazard", "hazard_net", "mtlr", "dsm"):
            model_iter = event["models"].values()
        else:
            model_iter = ()
        if model_iter:
            for arm_model in model_iter:
                best_epochs.append(float(arm_model.get("best_epoch", np.nan)))
                best_val_losses.append(float(arm_model.get("best_val_loss", np.nan)))
                full_refit_losses.append(float(arm_model.get("full_refit_loss", np.nan)))
        eif_parts = cnB_eif_grid(
            x_s[se], a[se], u[se], delta[se], event, censor, p_beta, p_learner,
            return_parts=True,
        )
        d[se] = eif_parts["d"]
        d10[se] = eif_parts["d10"]
        d01[se] = eif_parts["d01"]
        first10[se] = eif_parts["first10"]
        second10[se] = eif_parts["second10"]
        first01[se] = eif_parts["first01"]
        second01[se] = eif_parts["second01"]
        event_survival_floor_rates.append(float(eif_parts["event_survival_floor_rate"]))
        censor_survival_floor_rates.append(float(eif_parts["censor_survival_floor_rate"]))
        fold_ratio = normalize_capped_weights(
            predict_ratio(x_s[se], ratio), WEIGHT_CAP
        )
        r[se] = fold_ratio
        if ratio.get("kind") == "calibration":
            z_s = (
                calibration_design(
                    x_s[se],
                    ratio.get("guide_model"),
                    ratio.get("basis_model"),
                    ratio.get("basis_kind"),
                    guide_times=ratio.get("guide_times"),
                )
                - ratio["center"]
            ) / ratio["scale"]
            z_t = (
                calibration_design(
                    x_t[te],
                    ratio.get("guide_model"),
                    ratio.get("basis_model"),
                    ratio.get("basis_kind"),
                    guide_times=ratio.get("guide_times"),
                )
                - ratio["center"]
            ) / ratio["scale"]
            balance_gap = (r[se, None] * z_s).mean(axis=0) - z_t.mean(axis=0)
            balance_l2.append(float(np.sqrt(np.mean(balance_gap ** 2))))
    corr = r * d
    weighted_d10 = r * d10
    weighted_d01 = r * d01
    weighted_first10 = r * first10
    weighted_second10 = r * second10
    weighted_first01 = r * first01
    weighted_second01 = r * second01
    one = float(plugin.mean() + corr.mean())
    identity_residual = float(one - plugin.mean() - corr.mean())
    scenario = event_model_scenario = fold_cache[0]["event"][s_learner].get("scenario")
    if scenario is None:
        raise RuntimeError("Event model is missing the DGP scenario needed for survival diagnostics.")
    true_s0 = true_survival_grid(x_t, 0, scenario)
    true_s1 = true_survival_grid(x_t, 1, scenario)
    surv_err = np.concatenate([(surv_pred0 - true_s0).ravel(), (surv_pred1 - true_s1).ravel()])
    surv_calib = np.concatenate([surv_pred0.mean(axis=0) - true_s0.mean(axis=0), surv_pred1.mean(axis=0) - true_s1.mean(axis=0)])
    # The point estimate is the target plug-in mean plus the source EIF mean.
    # This is the complete two-sample cross-fitted EIF variance estimate, with
    # separate target and source contributions rather than a pooled variance.
    eif_se_complete = float(np.sqrt(
        np.var(plugin, ddof=1) / len(plugin) + np.var(corr, ddof=1) / len(corr)
    ))
    target_variance_component = float(np.var(plugin, ddof=1) / len(plugin))
    source_variance_component = float(np.var(corr, ddof=1) / len(corr))
    target_fold_variance = np.mean([
        np.mean((plugin[ft == fold] - np.mean(plugin[ft == fold])) ** 2)
        for fold in range(N_FOLDS)
    ])
    source_fold_variance = np.mean([
        np.mean((corr[fs == fold] - np.mean(corr[fs == fold])) ** 2)
        for fold in range(N_FOLDS)
    ])
    eif_se_fold_centered = float(np.sqrt(
        target_fold_variance / len(plugin)
        + source_fold_variance / len(corr)
    ))
    control_variate = None
    if ENABLE_CONTROL_VARIATE:
        control_variate = calibration_control_variate(
            plugin, d, r, x_s, x_t, fs, ft
        )
    se = eif_se_complete
    def q(arr, prob):
        return float(np.quantile(arr, prob))
    finite = np.isfinite(d) & np.isfinite(r) & np.isfinite(corr)
    # Reconstructing the uncapped value is not possible from the fitted model
    # after clipping; use the observed cap-hit rate as the relevant diagnostic.
    cap_hit = r >= (WEIGHT_CAP - 1e-10)
    multiplier_se, multiplier_ci_low, multiplier_ci_high = (np.nan, np.nan, np.nan)
    if return_contributions and combo[3] in {"calibration", "guided_calibration"}:
        multiplier_se, multiplier_ci_low, multiplier_ci_high = multiplier_bootstrap(
            plugin, corr, one, multiplier_seed, MULTIPLIER_REPS
        )
    out = {"source_only": float(source_plugin.mean()), "plugin": float(plugin.mean()), "one_step": one, "se": se,
            "ci_low": one - Z95 * se, "ci_high": one + Z95 * se,
            "multiplier_se": multiplier_se,
            "multiplier_ci_low": multiplier_ci_low,
            "multiplier_ci_high": multiplier_ci_high,
            "ess": float(r.sum() ** 2 / max((r ** 2).sum(), 1e-12)),
            "ratio_p99": float(np.quantile(r, .99)), "ratio_max": float(r.max()),
             "ratio_cap_hit_rate": float(np.mean(cap_hit)),
             "balance_l2": float(np.mean(balance_l2)) if balance_l2 else np.nan,
            "correction": float(corr.mean()), "correction_sd": float(np.std(corr, ddof=1)),
            "correction_p01": q(corr, .01), "correction_p99": q(corr, .99),
            "correction_abs_max": float(np.max(np.abs(corr))),
            "eif_p01": q(d, .01), "eif_p99": q(d, .99),
            "eif_abs_max": float(np.max(np.abs(d))),
            "eif_finite_rate": float(np.mean(finite)),
            "mean_d10": float(d10.mean()), "mean_d01": float(d01.mean()),
            "sd_d10": float(np.std(d10, ddof=1)), "sd_d01": float(np.std(d01, ddof=1)),
             "mean_first10": float(first10.mean()), "mean_second10": float(second10.mean()),
             "mean_first01": float(first01.mean()), "mean_second01": float(second01.mean()),
             "weighted_d10": float(weighted_d10.mean()), "weighted_d01": float(weighted_d01.mean()),
             "weighted_first10": float(weighted_first10.mean()), "weighted_second10": float(weighted_second10.mean()),
             "weighted_first01": float(weighted_first01.mean()), "weighted_second01": float(weighted_second01.mean()),
             "one_step_identity_residual": identity_residual,
             "se_fold_centered": eif_se_fold_centered,
             "ci_fold_low": one - Z95 * eif_se_fold_centered,
             "ci_fold_high": one + Z95 * eif_se_fold_centered,
             "surv_rmse": float(np.sqrt(np.mean(surv_err ** 2))),
             "surv_mae": float(np.mean(np.abs(surv_err))),
             "surv_bias": float(np.mean(surv_err)),
             "surv_calibration_abs": float(np.mean(np.abs(surv_calib))),
             "surv_calibration_max": float(np.max(np.abs(surv_calib))),
            "mean_best_epoch": float(np.nanmean(best_epochs)) if best_epochs else np.nan,
            "mean_best_val_loss": float(np.nanmean(best_val_losses)) if best_val_losses else np.nan,
            "mean_full_refit_loss": float(np.nanmean(full_refit_losses)) if full_refit_losses and np.any(np.isfinite(full_refit_losses)) else np.nan,
            "event_survival_floor_rate": float(np.mean(event_survival_floor_rates)) if event_survival_floor_rates else np.nan,
            "censor_survival_floor_rate": float(np.mean(censor_survival_floor_rates)) if censor_survival_floor_rates else np.nan,
             "event_rate": float(delta.mean())}
    out.update({
        "target_if_sd": float(np.std(plugin, ddof=1)),
        "source_if_sd": float(np.std(corr, ddof=1)),
        "target_variance_component": target_variance_component,
        "source_variance_component": source_variance_component,
        "source_variance_fraction": float(
            source_variance_component
            / max(target_variance_component + source_variance_component, 1e-12)
        ),
        "source_selection_strength": float(SOURCE_SELECTION_STRENGTH),
    })
    out["mean_propensity_ridge"] = (
        float(np.mean(propensity_ridges)) if propensity_ridges else np.nan
    )
    out["mean_propensity_cv_loss"] = (
        float(np.mean(propensity_cv_losses)) if propensity_cv_losses else np.nan
    )
    if control_variate is not None:
        out.update({
            "one_step_cv": control_variate["estimate"],
            "se_cv": control_variate["se"],
            "ci_cv_low": control_variate["estimate"] - Z95 * control_variate["se"],
            "ci_cv_high": control_variate["estimate"] + Z95 * control_variate["se"],
            "correction_cv": control_variate["source_mean"],
            "correction_cv_sd": control_variate["source_sd"],
            "target_cv_sd": control_variate["target_sd"],
            "target_cv_add_mean": control_variate["target_add_mean"],
            "control_variate_beta_norm": control_variate["mean_beta_norm"],
        })
    for name, values in stack_weights.items():
        out[f"stack_weight_{name}"] = (
            float(np.mean(values)) if values else np.nan
        )
    if return_contributions:
        out.update({
            "target_plugin_values": plugin.copy(),
            "source_eif_values": d.copy(),
            "source_d10_values": d10.copy(),
            "source_d01_values": d01.copy(),
            "source_ratio_values": r.copy(),
            "source_correction_values": corr.copy(),
            "eif_se_complete": eif_se_complete,
            "eif_se_direct": eif_se_complete,
            "eif_se_fold_centered": eif_se_fold_centered,
        })
        if control_variate is not None:
            out.update({
                "target_plugin_cv_values": control_variate["target_values"].copy(),
                "source_correction_cv_values": control_variate["source_values"].copy(),
            })
    return out


def run(label="light20", rep_start=0, save_contributions=False, multiplier_reps=None,
        output_dir=None, event_learners=None, scenarios=None):
    global MULTIPLIER_REPS
    global OUT
    if multiplier_reps is not None:
        MULTIPLIER_REPS = int(multiplier_reps)
    if output_dir is not None:
        OUT = Path(output_dir)
        OUT.mkdir(parents=True, exist_ok=True)
    event_learners = tuple(event_learners or ("linear_cox", "rsf"))
    scenarios = tuple(scenarios or ("linear", "nonlinear", "outcome_drift"))
    # Use a large, fixed target Monte Carlo sample so that the small screening
    # repetitions do not determine the bias benchmark.
    truth_rng = np.random.default_rng(SEED + 900)
    x_truth = sample_x(120000, MU_T, truth_rng)
    truths = {scenario: float(true_cnb(x_truth, scenario).mean()) for scenario in scenarios}
    x_truth_diagnostic = x_truth[:TRUTH_DIAGNOSTIC_SAMPLE_SIZE]
    numerical_truths = {}
    for scenario in scenarios:
        paired_gap = np.mean(
            true_cnb_grid(
                x_truth_diagnostic,
                scenario,
                n_intervals=TRUTH_DIAGNOSTIC_INTERVALS,
            )
            - true_cnb(x_truth_diagnostic, scenario)
        )
        numerical_truths[scenario] = float(truths[scenario] + paired_gap)
    oracle_ratio = fit_ratio_oracle()
    rows = []
    contribution_rows = []
    start = time.time()
    for scenario_id, scenario in enumerate(scenarios):
        for rep in range(N_REP):
            rep_id = rep_start + rep
            rng = np.random.default_rng(SEED + scenario_id * 100000 + rep_id)
            xs, a, u, dlt = generate_source(N_S, rng, scenario)
            xt = sample_x(N_T, MU_T, rng)
            if FOLD_SEED_OFFSET == 0:
                # Preserve the exact historical split when repeated
                # cross-fitting is not requested.
                fold_rng = rng
            else:
                fold_rng = np.random.default_rng(
                    SEED + 7000000 + scenario_id * 100000 + rep_id
                    + 104729 * FOLD_SEED_OFFSET
                )
            fs = make_folds(N_S, N_FOLDS, fold_rng)
            ft = make_folds(N_T, N_FOLDS, fold_rng)
            g_learner, p_learner, ratio_learners = SCENARIO_NUISANCE[scenario]
            if CENSOR_LEARNER_OVERRIDE is not None:
                g_learner = CENSOR_LEARNER_OVERRIDE
            if P_LEARNER_OVERRIDE is not None:
                p_learner = P_LEARNER_OVERRIDE
            if RATIO_LEARNERS_OVERRIDE is not None:
                ratio_learners = RATIO_LEARNERS_OVERRIDE
            # Cache nuisance fits shared by multiple combinations.  This keeps
            # the screen focused on model choice rather than repeated training.
            fold_cache = []
            for fold in range(N_FOLDS):
                st, se = fs != fold, fs == fold; tt, te = ft != fold, ft == fold
                cached = {"event": {}, "censor": {}, "p": {}, "ratio": {}}
                if "oracle" in event_learners:
                    cached["event"]["oracle"] = {
                        "kind": "oracle",
                        "scenario": scenario,
                    }
                if "linear_cox" in event_learners:
                    cached["event"]["linear_cox"] = fit_event_cox(xs[st], a[st], u[st], dlt[st])
                    cached["event"]["linear_cox"]["scenario"] = scenario
                if "rsf" in event_learners:
                    cached["event"]["rsf"] = fit_event_rsf(xs[st], a[st], u[st], dlt[st])
                    cached["event"]["rsf"]["scenario"] = scenario
                if "gbsa" in event_learners:
                    cached["event"]["gbsa"] = fit_event_gbsa(xs[st], a[st], u[st], dlt[st])
                    cached["event"]["gbsa"]["scenario"] = scenario
                if "sieve_cox" in event_learners:
                    cached["event"]["sieve_cox"] = fit_event_sieve_cox(
                        xs[st], a[st], u[st], dlt[st]
                    )
                    cached["event"]["sieve_cox"]["scenario"] = scenario
                if "survival_stack" in event_learners:
                    cached["event"]["survival_stack"] = fit_event_stack(
                        xs[st], a[st], u[st], dlt[st], x_target=xt[tt],
                        prefitted_candidates=cached["event"],
                    )
                    cached["event"]["survival_stack"]["scenario"] = scenario
                if "survival_selector" in event_learners:
                    cached["event"]["survival_selector"] = fit_event_selector(
                        xs[st], a[st], u[st], dlt[st], x_target=xt[tt]
                    )
                    cached["event"]["survival_selector"]["scenario"] = scenario
                if "mtlr" in event_learners:
                    cached["event"]["mtlr"] = fit_event_mtlr(xs[st], a[st], u[st], dlt[st])
                    cached["event"]["mtlr"]["scenario"] = scenario
                if g_learner == "oracle":
                    cached["censor"]["oracle"] = {
                        "kind": "oracle",
                        "scenario": scenario,
                    }
                else:
                    cached["censor"][g_learner] = fit_censor(
                        xs[st], a[st], u[st], dlt[st], g_learner
                    )
                if p_learner == "oracle":
                    cached["p"]["oracle"] = None
                else:
                    cached["p"][p_learner] = fit_propensity(
                        xs[st], a[st], p_learner
                    )
                if "calibration" in ratio_learners:
                    cached["ratio"]["calibration"] = fit_ratio_calibration(xs[st], xt[tt])
                if "guided_calibration" in ratio_learners:
                    for s_name in event_learners:
                        cached["ratio"][f"guided_calibration::{s_name}"] = (
                            fit_ratio_calibration(
                                xs[st], xt[tt],
                                guide_model=cached["event"][s_name],
                            )
                        )
                cached["ratio"]["oracle"] = oracle_ratio
                cached["ratio"]["identity"] = {"kind": "identity"}
                fold_cache.append(cached)
            combos = [
                (s_name, g_learner, p_learner, ratio_learner)
                for s_name in event_learners
                for ratio_learner in ratio_learners
            ]
            for cid, combo in enumerate(combos, 1):
                keep_contrib = save_contributions or MULTIPLIER_REPS > 0
                out = one_combo(xs, a, u, dlt, xt, fs, ft, combo, fold_cache,
                                return_contributions=keep_contrib,
                                multiplier_seed=SEED + scenario_id * 1000000 + rep_id * 100 + cid)
                row = {"scenario": scenario, "rep": rep_id, "n_s": N_S, "n_t": N_T, "fold_seed_offset": FOLD_SEED_OFFSET, "combo_id": cid, "S_learner": combo[0], "G_learner": combo[1], "p_learner": combo[2], "pi_learner": combo[3], "truth": truths[scenario], "truth_grid": numerical_truths[scenario], "truth_numerical": numerical_truths[scenario], "truth_numerical_intervals": TRUTH_DIAGNOSTIC_INTERVALS, **{
                    key: value for key, value in out.items()
                    if key not in {"target_plugin_values", "source_eif_values", "source_d10_values", "source_d01_values", "source_ratio_values", "source_correction_values", "target_plugin_cv_values", "source_correction_cv_values"}
                }}
                one_step_name = f"one_step_{combo[3]}"
                rows += [
                    {**row, "estimator": "S_only", "estimate": out["source_only"]},
                    {**row, "estimator": "plugin_T", "estimate": out["plugin"]},
                    {**row, "estimator": one_step_name, "estimate": out["one_step"]},
                ]
                if "one_step_cv" in out:
                    rows.append({
                        **row,
                        "estimator": f"{one_step_name}_cv",
                        "estimate": out["one_step_cv"],
                        "se": out["se_cv"],
                        "ci_low": out["ci_cv_low"],
                        "ci_high": out["ci_cv_high"],
                    })
                if keep_contrib:
                    for i in range(len(xs)):
                        contribution_rows.append({
                            "scenario": scenario, "rep": rep_id,
                            "S_learner": combo[0], "pi_learner": combo[3],
                            "sample_role": "source", "sample_index": int(i),
                            "target_plugin": np.nan,
                            "source_d": float(out["source_eif_values"][i]),
                            "source_d10": float(out["source_d10_values"][i]),
                            "source_d01": float(out["source_d01_values"][i]),
                            "ratio": float(out["source_ratio_values"][i]),
                            "source_correction": float(out["source_correction_values"][i]),
                            "eif_contribution": float(out["source_correction_values"][i]),
                        })
                    for i in range(len(xt)):
                        contribution_rows.append({
                            "scenario": scenario, "rep": rep_id,
                            "S_learner": combo[0], "pi_learner": combo[3],
                            "sample_role": "target", "sample_index": int(i),
                            "target_plugin": float(out["target_plugin_values"][i]),
                            "source_d": np.nan,
                            "source_d10": np.nan,
                            "source_d01": np.nan,
                            "ratio": np.nan,
                            "source_correction": np.nan,
                            "eif_contribution": float(out["target_plugin_values"][i]),
                        })
            print(f"{scenario} rep {rep_id + 1:03d}; elapsed {time.time() - start:.1f}s", flush=True)
    raw = pd.DataFrame(rows)
    raw_path = OUT / f"dgp_comparison_{label}_raw.csv"
    summary_path = OUT / f"dgp_comparison_{label}_summary.csv"
    report_path = OUT / f"dgp_comparison_{label}_report.md"
    raw.to_csv(raw_path, index=False)
    if save_contributions:
        contribution_path = OUT / f"dgp_comparison_{label}_eif_contributions.csv"
        pd.DataFrame(contribution_rows).to_csv(contribution_path, index=False)
    groups = ["scenario", "combo_id", "S_learner", "G_learner", "p_learner", "pi_learner", "estimator"]
    out_rows = []
    for keys, g in raw.groupby(groups, sort=True):
        vals, tr = g.estimate.to_numpy(), float(g.truth.iloc[0]); os = str(keys[-1]).startswith("one_step_")
        has_multiplier = (
            os
            and str(keys[5]) in {"calibration", "guided_calibration"}
            and g.multiplier_ci_low.notna().any()
        )
        out_rows.append({
            **dict(zip(groups, keys)), "truth": tr,
            "n_s": int(g.n_s.iloc[0]), "n_t": int(g.n_t.iloc[0]),
            "mean_estimate": vals.mean(), "bias": vals.mean() - tr,
            "truth_grid": float(g.truth_grid.iloc[0]),
            "grid_truth_gap": tr - float(g.truth_grid.iloc[0]),
            "sd": vals.std(ddof=1), "rmse": np.sqrt(np.mean((vals - tr) ** 2)),
            "mean_se": g.se.mean() if os else np.nan,
            "mean_se_fold_centered": g.se_fold_centered.mean() if os else np.nan,
            "mean_eif_se_direct": g.se.mean() if os else np.nan,
            "mean_target_if_sd": g["target_if_sd"].mean() if os and "target_if_sd" in g else np.nan,
            "mean_source_if_sd": g["source_if_sd"].mean() if os and "source_if_sd" in g else np.nan,
            "mean_target_variance_component": g.target_variance_component.mean(),
            "mean_source_variance_component": g.source_variance_component.mean(),
            "mean_source_variance_fraction": g.source_variance_fraction.mean(),
            "source_selection_strength": float(g.source_selection_strength.iloc[0]),
            "mean_multiplier_se": g.multiplier_se.mean() if has_multiplier else np.nan,
            "coverage": np.mean((g.ci_low <= tr) & (tr <= g.ci_high)) if os else np.nan,
            "coverage_fold_centered": np.mean(
                (g.ci_fold_low <= tr) & (tr <= g.ci_fold_high)
            ) if os else np.nan,
            "coverage_multiplier": np.mean((g.multiplier_ci_low <= tr) & (tr <= g.multiplier_ci_high)) if has_multiplier else np.nan,
            "mean_ess": g.ess.mean(),
             "mean_ratio_p99": g.ratio_p99.mean(),
             "mean_ratio_max": g.ratio_max.mean(),
             "mean_ratio_cap_hit_rate": g.ratio_cap_hit_rate.mean(),
             "mean_balance_l2": g.balance_l2.mean(),
            "mean_correction": g.correction.mean(),
            "sd_correction_across_reps": g.correction.std(ddof=1),
            "mean_correction_sd": g.correction_sd.mean(),
            "mean_correction_p01": g.correction_p01.mean(),
            "mean_correction_p99": g.correction_p99.mean(),
            "mean_correction_abs_max": g.correction_abs_max.mean(),
            "mean_eif_p01": g.eif_p01.mean(),
            "mean_eif_p99": g.eif_p99.mean(),
            "mean_eif_abs_max": g.eif_abs_max.mean(),
            "min_eif_finite_rate": g.eif_finite_rate.min(),
            "mean_d10": g.mean_d10.mean(), "mean_d01": g.mean_d01.mean(),
            "mean_sd_d10": g.sd_d10.mean(), "mean_sd_d01": g.sd_d01.mean(),
             "mean_first10": g.mean_first10.mean(), "mean_second10": g.mean_second10.mean(),
             "mean_first01": g.mean_first01.mean(), "mean_second01": g.mean_second01.mean(),
             "mean_weighted_d10": g.weighted_d10.mean(), "mean_weighted_d01": g.weighted_d01.mean(),
             "mean_weighted_first10": g.weighted_first10.mean(), "mean_weighted_second10": g.weighted_second10.mean(),
             "mean_weighted_first01": g.weighted_first01.mean(), "mean_weighted_second01": g.weighted_second01.mean(),
             "max_one_step_identity_residual": g.one_step_identity_residual.abs().max(),
             "mean_surv_rmse": g.surv_rmse.mean(),
             "mean_surv_mae": g.surv_mae.mean(),
             "mean_surv_bias": g.surv_bias.mean(),
             "mean_surv_calibration_abs": g.surv_calibration_abs.mean(),
             "mean_surv_calibration_max": g.surv_calibration_max.mean(),
            "mean_best_epoch": g.mean_best_epoch.mean(),
            "mean_best_val_loss": g.mean_best_val_loss.mean(),
            "mean_full_refit_loss": g.mean_full_refit_loss.mean(),
            "mean_event_survival_floor_rate": g.event_survival_floor_rate.mean(),
            "mean_censor_survival_floor_rate": g.censor_survival_floor_rate.mean(),
            "mean_stack_weight_linear_cox": g.stack_weight_linear_cox.mean(),
            "mean_stack_weight_rsf": g.stack_weight_rsf.mean(),
            "mean_stack_weight_gbsa": g.stack_weight_gbsa.mean(),
            "mean_propensity_ridge": g.mean_propensity_ridge.mean(),
            "mean_propensity_cv_loss": g.mean_propensity_cv_loss.mean(),
        })
    summary = pd.DataFrame(out_rows)
    summary.to_csv(summary_path, index=False)
    rank = summary[summary.estimator.str.startswith("one_step_")].sort_values("rmse")
    lines = ["# Linear, MTLR/PMF, and outcome-drift DGP diagnostic simulation", "", f"nS={N_S}, nT={N_T}, reps={N_REP}, folds={N_FOLDS}, tau={TAU}, intervals={N_INTERVALS}, MTLR epochs={DISCRETE_HAZARD_EPOCHS}", "", "本阶段比较 Linear Cox 与不依赖比例风险假设的 MTLR/PMF 生存函数模型，并将 oracle ratio 与 Calibration Weighting ratio 分开报告。删失 nuisance 同样使用 MTLR/PMF。主 bias 以连续时间 CNB 真值计算，fixed-grid 真值仅用于诊断离散化误差；outcome_drift 情形用于验证：协变量校准不能修复条件生存函数漂移。"]
    for scenario in scenarios:
        g_learner, p_learner, ratio_learners = SCENARIO_NUISANCE[scenario]
        lines += ["", f"## DGP = {scenario}", f"CNB truth (120,000-target Monte Carlo): {truths[scenario]:.6f}", f"固定 nuisance：G={g_learner}, p={p_learner}, ratio=oracle/calibration", "", "|S learner|Ratio|Estimator|Bias|SD|RMSE|Mean SE|Multiplier SE|Coverage|Multiplier coverage|ESS|Balance L2|ratio p99|ratio max|cap hit|Corr mean|Corr SD|Corr p01|Corr p99|Corr abs max|EIF p01|EIF p99|EIF abs max|S floor|K floor|", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s_name in event_learners:
            for ratio in tuple(summary.loc[summary.scenario == scenario, "pi_learner"].unique()):
                for estimator in ("S_only", "plugin_T", f"one_step_{ratio}"):
                    rr = summary[(summary.scenario == scenario) & (summary.S_learner == s_name) & (summary.pi_learner == ratio) & (summary.estimator == estimator)].iloc[0]
                    lines.append(f"|{s_name}|{ratio}|{estimator}|{rr.bias:.4f}|{rr.sd:.4f}|{rr.rmse:.4f}|{rr.mean_se if pd.notna(rr.mean_se) else float('nan'):.4f}|{rr.mean_multiplier_se if pd.notna(rr.mean_multiplier_se) else float('nan'):.4f}|{rr.coverage if pd.notna(rr.coverage) else float('nan'):.3f}|{rr.coverage_multiplier if pd.notna(rr.coverage_multiplier) else float('nan'):.3f}|{rr.mean_ess:.1f}|{rr.mean_balance_l2:.4f}|{rr.mean_ratio_p99:.2f}|{rr.mean_ratio_max:.2f}|{rr.mean_ratio_cap_hit_rate:.3f}|{rr.mean_correction:.4f}|{rr.mean_correction_sd:.4f}|{rr.mean_correction_p01:.3f}|{rr.mean_correction_p99:.3f}|{rr.mean_correction_abs_max:.2f}|{rr.mean_eif_p01:.3f}|{rr.mean_eif_p99:.3f}|{rr.mean_eif_abs_max:.2f}|{rr.mean_event_survival_floor_rate:.3f}|{rr.mean_censor_survival_floor_rate:.3f}|")
        lines += ["", "EIF decomposition (one-step rows; weighted terms are transport-weighted):", "", "|S learner|mean D10|mean D01|weighted D10|weighted D01|weighted first10|weighted second10|weighted first01|weighted second01|SD D10|SD D01|max identity residual|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s_name in event_learners:
            ratio = str(summary.loc[summary.scenario == scenario, "pi_learner"].iloc[0])
            rr = summary[(summary.scenario == scenario) & (summary.S_learner == s_name) & (summary.pi_learner == ratio) & (summary.estimator == f"one_step_{ratio}")].iloc[0]
            lines.append(f"|{s_name}|{rr.mean_d10:.4f}|{rr.mean_d01:.4f}|{rr.mean_weighted_d10:.4f}|{rr.mean_weighted_d01:.4f}|{rr.mean_weighted_first10:.4f}|{rr.mean_weighted_second10:.4f}|{rr.mean_weighted_first01:.4f}|{rr.mean_weighted_second01:.4f}|{rr.mean_sd_d10:.4f}|{rr.mean_sd_d01:.4f}|{rr.max_one_step_identity_residual:.2e}|")
    report_status = "本次为正式500次Monte Carlo模拟。" if N_REP >= 500 else "本次为轻量筛选，正式结论需增加重复次数。"
    lines += ["", f"注：MTLR/PMF 为不含比例风险假设的归一化离散生存时间模型，先估计时间区间概率质量，再转换为 hazard 增量；EIF 仅在逆生存概率中使用下限稳定化。{report_status}"]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Replace the legacy report text with a compact UTF-8 report aligned to the
    # current Cox/RSF selector implementation.
    clean_lines = [
        "# Cox/RSF 选择器与迁移 EIF 的 DGP 诊断模拟",
        "",
        f"nS={N_S}, nT={N_T}, reps={N_REP}, folds={N_FOLDS}, tau={TAU}, intervals={N_INTERVALS}",
        "",
        "事件生存 nuisance 由仅使用训练折的内层交叉验证在 Linear Cox 与连续时间 RSF 之间选择。",
        "",
        "|DGP|Bias|SD|Mean EIF SE|RMSE|Coverage|ESS|Cox selection|RSF selection|Source variance fraction|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selected = summary[
        (summary.S_learner == "survival_selector")
        & (summary.estimator == "one_step_guided_calibration")
    ]
    for rr in selected.itertuples(index=False):
        clean_lines.append(
            f"|{rr.scenario}|{rr.bias:.4f}|{rr.sd:.4f}|{rr.mean_se:.4f}|{rr.rmse:.4f}|"
            f"{rr.coverage:.3f}|{rr.mean_ess:.1f}|{rr.mean_stack_weight_linear_cox:.3f}|"
            f"{rr.mean_stack_weight_rsf:.3f}|{rr.mean_source_variance_fraction:.3f}|"
        )
    clean_lines += [
        "",
        "注：倾向评分和 Cox/RSF 选择均只使用相应训练折；固定时间网格真值仅用于诊断数值积分误差。",
        (
            "本次为 500 次正式 Monte Carlo 模拟。"
            if N_REP >= 500
            else "本次为开发或独立验证模拟；正式结论需以锁定配置后的 500 次重复为准。"
        ),
    ]
    report_path.write_text("\n".join(clean_lines), encoding="utf-8")
    for scenario in scenarios:
        print(f"\nDGP={scenario}; truth={truths[scenario]:.6f}")
        print(summary[(summary.scenario == scenario) & summary.estimator.str.startswith("one_step_")][["S_learner", "G_learner", "p_learner", "pi_learner", "bias", "sd", "rmse", "coverage", "mean_ess"]].to_string(index=False))
    print(f"saved to {OUT}")


def main():
    parser = argparse.ArgumentParser(description="Run selected-nuisance simulations under linear, nonlinear, and outcome-drift DGPs.")
    parser.add_argument("--label", default="light20")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--n-s", type=int, default=None)
    parser.add_argument("--n-t", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--intervals", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument(
        "--fold-seed-offset",
        type=int,
        default=0,
        help="Positive repeat index for independent outer cross-fitting splits.",
    )
    parser.add_argument("--save-contributions", action="store_true")
    parser.add_argument(
        "--control-variate", action="store_true",
        help="Enable the diagnostic calibration control-variate estimator.",
    )
    parser.add_argument("--multiplier-reps", type=int, default=0)
    parser.add_argument(
        "--g-learner",
        choices=("cox", "flexible", "rsf", "mtlr", "oracle"),
        default=None,
    )
    parser.add_argument("--p-learner", choices=("logistic", "flexible", "oracle"), default=None)
    parser.add_argument("--ratio-learners", default=None,
                        help="Comma-separated ratio learners: calibration,guided_calibration,oracle,identity")
    parser.add_argument("--source-mode", choices=("shifted", "target"), default="shifted")
    parser.add_argument("--overlap-strength", type=float, default=None)
    parser.add_argument("--randomized-treatment", action="store_true")
    parser.add_argument("--weight-cap", type=float, default=None)
    parser.add_argument("--calibration-basis", choices=("linear", "quadratic", "spline"), default=None)
    parser.add_argument("--calibration-ridge", type=float, default=None)
    parser.add_argument("--calibration-ridge-rate", choices=("fixed", "sqrt_n"), default=None)
    parser.add_argument("--rsf-jobs", type=int, default=None)
    parser.add_argument("--rsf-trees", type=int, default=None)
    parser.add_argument("--rsf-min-leaf", type=int, default=None)
    parser.add_argument("--rsf-max-features", default=None)
    parser.add_argument("--gbsa-trees", type=int, default=None)
    parser.add_argument("--gbsa-learning-rate", type=float, default=None)
    parser.add_argument("--gbsa-max-depth", type=int, default=None)
    parser.add_argument("--gbsa-min-leaf", type=int, default=None)
    parser.add_argument("--stack-inner-folds", type=int, default=None)
    parser.add_argument(
        "--rsf-prediction-mode",
        choices=("survival", "cumhaz"),
        default=None,
        help="Use direct RSF survival predictions or the legacy cumulative-hazard interface.",
    )
    parser.add_argument("--s-learners", default="linear_cox,rsf",
                        help="Comma-separated continuous-time event learners: linear_cox,sieve_cox,rsf,gbsa,survival_stack,survival_selector,oracle")
    parser.add_argument("--scenarios", default="linear,nonlinear,outcome_drift",
                        help="Comma-separated DGP scenarios")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    global N_REP, N_S, N_T, N_FOLDS, N_INTERVALS, DISCRETE_HAZARD_EPOCHS
    global CENSOR_LEARNER_OVERRIDE, P_LEARNER_OVERRIDE, RATIO_LEARNERS_OVERRIDE
    global FOLD_SEED_OFFSET
    global SOURCE_MODE, RANDOMIZED_TREATMENT, WEIGHT_CAP, CALIBRATION_BASIS_KIND
    global SOURCE_SELECTION_STRENGTH
    global CALIBRATION_RIDGE, CALIBRATION_RIDGE_RATE
    global RSF_N_JOBS, RSF_TREES, RSF_MIN_LEAF, RSF_MAX_FEATURES
    global GBSA_TREES, GBSA_LEARNING_RATE, GBSA_MAX_DEPTH, GBSA_MIN_LEAF
    global STACK_INNER_FOLDS
    global RSF_PREDICTION_MODE
    global ENABLE_CONTROL_VARIATE
    for name, value in (("N_REP", args.reps), ("N_S", args.n_s), ("N_T", args.n_t), ("N_FOLDS", args.folds), ("N_INTERVALS", args.intervals), ("DISCRETE_HAZARD_EPOCHS", args.epochs)):
        if value is not None:
            globals()[name] = value
    CENSOR_LEARNER_OVERRIDE = args.g_learner
    P_LEARNER_OVERRIDE = args.p_learner
    FOLD_SEED_OFFSET = max(0, int(args.fold_seed_offset))
    if args.ratio_learners is not None:
        RATIO_LEARNERS_OVERRIDE = tuple(
            x.strip() for x in args.ratio_learners.split(",") if x.strip()
        )
    SOURCE_MODE = args.source_mode
    if args.overlap_strength is not None:
        SOURCE_SELECTION_STRENGTH = float(args.overlap_strength)
    RANDOMIZED_TREATMENT = bool(args.randomized_treatment)
    if args.weight_cap is not None:
        WEIGHT_CAP = float(args.weight_cap)
    if args.calibration_basis is not None:
        CALIBRATION_BASIS_KIND = args.calibration_basis
    if args.calibration_ridge is not None:
        CALIBRATION_RIDGE = float(args.calibration_ridge)
    if args.calibration_ridge_rate is not None:
        CALIBRATION_RIDGE_RATE = args.calibration_ridge_rate
    if args.rsf_jobs is not None:
        RSF_N_JOBS = max(1, int(args.rsf_jobs))
    if args.rsf_trees is not None:
        RSF_TREES = max(50, int(args.rsf_trees))
    if args.rsf_min_leaf is not None:
        RSF_MIN_LEAF = max(2, int(args.rsf_min_leaf))
    if args.rsf_max_features is not None:
        value = args.rsf_max_features
        try:
            value = float(value)
        except ValueError:
            pass
        RSF_MAX_FEATURES = value
    if args.gbsa_trees is not None:
        GBSA_TREES = max(25, int(args.gbsa_trees))
    if args.gbsa_learning_rate is not None:
        GBSA_LEARNING_RATE = float(args.gbsa_learning_rate)
    if args.gbsa_max_depth is not None:
        GBSA_MAX_DEPTH = max(1, int(args.gbsa_max_depth))
    if args.gbsa_min_leaf is not None:
        GBSA_MIN_LEAF = max(2, int(args.gbsa_min_leaf))
    if args.stack_inner_folds is not None:
        STACK_INNER_FOLDS = max(2, int(args.stack_inner_folds))
    if args.rsf_prediction_mode is not None:
        RSF_PREDICTION_MODE = args.rsf_prediction_mode
    ENABLE_CONTROL_VARIATE = bool(args.control_variate)
    event_learners = tuple(x.strip() for x in args.s_learners.split(",") if x.strip())
    scenarios = tuple(x.strip() for x in args.scenarios.split(",") if x.strip())
    run(label=args.label, rep_start=args.rep_start,
        save_contributions=args.save_contributions,
        multiplier_reps=args.multiplier_reps,
        output_dir=args.output_dir,
        event_learners=event_learners,
        scenarios=scenarios)


if __name__ == "__main__":
    main()
