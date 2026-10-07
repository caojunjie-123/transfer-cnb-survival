import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

try:
    from coxkan import CoxKAN as OfficialCoxKAN
except Exception:
    OfficialCoxKAN = None


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
BASE_DIR = ROOT / "output" / "final_cnb_transfer_standard_onestep_calibration"
sys.path.insert(0, str(BASE_DIR))

import final_cnb_transfer_standard_onestep_calibration as base


OUT_DIR = THIS_DIR
RAW_PATH = OUT_DIR / "dl_nuisance_onestep_raw.csv"
SUMMARY_PATH = OUT_DIR / "dl_nuisance_onestep_summary.csv"
DIAGNOSTICS_PATH = OUT_DIR / "dl_nuisance_onestep_diagnostics.csv"
NB_PATH = OUT_DIR / "dl_nuisance_onestep_comparison.ipynb"

SEED = 20260608
N_S = 1000
N_T = 200
N_REP = 100
N_FOLDS = 5
TAU = base.TAU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_SIZE = 3
CONSERVATIVE_SE_MULTIPLIER = 1.20
COXKAN_GRID_UPDATE_EPOCHS = 50
COXKAN_GRID_UPDATE_FREQ = 5
HYBRID_ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.00)
GATED_ALPHA_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
GATED_MIN_GAIN = 0.005
GATED_SE_MULTIPLIER = 0.50
RESIDUAL_ORTHOGONAL_RIDGE = 1e-3
DISCRETE_HAZARD_INTERVALS = 24
DISCRETE_HAZARD_MIN = 1e-5
DISCRETE_HAZARD_MAX = 0.95
DISCRETE_HAZARD_USE_OFFSET_CALIBRATION = False
DISCRETE_HAZARD_OFFSET_RIDGE = 1e-2
DISCRETE_HAZARD_OFFSET_CLIP = 5.0
WEIBULL_HAZARD_INTERVALS = 48
WEIBULL_MIN_TIME = 1e-4
WEIBULL_SHAPE_BOUNDS = (0.20, 5.00)
WEIBULL_SCALE_BOUNDS = (0.50, 80.00)
PMF_SURVIVAL_INTERVALS = 48
PMF_SURVIVAL_MIN_PROB = 1e-6
PMF_SURVIVAL_GRID = "event_quantile"
PMF_SURVIVAL_LOSS_WEIGHT = "none"
PMF_OUTCOME_WEIGHT = "none"
PMF_WEIGHT_CAP = 5.0
# Full DeepHit combines the discrete-time likelihood with a pairwise
# ranking loss.  The PMF output retains one administrative tail bin so the
# existing censored-survival and CNB prediction interfaces remain unchanged.
DEEPHIT_RANKING_WEIGHT = 0.20
DEEPHIT_RANKING_SIGMA = 0.10
SHARED_HAZARD_EVENT_WEIGHT = 1.0
SHARED_HAZARD_EVENT_WEIGHT_ARM0 = None
SHARED_HAZARD_EVENT_WEIGHT_ARM1 = None
SHARED_HAZARD_EPOCHS = 250
EIF_WEIGHT_CAP = base.DEFAULT_WEIGHT_CAP
EIF_CALIBRATION_RIDGE = base.DEFAULT_CALIBRATION_RIDGE
CI_METHOD = "eif"
CI_METHOD_CHOICES = ("eif", "fold", "max")
NORMAL_CRITICAL_95 = 1.959963984540054
DGP_SCENARIO = "standard"
DGP_TARGET_SHIFT_SCALE = 1.0
DGP_SCENARIO_CHOICES = ("linear_ph", "nonlinear_additive_ph", "standard", "nonlinear_moderate_shift", "nonlinear_model_free", "nonlinear_harder", "nonph_time_varying")
RCS_KNOTS = np.array([-2.5, -0.75, 0.0, 0.75, 2.5], dtype=float)
CENSOR_LEARNER = "cox"
CENSOR_LEARNER_CHOICES = ("cox", "neural")
RATIO_LEARNER = "calibration"
RATIO_LEARNER_CHOICES = ("calibration", "logistic", "neural")
PMF_GRID_CHOICES = ("uniform", "event_quantile")
PMF_LOSS_WEIGHT_CHOICES = ("none", "event_balanced")
PMF_OUTCOME_WEIGHT_CHOICES = ("none", "density_ratio", "calibration")


def label_path(path, label):
    if not label:
        return path
    return path.with_name(f"{path.stem}_{label}{path.suffix}")


def safe_to_csv(df, path):
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.tmp_{os.getpid()}_{time.time_ns()}{path.suffix}")
    csv_text = df.to_csv(index=False)
    try:
        tmp.write_text(csv_text, encoding="utf-8")
        tmp.replace(path)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_fallback_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        fallback.write_text(csv_text, encoding="utf-8")
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        print(f"Permission denied while writing {path}; wrote {fallback} instead.", flush=True)
        return fallback


def current_target_mu():
    return base.MU_S + DGP_TARGET_SHIFT_SCALE * (base.MU_T - base.MU_S)


def sample_x_dgp(n, mu, rng):
    return base.sample_x(n, mu, rng)


def eta_dgp(X, a):
    if DGP_SCENARIO == "linear_ph":
        X = np.asarray(X)
        if a == 0:
            return 0.65 * X[:, 0] - 0.55 * X[:, 1]
        return 0.10 * X[:, 0] - 0.15 * X[:, 1] + base.TREATMENT_LOG_RISK_SHIFT
    if DGP_SCENARIO == "nonlinear_additive_ph":
        X = np.asarray(X)
        common = (
            0.35 * (X[:, 0] ** 2 - 1.0)
            + 0.50 * np.sin(X[:, 1])
            - 0.30 * X[:, 2]
            + 0.20 * X[:, 3]
        )
        return common + (base.TREATMENT_LOG_RISK_SHIFT if a == 1 else 0.0)
    if DGP_SCENARIO == "standard":
        return base.eta_true(X, a)
    X = np.asarray(X)
    base_eta = base.eta_true(X, a)
    if DGP_SCENARIO == "nonlinear_moderate_shift":
        common = (
            0.25 * np.sin(1.1 * X[:, 0])
            + 0.18 * (X[:, 1] ** 2 - 1.0)
            - 0.18 * X[:, 2] * X[:, 3]
        )
        if a == 0:
            arm_term = 0.18 * np.cos(X[:, 0] * X[:, 4]) - 0.10 * X[:, 3] ** 2
        else:
            arm_term = -0.22 * np.sin(X[:, 2] + 0.5 * X[:, 4]) + 0.25 * X[:, 0] * X[:, 1]
        return base_eta + common + arm_term
    if DGP_SCENARIO == "nonlinear_model_free":
        base_eta = 0.55 * base_eta
        common = (
            0.38 * np.sin(1.2 * X[:, 0])
            + 0.28 * (X[:, 1] ** 2 - 1.0)
            - 0.30 * X[:, 2] * X[:, 3]
            + 0.25 * np.tanh(X[:, 3] * X[:, 4])
        )
        if a == 0:
            arm_term = 0.36 * np.cos(X[:, 0] * X[:, 4]) - 0.20 * X[:, 3] ** 2
        else:
            arm_term = -0.42 * np.sin(X[:, 2] + 0.5 * X[:, 4]) + 0.42 * X[:, 0] * X[:, 1]
        return base_eta + common + arm_term
    if DGP_SCENARIO != "nonlinear_harder":
        raise ValueError(f"Unknown DGP scenario: {DGP_SCENARIO}")
    common = (
        0.35 * np.sin(1.2 * X[:, 0])
        + 0.28 * (X[:, 1] ** 2 - 1.0)
        - 0.30 * X[:, 2] * X[:, 3]
    )
    if a == 0:
        arm_term = 0.30 * np.cos(X[:, 0] * X[:, 4]) - 0.22 * X[:, 3] ** 2
    else:
        arm_term = -0.36 * np.sin(X[:, 2] + X[:, 4]) + 0.32 * X[:, 0] * X[:, 1]
    return base_eta + common + arm_term


NONPH_CHANGE_TIME = 6.0


def nonph_eta_segments(X, a):
    """Return early/late log-hazard scores for a time-varying-effect DGP."""
    X = np.asarray(X)
    common = (
        0.40 * X[:, 0]
        - 0.30 * X[:, 1]
        + 0.25 * np.sin(X[:, 2])
        - 0.20 * X[:, 3] * X[:, 4]
    )
    if a == 0:
        return common, common
    early_effect = -0.45 + 0.15 * X[:, 0]
    late_effect = 0.35 + 0.15 * X[:, 0]
    return common + early_effect, common + late_effect


def sample_nonph_event_time(scale, shape, eta_early, eta_late, rng, change_time=NONPH_CHANGE_TIME):
    """Inverse-transform sample from a two-segment Weibull-baseline hazard."""
    eta_early = np.asarray(eta_early, dtype=float)
    eta_late = np.asarray(eta_late, dtype=float)
    e = -np.log(np.clip(rng.uniform(size=eta_early.shape[0]), 1e-12, 1.0))
    h_change = (change_time / scale) ** shape
    early_mass = np.exp(np.clip(eta_early, -8, 8)) * h_change
    before = e <= early_mass
    out = np.empty_like(e)
    out[before] = scale * (
        e[before] / np.exp(np.clip(eta_early[before], -8, 8))
    ) ** (1.0 / shape)
    late_h = h_change + (
        e[~before] - early_mass[~before]
    ) / np.exp(np.clip(eta_late[~before], -8, 8))
    out[~before] = scale * np.maximum(late_h, 1e-12) ** (1.0 / shape)
    return out


def nonph_cumulative_hazard(t, X, a, scale, shape, change_time=NONPH_CHANGE_TIME):
    """Cumulative hazard under the two-segment time-varying-effect DGP."""
    eta_early, eta_late = nonph_eta_segments(X, a)
    t_arr = np.asarray(t, dtype=float)
    h_base = (np.maximum(t_arr, 0.0) / scale) ** shape
    h_change = (change_time / scale) ** shape
    early_mass = np.exp(np.clip(eta_early, -8, 8)) * h_change
    late_mass = np.maximum(h_base - h_change, 0.0)[..., None] * np.exp(np.clip(eta_late, -8, 8))[None, :]
    return np.where(
        t_arr[..., None] <= change_time,
        np.exp(np.clip(eta_early, -8, 8)) * h_base[..., None],
        early_mass[None, :] + late_mass,
    )


def nonph_hazard(t, X, a, scale, shape, change_time=NONPH_CHANGE_TIME):
    """Instantaneous hazard under the two-segment time-varying-effect DGP."""
    eta_early, eta_late = nonph_eta_segments(X, a)
    t_arr = np.asarray(t, dtype=float)
    base_hazard = (shape / scale) * np.maximum(t_arr, 1e-8) ** (shape - 1.0) / (scale ** (shape - 1.0))
    eta = np.where(t_arr[..., None] <= change_time, eta_early[None, :], eta_late[None, :])
    return base_hazard[..., None] * np.exp(np.clip(eta, -8, 8))


def generate_source_data_dgp(n, rng):
    if DGP_SCENARIO == "standard":
        return base.generate_source_data(n, rng)
    X = sample_x_dgp(n, base.MU_S, rng)
    pi = base.propensity_true(X)
    A = rng.binomial(1, pi, size=n)
    if DGP_SCENARIO == "nonph_time_varying":
        eta00, eta01 = nonph_eta_segments(X, 0)
        eta10, eta11 = nonph_eta_segments(X, 1)
        T0 = sample_nonph_event_time(16.0, 1.35, eta00, eta01, rng)
        T1 = sample_nonph_event_time(18.0, 1.25, eta10, eta11, rng)
    else:
        T0 = base.sample_weibull_cox(16.0, 1.35, eta_dgp(X, 0), rng)
        T1 = base.sample_weibull_cox(18.0, 1.25, eta_dgp(X, 1), rng)
    T = np.where(A == 1, T1, T0)
    if DGP_SCENARIO in {"linear_ph", "nonlinear_additive_ph", "nonlinear_moderate_shift", "nonlinear_model_free"}:
        eta_C = 0.20 * A + 0.25 * X[:, 0] - 0.12 * X[:, 1] + 0.10 * X[:, 3]
    else:
        eta_C = 0.25 * A + 0.30 * X[:, 0] - 0.18 * X[:, 1] ** 2 + 0.12 * X[:, 3]
    C = base.sample_weibull_cox(22.0, 1.10, eta_C, rng)
    U = np.minimum(T, C)
    Delta = (T <= C).astype(int)
    return X, A, U, Delta


def true_cnb_g_dgp(X, tau=TAU):
    if DGP_SCENARIO == "standard":
        return base.true_cnb_g_cox(X, tau=tau)
    grid = np.linspace(0.0, tau, 301)
    dt = np.diff(grid)
    mid = (grid[:-1] + grid[1:]) / 2.0
    if DGP_SCENARIO == "nonph_time_varying":
        H0 = nonph_cumulative_hazard(mid, X, 0, 16.0, 1.35)
        H1 = nonph_cumulative_hazard(mid, X, 1, 18.0, 1.25)
        h0 = nonph_hazard(mid, X, 0, 16.0, 1.35)
        h1 = nonph_hazard(mid, X, 1, 18.0, 1.25)
        return np.sum(np.exp(-H0 - H1) * (h0 - h1) * dt[:, None], axis=0)
    e0 = np.exp(np.clip(eta_dgp(X, 0), -8, 8))
    e1 = np.exp(np.clip(eta_dgp(X, 1), -8, 8))
    scale0, shape0 = 16.0, 1.35
    scale1, shape1 = 18.0, 1.25
    H0 = (mid[:, None] / scale0) ** shape0 * e0[None, :]
    H1 = (mid[:, None] / scale1) ** shape1 * e1[None, :]
    h0 = (shape0 / scale0) * (mid[:, None] / scale0) ** (shape0 - 1.0) * e0[None, :]
    h1 = (shape1 / scale1) * (mid[:, None] / scale1) ** (shape1 - 1.0) * e1[None, :]
    return np.sum(np.exp(-H0 - H1) * (h0 - h1) * dt[:, None], axis=0)


def set_torch_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def phi_spline(X, knots=RCS_KNOTS):
    """Restricted cubic-spline Cox features with fixed simulation knots."""
    X = np.asarray(X, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 1 or len(knots) < 4 or not np.all(np.diff(knots) > 0):
        raise ValueError("RCS knots must be a strictly increasing 1D array with at least four values")
    boundary_left, boundary_right = knots[0], knots[-1]
    last_internal = knots[-2]
    denom = boundary_right - last_internal
    basis = []
    for knot in knots[1:-2]:
        term = np.maximum(X - knot, 0.0) ** 3
        term -= ((boundary_right - knot) / denom) * np.maximum(X - last_internal, 0.0) ** 3
        term += ((last_internal - knot) / denom) * np.maximum(X - boundary_right, 0.0) ** 3
        basis.append(term)
    return np.column_stack([X, *basis])


class DeepSurvNet(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DiscreteHazardNet(nn.Module):
    def __init__(self, p, n_intervals):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_intervals),
        )

    def forward(self, x):
        return self.net(x)


class SharedArmDiscreteHazardNet(nn.Module):
    def __init__(self, p, n_intervals):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(p, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head0 = nn.Linear(16, n_intervals)
        self.head1 = nn.Linear(16, n_intervals)

    def forward(self, x):
        z = self.trunk(x)
        return torch.stack([self.head0(z), self.head1(z)], dim=1)


class WeibullPHNet(nn.Module):
    def __init__(self, p, init_shape=1.20, init_scale=16.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.log_shape = nn.Parameter(torch.tensor(float(np.log(init_shape)), dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(init_scale)), dtype=torch.float32))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def shape_scale(self):
        shape = torch.exp(torch.clamp(
            self.log_shape,
            float(np.log(WEIBULL_SHAPE_BOUNDS[0])),
            float(np.log(WEIBULL_SHAPE_BOUNDS[1])),
        ))
        scale = torch.exp(torch.clamp(
            self.log_scale,
            float(np.log(WEIBULL_SCALE_BOUNDS[0])),
            float(np.log(WEIBULL_SCALE_BOUNDS[1])),
        ))
        return shape, scale


class DeepHitSurvivalNet(nn.Module):
    def __init__(self, p, n_intervals):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_intervals + 1),
        )

    def forward(self, x):
        return self.net(x)


# Backward-compatible name used by older simulation code.
PMFSurvivalNet = DeepHitSurvivalNet


class DensityRatioNet(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_neural_cox_model(learner, p, seed):
    if learner == "deepsurv":
        return DeepSurvNet(p).float()
    if learner == "coxkan":
        if OfficialCoxKAN is None:
            raise RuntimeError("Official CoxKAN is not importable in this Python environment.")
        hidden = 8 if p <= 10 else 12
        return OfficialCoxKAN(
            width=[p, hidden, 1],
            grid=3,
            k=3,
            base_fun="silu",
            symbolic_enabled=False,
            grid_range=[-3.0, 3.0],
            seed=int(seed),
            device=str(DEVICE),
        )
    raise ValueError(f"Unknown neural Cox learner: {learner}")


def set_module_mode(model, training):
    nn.Module.train(model, bool(training))


def model_eta(model, x):
    return model(x).reshape(-1)


def maybe_update_coxkan_grid(model, x):
    if OfficialCoxKAN is not None and isinstance(model, OfficialCoxKAN):
        with torch.no_grad():
            model.update_grid_from_samples(x)


def cox_ph_loss(eta, time, event):
    order = torch.argsort(time, descending=True)
    eta_ord = torch.clamp(eta[order], -20.0, 20.0)
    event_ord = event[order]
    log_risk = torch.logcumsumexp(eta_ord, dim=0)
    n_events = torch.clamp(event_ord.sum(), min=1.0)
    return ((log_risk - eta_ord) * event_ord).sum() / n_events


def validation_split_indices(Delta, seed):
    rng = np.random.default_rng(seed)
    event_idx = np.flatnonzero(Delta == 1)
    non_event_idx = np.flatnonzero(Delta == 0)
    rng.shuffle(event_idx)
    rng.shuffle(non_event_idx)
    val_idx = np.r_[
        event_idx[: max(1, int(0.20 * len(event_idx)))],
        non_event_idx[: max(1, int(0.20 * len(non_event_idx)))],
    ]
    train_mask = np.ones(len(Delta), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.flatnonzero(train_mask)
    if np.sum(Delta[train_idx]) < 5 or np.sum(Delta[val_idx]) < 2:
        train_idx = np.arange(len(Delta))
        val_idx = np.arange(len(Delta))
    return train_idx, val_idx


def cox_ph_loss_numpy(eta, U, Delta):
    eta_t = torch.as_tensor(eta, dtype=torch.float32, device=DEVICE)
    u_t = torch.as_tensor(U, dtype=torch.float32, device=DEVICE)
    d_t = torch.as_tensor(Delta.astype(float), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return float(cox_ph_loss(eta_t, u_t, d_t).cpu())


def cox_event_loss_contributions(eta, U, Delta):
    eta = np.clip(np.asarray(eta, dtype=float), -20.0, 20.0)
    U = np.asarray(U)
    Delta = np.asarray(Delta)
    order = np.argsort(-U)
    eta_ord = eta[order]
    event_ord = Delta[order].astype(bool)
    log_risk = np.logaddexp.accumulate(eta_ord)
    contrib = log_risk[event_ord] - eta_ord[event_ord]
    if len(contrib) == 0:
        contrib = np.array([0.0])
    return contrib


def cox_event_loss_mean_se(eta, U, Delta):
    contrib = cox_event_loss_contributions(eta, U, Delta)
    mean = float(np.mean(contrib))
    se = float(np.std(contrib, ddof=1) / np.sqrt(len(contrib))) if len(contrib) > 1 else 0.0
    return mean, se


def make_discrete_time_grid(tau=TAU, n_intervals=DISCRETE_HAZARD_INTERVALS):
    return np.linspace(0.0, tau, int(n_intervals) + 1)


def make_event_quantile_time_grid(U, Delta, tau=TAU, n_intervals=PMF_SURVIVAL_INTERVALS):
    U = np.asarray(U, dtype=float)
    Delta = np.asarray(Delta, dtype=int)
    n_intervals = int(n_intervals)
    event_times = U[(Delta == 1) & (U > 0.0) & (U <= tau)]
    if len(event_times) < 2:
        return make_discrete_time_grid(tau=tau, n_intervals=n_intervals)
    probs = np.linspace(0.0, 1.0, n_intervals + 1)
    interior = np.quantile(event_times, probs[1:-1]) if n_intervals > 1 else np.array([], dtype=float)
    grid = np.unique(np.r_[0.0, interior, float(tau)])
    grid = grid[(grid >= 0.0) & (grid <= tau)]
    if len(grid) < 3:
        return make_discrete_time_grid(tau=tau, n_intervals=n_intervals)
    grid[0] = 0.0
    grid[-1] = float(tau)
    return grid.astype(float)


def make_pmf_time_grid(U, Delta, tau=TAU, n_intervals=PMF_SURVIVAL_INTERVALS, grid=PMF_SURVIVAL_GRID):
    if grid == "uniform":
        return make_discrete_time_grid(tau=tau, n_intervals=n_intervals)
    if grid == "event_quantile":
        return make_event_quantile_time_grid(U, Delta, tau=tau, n_intervals=n_intervals)
    raise ValueError(f"Unknown PMF time grid: {grid}")


def make_discrete_hazard_arrays(U, Delta, time_grid):
    U = np.asarray(U, dtype=float)
    Delta = np.asarray(Delta, dtype=int)
    right = np.asarray(time_grid[1:], dtype=float)
    tau = float(time_grid[-1])
    n, j = len(U), len(right)
    target = np.zeros((n, j), dtype=np.float32)
    mask = right[None, :] <= np.minimum(U[:, None], tau) + 1e-12
    event_rows = np.flatnonzero((Delta == 1) & (U <= tau))
    if len(event_rows):
        event_bins = np.searchsorted(right, U[event_rows], side="left")
        event_bins = np.clip(event_bins, 0, j - 1)
        target[event_rows, event_bins] = 1.0
        mask[event_rows, event_bins] = True
    return target, mask.astype(np.float32)


def discrete_hazard_loss(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denom = torch.clamp(mask.sum(), min=1.0)
    return (loss * mask).sum() / denom


def weighted_masked_mean(values, weight_mask):
    denom = torch.clamp(weight_mask.sum(), min=1.0)
    return (values * weight_mask).sum() / denom


def shared_arm_discrete_hazard_loss(
    logits_by_arm,
    arm,
    target,
    mask,
    sample_weight=None,
    loss_weight=PMF_SURVIVAL_LOSS_WEIGHT,
    event_weight=SHARED_HAZARD_EVENT_WEIGHT,
    event_weight_arm0=SHARED_HAZARD_EVENT_WEIGHT_ARM0,
    event_weight_arm1=SHARED_HAZARD_EVENT_WEIGHT_ARM1,
):
    row = torch.arange(logits_by_arm.shape[0], device=logits_by_arm.device)
    selected_logits = logits_by_arm[row, arm.long(), :]
    loss = F.binary_cross_entropy_with_logits(selected_logits, target, reduction="none")
    weighted_mask = mask
    if sample_weight is not None:
        weighted_mask = weighted_mask * sample_weight[:, None]
    if event_weight_arm0 is not None or event_weight_arm1 is not None:
        ew0 = float(event_weight if event_weight_arm0 is None else event_weight_arm0)
        ew1 = float(event_weight if event_weight_arm1 is None else event_weight_arm1)
        selected_event_weight = torch.where(
            arm.long() == 0,
            torch.full_like(arm.float(), ew0),
            torch.full_like(arm.float(), ew1),
        )
    else:
        selected_event_weight = torch.full_like(arm.float(), float(event_weight))
    if loss_weight == "none" and torch.any(selected_event_weight != 1.0):
        class_weight = 1.0 + (selected_event_weight[:, None] - 1.0) * (target > 0.5).float()
        weighted_mask = weighted_mask * class_weight
    if loss_weight == "event_balanced":
        event_mask = weighted_mask * (target > 0.5).float()
        non_event_mask = weighted_mask * (target <= 0.5).float()
        if event_mask.sum() > 0 and non_event_mask.sum() > 0:
            return 0.5 * weighted_masked_mean(loss, event_mask) + 0.5 * weighted_masked_mean(loss, non_event_mask)
    elif loss_weight != "none":
        raise ValueError(f"Unknown shared discrete hazard loss weighting: {loss_weight}")
    denom = torch.clamp(weighted_mask.sum(), min=1.0)
    return (loss * weighted_mask).sum() / denom


def fit_interval_logit_offsets(logits, target, mask, ridge=DISCRETE_HAZARD_OFFSET_RIDGE):
    logits = np.asarray(logits, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(mask, dtype=float) > 0.5
    offsets = np.zeros(logits.shape[1], dtype=float)
    for j in range(logits.shape[1]):
        idx = mask[:, j]
        if np.sum(idx) < 5:
            continue
        z = logits[idx, j]
        y = target[idx, j]
        offset = 0.0
        for _ in range(25):
            p = 1.0 / (1.0 + np.exp(-np.clip(z + offset, -40.0, 40.0)))
            grad = float(np.sum(p - y) + ridge * offset)
            hess = float(np.sum(p * (1.0 - p)) + ridge)
            if hess <= 1e-10:
                break
            step = grad / hess
            offset -= step
            if abs(step) < 1e-6:
                break
        offsets[j] = float(np.clip(offset, -DISCRETE_HAZARD_OFFSET_CLIP, DISCRETE_HAZARD_OFFSET_CLIP))
    return offsets


def fit_discrete_hazard(X, U, Delta, seed, n_intervals=DISCRETE_HAZARD_INTERVALS, epochs=250, lr=1e-3, weight_decay=1e-4, patience=25):
    set_torch_seed(seed)
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    Xs = (X - x_mean) / x_scale
    time_grid = make_discrete_time_grid(TAU, n_intervals)
    target_np, mask_np = make_discrete_hazard_arrays(U, Delta, time_grid)
    train_idx, val_idx = validation_split_indices(Delta, seed)

    x_train = torch.as_tensor(Xs[train_idx], dtype=torch.float32, device=DEVICE)
    y_train = torch.as_tensor(target_np[train_idx], dtype=torch.float32, device=DEVICE)
    m_train = torch.as_tensor(mask_np[train_idx], dtype=torch.float32, device=DEVICE)
    x_val = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = torch.as_tensor(target_np[val_idx], dtype=torch.float32, device=DEVICE)
    m_val = torch.as_tensor(mask_np[val_idx], dtype=torch.float32, device=DEVICE)

    model = DiscreteHazardNet(X.shape[1], n_intervals).float().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = np.inf
    best_train = np.inf
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(epochs):
        set_module_mode(model, True)
        opt.zero_grad()
        loss = discrete_hazard_loss(model(x_train), y_train, m_train)
        loss.backward()
        opt.step()
        set_module_mode(model, False)
        with torch.no_grad():
            train_loss = float(discrete_hazard_loss(model(x_train), y_train, m_train).cpu())
            val_loss = float(discrete_hazard_loss(model(x_val), y_val, m_val).cpu())
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    full_epochs = max(1, int(best_epoch))
    set_torch_seed(seed + 7919)
    x_all = torch.as_tensor(Xs, dtype=torch.float32, device=DEVICE)
    y_all = torch.as_tensor(target_np, dtype=torch.float32, device=DEVICE)
    m_all = torch.as_tensor(mask_np, dtype=torch.float32, device=DEVICE)
    final_model = DiscreteHazardNet(X.shape[1], n_intervals).float().to(DEVICE)
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(full_epochs):
        set_module_mode(final_model, True)
        final_opt.zero_grad()
        full_loss = discrete_hazard_loss(final_model(x_all), y_all, m_all)
        full_loss.backward()
        final_opt.step()
    set_module_mode(final_model, False)
    with torch.no_grad():
        logits_all = final_model(x_all)
        full_refit_loss = float(discrete_hazard_loss(logits_all, y_all, m_all).cpu())
        logits_all_np = logits_all.detach().cpu().numpy()
    if DISCRETE_HAZARD_USE_OFFSET_CALIBRATION:
        logit_offsets = fit_interval_logit_offsets(logits_all_np, target_np, mask_np)
    else:
        logit_offsets = np.zeros(n_intervals, dtype=float)
    hazards_all = 1.0 / (1.0 + np.exp(-np.clip(logits_all_np + logit_offsets[None, :], -40.0, 40.0)))
    final_model = final_model.cpu()
    set_module_mode(final_model, False)
    return {
        "discrete_hazard": True,
        "model": final_model,
        "center": x_mean,
        "scale": x_scale,
        "time_grid": time_grid,
        "logit_offsets": logit_offsets,
        "best_epoch": best_epoch,
        "best_train_loss": best_train,
        "best_val_loss": best_val,
        "full_refit_loss": full_refit_loss,
        "discrete_hazard_mean": float(np.mean(hazards_all)),
        "discrete_hazard_max": float(np.max(hazards_all)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_events": int(np.sum(Delta)),
    }


def fit_shared_arm_discrete_hazard(
    X,
    A,
    U,
    Delta,
    seed,
    n_intervals=PMF_SURVIVAL_INTERVALS,
    grid=PMF_SURVIVAL_GRID,
    loss_weight=PMF_SURVIVAL_LOSS_WEIGHT,
    event_weight=SHARED_HAZARD_EVENT_WEIGHT,
    event_weight_arm0=SHARED_HAZARD_EVENT_WEIGHT_ARM0,
    event_weight_arm1=SHARED_HAZARD_EVENT_WEIGHT_ARM1,
    sample_weight=None,
    outcome_weight_label=PMF_OUTCOME_WEIGHT,
    weight_cap=PMF_WEIGHT_CAP,
    epochs=SHARED_HAZARD_EPOCHS,
    lr=1e-3,
    weight_decay=1e-4,
    patience=25,
):
    set_torch_seed(seed)
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    Xs = (X - x_mean) / x_scale
    time_grid = make_pmf_time_grid(U, Delta, tau=TAU, n_intervals=n_intervals, grid=grid)
    n_actual_intervals = len(time_grid) - 1
    target_np, mask_np = make_discrete_hazard_arrays(U, Delta, time_grid)
    train_idx, val_idx = validation_split_indices(Delta, seed)
    if sample_weight is None:
        sample_weight = np.ones(len(U), dtype=float)
    sample_weight = base.stabilize_ratio(sample_weight, cap=weight_cap)

    x_train = torch.as_tensor(Xs[train_idx], dtype=torch.float32, device=DEVICE)
    a_train = torch.as_tensor(A[train_idx].astype(int), dtype=torch.long, device=DEVICE)
    y_train = torch.as_tensor(target_np[train_idx], dtype=torch.float32, device=DEVICE)
    m_train = torch.as_tensor(mask_np[train_idx], dtype=torch.float32, device=DEVICE)
    w_train = torch.as_tensor(sample_weight[train_idx], dtype=torch.float32, device=DEVICE)
    x_val = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=DEVICE)
    a_val = torch.as_tensor(A[val_idx].astype(int), dtype=torch.long, device=DEVICE)
    y_val = torch.as_tensor(target_np[val_idx], dtype=torch.float32, device=DEVICE)
    m_val = torch.as_tensor(mask_np[val_idx], dtype=torch.float32, device=DEVICE)
    w_val = torch.as_tensor(sample_weight[val_idx], dtype=torch.float32, device=DEVICE)

    model = SharedArmDiscreteHazardNet(X.shape[1], n_actual_intervals).float().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = np.inf
    best_train = np.inf
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(epochs):
        set_module_mode(model, True)
        opt.zero_grad()
        loss = shared_arm_discrete_hazard_loss(model(x_train), a_train, y_train, m_train, sample_weight=w_train, loss_weight=loss_weight, event_weight=event_weight, event_weight_arm0=event_weight_arm0, event_weight_arm1=event_weight_arm1)
        loss.backward()
        opt.step()
        set_module_mode(model, False)
        with torch.no_grad():
            train_loss = float(shared_arm_discrete_hazard_loss(model(x_train), a_train, y_train, m_train, sample_weight=w_train, loss_weight=loss_weight, event_weight=event_weight, event_weight_arm0=event_weight_arm0, event_weight_arm1=event_weight_arm1).cpu())
            val_loss = float(shared_arm_discrete_hazard_loss(model(x_val), a_val, y_val, m_val, sample_weight=w_val, loss_weight=loss_weight, event_weight=event_weight, event_weight_arm0=event_weight_arm0, event_weight_arm1=event_weight_arm1).cpu())
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    full_epochs = max(1, int(best_epoch))
    set_torch_seed(seed + 7919)
    x_all = torch.as_tensor(Xs, dtype=torch.float32, device=DEVICE)
    a_all = torch.as_tensor(A.astype(int), dtype=torch.long, device=DEVICE)
    y_all = torch.as_tensor(target_np, dtype=torch.float32, device=DEVICE)
    m_all = torch.as_tensor(mask_np, dtype=torch.float32, device=DEVICE)
    w_all = torch.as_tensor(sample_weight, dtype=torch.float32, device=DEVICE)
    final_model = SharedArmDiscreteHazardNet(X.shape[1], n_actual_intervals).float().to(DEVICE)
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(full_epochs):
        set_module_mode(final_model, True)
        final_opt.zero_grad()
        full_loss = shared_arm_discrete_hazard_loss(final_model(x_all), a_all, y_all, m_all, sample_weight=w_all, loss_weight=loss_weight, event_weight=event_weight, event_weight_arm0=event_weight_arm0, event_weight_arm1=event_weight_arm1)
        full_loss.backward()
        final_opt.step()
    set_module_mode(final_model, False)
    with torch.no_grad():
        logits_all = final_model(x_all)
        full_refit_loss = float(shared_arm_discrete_hazard_loss(logits_all, a_all, y_all, m_all, sample_weight=w_all, loss_weight=loss_weight, event_weight=event_weight, event_weight_arm0=event_weight_arm0, event_weight_arm1=event_weight_arm1).cpu())
        logits_all_np = logits_all.detach().cpu().numpy()
    logit_offsets = np.zeros((2, n_actual_intervals), dtype=float)
    hazards_by_arm = 1.0 / (1.0 + np.exp(-np.clip(logits_all_np, -40.0, 40.0)))
    final_model = final_model.cpu()
    set_module_mode(final_model, False)
    return {
        "shared_arm_discrete_hazard": True,
        "model": final_model,
        "center": x_mean,
        "scale": x_scale,
        "time_grid": time_grid,
        "logit_offsets": logit_offsets,
        "pmf_grid": grid,
        "pmf_loss_weight": loss_weight,
        "shared_hazard_event_weight": float(event_weight),
        "shared_hazard_event_weight_arm0": float(event_weight if event_weight_arm0 is None else event_weight_arm0),
        "shared_hazard_event_weight_arm1": float(event_weight if event_weight_arm1 is None else event_weight_arm1),
        "pmf_outcome_weight": outcome_weight_label,
        "pmf_weight_cap": float(weight_cap),
        "pmf_weight_mean": float(np.mean(sample_weight)),
        "pmf_weight_sd": float(np.std(sample_weight, ddof=1)) if len(sample_weight) > 1 else 0.0,
        "pmf_weight_min": float(np.min(sample_weight)),
        "pmf_weight_max": float(np.max(sample_weight)),
        "pmf_weight_p90": float(np.quantile(sample_weight, 0.90)),
        "pmf_requested_intervals": int(n_intervals),
        "pmf_intervals": int(n_actual_intervals),
        "best_epoch": best_epoch,
        "best_train_loss": best_train,
        "best_val_loss": best_val,
        "full_refit_loss": full_refit_loss,
        "discrete_hazard_mean_arm0": float(np.mean(hazards_by_arm[:, 0, :])),
        "discrete_hazard_mean_arm1": float(np.mean(hazards_by_arm[:, 1, :])),
        "discrete_hazard_max_arm0": float(np.max(hazards_by_arm[:, 0, :])),
        "discrete_hazard_max_arm1": float(np.max(hazards_by_arm[:, 1, :])),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_events": int(np.sum(Delta)),
        "n_events_arm0": int(np.sum((A == 0) & (Delta == 1))),
        "n_events_arm1": int(np.sum((A == 1) & (Delta == 1))),
    }


def shared_arm_fit_for_prediction(shared_fit, arm):
    fitted = dict(shared_fit)
    fitted["arm"] = int(arm)
    return fitted


def fit_censoring_nuisance(X, A, U, Delta, seed, learner=None):
    learner = CENSOR_LEARNER if learner is None else learner
    if learner not in CENSOR_LEARNER_CHOICES:
        raise ValueError(f"Unknown censoring learner: {learner}")
    if learner == "cox":
        return {
            "learner": "cox",
            "model": base.fit_censoring_nuisance(X, A, U, Delta),
            "diagnostics": [],
        }
    censor_event = 1 - np.asarray(Delta, dtype=int)
    shared = fit_shared_arm_discrete_hazard(
        X,
        A,
        U,
        censor_event,
        seed=seed,
        n_intervals=PMF_SURVIVAL_INTERVALS,
        grid=PMF_SURVIVAL_GRID,
        loss_weight="none",
        event_weight=1.0,
        event_weight_arm0=None,
        event_weight_arm1=None,
        sample_weight=None,
        outcome_weight_label="none",
        weight_cap=PMF_WEIGHT_CAP,
        epochs=SHARED_HAZARD_EPOCHS,
    )
    diagnostics = [{
        "censor_learner": "neural",
        "censor_best_epoch": shared["best_epoch"],
        "censor_best_val_loss": shared["best_val_loss"],
        "censor_full_refit_loss": shared["full_refit_loss"],
        "censor_hazard_mean_arm0": shared["discrete_hazard_mean_arm0"],
        "censor_hazard_mean_arm1": shared["discrete_hazard_mean_arm1"],
        "censor_hazard_max_arm0": shared["discrete_hazard_max_arm0"],
        "censor_hazard_max_arm1": shared["discrete_hazard_max_arm1"],
        "censor_intervals": shared["pmf_intervals"],
        "censor_events": int(np.sum(censor_event)),
        "censor_events_arm0": int(np.sum((np.asarray(A) == 0) & (censor_event == 1))),
        "censor_events_arm1": int(np.sum((np.asarray(A) == 1) & (censor_event == 1))),
    }]
    return {
        "learner": "neural",
        "shared": shared,
        "fits": {
            0: shared_arm_fit_for_prediction(shared, 0),
            1: shared_arm_fit_for_prediction(shared, 1),
        },
        "diagnostics": diagnostics,
    }


def predict_survival_from_hazard_at_times(X, fitted, eval_times, min_survival=0.02):
    eval_times = np.asarray(eval_times, dtype=float)
    if eval_times.size == 0:
        return np.zeros((X.shape[0], 0), dtype=float)
    right = fitted_right_grid(fitted)
    dLambda = predict_dlambda_matrix(X, fitted)
    cumulative_hazard = np.cumsum(dLambda, axis=1)
    out = np.ones((X.shape[0], len(eval_times)), dtype=float)
    for j, t in enumerate(eval_times):
        idx = np.searchsorted(right, float(t), side="right") - 1
        if idx >= 0:
            out[:, j] = np.exp(-np.clip(cumulative_hazard[:, idx], 0.0, 700.0))
    return np.clip(out, min_survival, 1.0)


def censor_survival_matrix(X, A, censor_nuisance, eval_times, min_survival=0.02):
    if not isinstance(censor_nuisance, dict) or censor_nuisance.get("learner") == "cox":
        model = censor_nuisance.get("model") if isinstance(censor_nuisance, dict) else censor_nuisance
        return base.censor_survival_matrix(X, A, model, eval_times, min_survival=min_survival)
    if censor_nuisance.get("learner") != "neural":
        raise ValueError(f"Unknown censoring learner: {censor_nuisance.get('learner')}")
    A = np.asarray(A, dtype=int)
    out = np.ones((X.shape[0], len(eval_times)), dtype=float)
    for arm in [0, 1]:
        idx = A == arm
        if np.any(idx):
            out[idx] = predict_survival_from_hazard_at_times(
                X[idx],
                censor_nuisance["fits"][arm],
                eval_times,
                min_survival=min_survival,
            )
    return out


def weibull_ph_loss(model, x, time, event, eta_offset=None):
    eta_raw = model(x)
    if eta_offset is None:
        eta = eta_raw - eta_raw.mean()
    else:
        eta = eta_raw - eta_offset
    eta = torch.clamp(eta, -8.0, 8.0)
    t = torch.clamp(time, min=WEIBULL_MIN_TIME)
    log_t = torch.log(t)
    shape, scale = model.shape_scale()
    log_scale = torch.log(scale)
    log_base_cum = torch.clamp(shape * (log_t - log_scale), -30.0, 30.0)
    base_cum = torch.exp(log_base_cum)
    cumulative_hazard = torch.exp(eta) * base_cum
    log_hazard = torch.log(shape) - log_scale + (shape - 1.0) * (log_t - log_scale) + eta
    return torch.mean(cumulative_hazard - event * log_hazard)


def make_weibull_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        [
            {"params": model.net.parameters(), "weight_decay": weight_decay},
            {"params": [model.log_shape, model.log_scale], "weight_decay": 0.0},
        ],
        lr=lr,
    )


def fit_neural_weibull_ph(X, U, Delta, seed, epochs=250, lr=1e-3, weight_decay=1e-4, patience=25):
    if np.sum(Delta) < 5:
        return None
    set_torch_seed(seed)
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    Xs = (X - x_mean) / x_scale
    train_idx, val_idx = validation_split_indices(Delta, seed)
    init_scale = float(np.clip(np.median(U), WEIBULL_SCALE_BOUNDS[0], WEIBULL_SCALE_BOUNDS[1]))

    x_train = torch.as_tensor(Xs[train_idx], dtype=torch.float32, device=DEVICE)
    u_train = torch.as_tensor(U[train_idx], dtype=torch.float32, device=DEVICE)
    d_train = torch.as_tensor(Delta[train_idx].astype(float), dtype=torch.float32, device=DEVICE)
    x_val = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=DEVICE)
    u_val = torch.as_tensor(U[val_idx], dtype=torch.float32, device=DEVICE)
    d_val = torch.as_tensor(Delta[val_idx].astype(float), dtype=torch.float32, device=DEVICE)

    model = WeibullPHNet(X.shape[1], init_shape=1.2, init_scale=init_scale).float().to(DEVICE)
    opt = make_weibull_optimizer(model, lr, weight_decay)
    best_val = np.inf
    best_train = np.inf
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(epochs):
        set_module_mode(model, True)
        opt.zero_grad()
        loss = weibull_ph_loss(model, x_train, u_train, d_train)
        loss.backward()
        opt.step()
        set_module_mode(model, False)
        with torch.no_grad():
            train_eta_offset = model(x_train).mean()
            train_loss = float(weibull_ph_loss(model, x_train, u_train, d_train, eta_offset=train_eta_offset).cpu())
            val_loss = float(weibull_ph_loss(model, x_val, u_val, d_val, eta_offset=train_eta_offset).cpu())
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    full_epochs = max(1, int(best_epoch))
    set_torch_seed(seed + 7919)
    x_all = torch.as_tensor(Xs, dtype=torch.float32, device=DEVICE)
    u_all = torch.as_tensor(U, dtype=torch.float32, device=DEVICE)
    d_all = torch.as_tensor(Delta.astype(float), dtype=torch.float32, device=DEVICE)
    final_model = WeibullPHNet(X.shape[1], init_shape=1.2, init_scale=init_scale).float().to(DEVICE)
    final_opt = make_weibull_optimizer(final_model, lr, weight_decay)
    for _ in range(full_epochs):
        set_module_mode(final_model, True)
        final_opt.zero_grad()
        full_loss = weibull_ph_loss(final_model, x_all, u_all, d_all)
        full_loss.backward()
        final_opt.step()
    set_module_mode(final_model, False)
    with torch.no_grad():
        eta_raw = final_model(x_all).detach().cpu().numpy()
        eta_offset = float(np.mean(eta_raw))
        full_refit_loss = float(weibull_ph_loss(final_model, x_all, u_all, d_all, eta_offset=torch.as_tensor(eta_offset, dtype=torch.float32, device=DEVICE)).cpu())
        shape_t, scale_t = final_model.shape_scale()
        weibull_shape = float(shape_t.detach().cpu())
        weibull_scale = float(scale_t.detach().cpu())
    final_model = final_model.cpu()
    set_module_mode(final_model, False)
    return {
        "weibull_ph": True,
        "model": final_model,
        "center": x_mean,
        "scale": x_scale,
        "eta_offset": eta_offset,
        "time_grid": make_discrete_time_grid(TAU, WEIBULL_HAZARD_INTERVALS),
        "weibull_shape": weibull_shape,
        "weibull_scale": weibull_scale,
        "weibull_eta_sd": float(np.std(eta_raw - eta_offset, ddof=1)),
        "best_epoch": best_epoch,
        "best_train_loss": best_train,
        "best_val_loss": best_val,
        "full_refit_loss": full_refit_loss,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_events": int(np.sum(Delta)),
    }


def torch_weighted_mean(values, weights=None):
    if weights is None:
        return values.mean()
    weights = torch.clamp(weights, min=0.0)
    denom = torch.clamp(weights.sum(), min=PMF_SURVIVAL_MIN_PROB)
    return torch.sum(values * weights) / denom


def deephit_survival_loss(
    logits,
    time,
    event,
    time_grid,
    loss_weight=PMF_SURVIVAL_LOSS_WEIGHT,
    sample_weight=None,
    ranking_weight=DEEPHIT_RANKING_WEIGHT,
    ranking_sigma=DEEPHIT_RANKING_SIGMA,
):
    """Full single-risk DeepHit objective for right-censored data.

    The first term is the discrete-time event/censoring likelihood.  The
    second term is the DeepHit pairwise ranking loss: an observed event at
    t_i should have larger cumulative incidence by t_i than subjects whose
    observed follow-up exceeds t_i.  The final PMF bin is an administrative
    tail bin and is excluded from the ranking CDF.
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    right = torch.as_tensor(time_grid[1:], dtype=torch.float32, device=logits.device)
    tau = float(time_grid[-1])
    event_in_tau = (event > 0.5) & (time <= tau)
    event_loss = None
    event_weight = None
    censored_loss = None
    censored_weight = None
    if torch.any(event_in_tau):
        bins = torch.searchsorted(right, torch.clamp(time[event_in_tau], max=tau), right=False)
        bins = torch.clamp(bins, 0, len(right) - 1)
        event_loss = -log_probs[event_in_tau, bins]
        event_weight = sample_weight[event_in_tau] if sample_weight is not None else None
    censored_or_tail = ~event_in_tau
    if torch.any(censored_or_tail):
        t_c = torch.clamp(time[censored_or_tail], max=tau)
        event_bin_right_after_c = right[None, :] > t_c[:, None]
        tail_mask = torch.ones((event_bin_right_after_c.shape[0], 1), dtype=torch.bool, device=logits.device)
        surv_mask = torch.cat([event_bin_right_after_c, tail_mask], dim=1)
        surv_prob = torch.sum(probs[censored_or_tail] * surv_mask.float(), dim=1)
        censored_loss = -torch.log(torch.clamp(surv_prob, min=PMF_SURVIVAL_MIN_PROB))
        censored_weight = sample_weight[censored_or_tail] if sample_weight is not None else None
    losses = [x for x in [event_loss, censored_loss] if x is not None]
    if not losses:
        return torch.mean(logits * 0.0)
    if loss_weight == "event_balanced" and event_loss is not None and censored_loss is not None:
        nll = 0.5 * torch_weighted_mean(event_loss, event_weight) + 0.5 * torch_weighted_mean(censored_loss, censored_weight)
    elif loss_weight != "none":
        if loss_weight != "event_balanced":
            raise ValueError(f"Unknown DeepHit likelihood weighting: {loss_weight}")
        nll = torch.mean(torch.cat(losses))
    elif sample_weight is None:
        nll = torch.mean(torch.cat(losses))
    else:
        weights = [w for w in [event_weight, censored_weight] if w is not None]
        nll = torch_weighted_mean(torch.cat(losses), torch.cat(weights))

    # DeepHit ranking term.  Use the event CDF over the actual event bins;
    # each observed event is compared with subjects still under observation
    # after that event time.  The pairwise objective is evaluated in a
    # vectorized O(n^2) form to avoid Python/GPU synchronization per event.
    rank_terms = []
    rank_weights = []
    if ranking_weight > 0.0 and torch.any(event_in_tau):
        event_indices = torch.nonzero(event_in_tau, as_tuple=False).reshape(-1)
        event_bins = torch.searchsorted(right, torch.clamp(time[event_indices], max=tau), right=False)
        event_bins = torch.clamp(event_bins, 0, len(right) - 1)
        cdf = torch.cumsum(probs[:, :-1], dim=1)
        sigma = max(float(ranking_sigma), 1e-4)
        # cdf[:, event_bins].T gives, for each event i, the CDF at i's event
        # bin evaluated for every comparison subject j.
        f_i = cdf[event_indices, event_bins][:, None]
        f_j = cdf[:, event_bins].T
        comparable = time[None, :] > time[event_indices, None]
        if torch.any(comparable):
            pair_loss = torch.exp(torch.clamp(-(f_i - f_j) / sigma, -20.0, 20.0))
            rank_terms.append(pair_loss[comparable])
            if sample_weight is None:
                rank_weights.append(torch.ones_like(pair_loss[comparable]))
            else:
                pair_weights = sample_weight[event_indices, None] * sample_weight[None, :]
                rank_weights.append(pair_weights[comparable])
    if rank_terms:
        ranking_loss = torch_weighted_mean(torch.cat(rank_terms), torch.cat(rank_weights))
    else:
        ranking_loss = torch.mean(logits * 0.0)
    return nll + float(ranking_weight) * ranking_loss


# Compatibility alias for code written against the earlier PMF-only module.
pmf_survival_loss = deephit_survival_loss


def deephit_loss_components(
    logits,
    time,
    event,
    time_grid,
    sample_weight=None,
    ranking_sigma=DEEPHIT_RANKING_SIGMA,
):
    """Return likelihood and ranking components for diagnostics."""
    # Temporarily evaluate the combined objective with two settings so the
    # public training path remains compact and the reported components match
    # the exact objective used for optimization.
    total = deephit_survival_loss(
        logits, time, event, time_grid, sample_weight=sample_weight,
        ranking_weight=DEEPHIT_RANKING_WEIGHT, ranking_sigma=ranking_sigma,
    )
    nll = deephit_survival_loss(
        logits, time, event, time_grid, sample_weight=sample_weight,
        ranking_weight=0.0, ranking_sigma=ranking_sigma,
    )
    ranking = (total - nll) / max(DEEPHIT_RANKING_WEIGHT, 1e-8)
    return nll, ranking


def fit_neural_pmf_survival(
    X,
    U,
    Delta,
    seed,
    n_intervals=PMF_SURVIVAL_INTERVALS,
    grid=PMF_SURVIVAL_GRID,
    loss_weight=PMF_SURVIVAL_LOSS_WEIGHT,
    sample_weight=None,
    outcome_weight_label=PMF_OUTCOME_WEIGHT,
    weight_cap=PMF_WEIGHT_CAP,
    epochs=250,
    lr=1e-3,
    weight_decay=1e-4,
    patience=25,
    ranking_weight=DEEPHIT_RANKING_WEIGHT,
    ranking_sigma=DEEPHIT_RANKING_SIGMA,
):
    if np.sum(Delta) < 5:
        return None
    set_torch_seed(seed)
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    Xs = (X - x_mean) / x_scale
    time_grid = make_pmf_time_grid(U, Delta, tau=TAU, n_intervals=n_intervals, grid=grid)
    n_actual_intervals = len(time_grid) - 1
    train_idx, val_idx = validation_split_indices(Delta, seed)
    if sample_weight is None:
        sample_weight = np.ones(len(U), dtype=float)
    sample_weight = base.stabilize_ratio(sample_weight, cap=weight_cap)

    x_train = torch.as_tensor(Xs[train_idx], dtype=torch.float32, device=DEVICE)
    u_train = torch.as_tensor(U[train_idx], dtype=torch.float32, device=DEVICE)
    d_train = torch.as_tensor(Delta[train_idx].astype(float), dtype=torch.float32, device=DEVICE)
    w_train = torch.as_tensor(sample_weight[train_idx], dtype=torch.float32, device=DEVICE)
    x_val = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=DEVICE)
    u_val = torch.as_tensor(U[val_idx], dtype=torch.float32, device=DEVICE)
    d_val = torch.as_tensor(Delta[val_idx].astype(float), dtype=torch.float32, device=DEVICE)
    w_val = torch.as_tensor(sample_weight[val_idx], dtype=torch.float32, device=DEVICE)

    model = DeepHitSurvivalNet(X.shape[1], n_actual_intervals).float().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = np.inf
    best_train = np.inf
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(epochs):
        set_module_mode(model, True)
        opt.zero_grad()
        loss = deephit_survival_loss(
            model(x_train), u_train, d_train, time_grid,
            loss_weight=loss_weight, sample_weight=w_train,
            ranking_weight=ranking_weight, ranking_sigma=ranking_sigma,
        )
        loss.backward()
        opt.step()
        set_module_mode(model, False)
        with torch.no_grad():
            train_loss = float(deephit_survival_loss(
                model(x_train), u_train, d_train, time_grid,
                loss_weight=loss_weight, sample_weight=w_train,
                ranking_weight=ranking_weight, ranking_sigma=ranking_sigma,
            ).cpu())
            val_loss = float(deephit_survival_loss(
                model(x_val), u_val, d_val, time_grid,
                loss_weight=loss_weight, sample_weight=w_val,
                ranking_weight=ranking_weight, ranking_sigma=ranking_sigma,
            ).cpu())
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    full_epochs = max(1, int(best_epoch))
    set_torch_seed(seed + 7919)
    x_all = torch.as_tensor(Xs, dtype=torch.float32, device=DEVICE)
    u_all = torch.as_tensor(U, dtype=torch.float32, device=DEVICE)
    d_all = torch.as_tensor(Delta.astype(float), dtype=torch.float32, device=DEVICE)
    w_all = torch.as_tensor(sample_weight, dtype=torch.float32, device=DEVICE)
    final_model = DeepHitSurvivalNet(X.shape[1], n_actual_intervals).float().to(DEVICE)
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(full_epochs):
        set_module_mode(final_model, True)
        final_opt.zero_grad()
        full_loss = deephit_survival_loss(
            final_model(x_all), u_all, d_all, time_grid,
            loss_weight=loss_weight, sample_weight=w_all,
            ranking_weight=ranking_weight, ranking_sigma=ranking_sigma,
        )
        full_loss.backward()
        final_opt.step()
    set_module_mode(final_model, False)
    with torch.no_grad():
        logits_all = final_model(x_all)
        full_refit_loss = float(deephit_survival_loss(
            logits_all, u_all, d_all, time_grid,
            loss_weight=loss_weight, sample_weight=w_all,
            ranking_weight=ranking_weight, ranking_sigma=ranking_sigma,
        ).cpu())
        full_refit_nll, full_refit_ranking = deephit_loss_components(
            logits_all, u_all, d_all, time_grid,
            sample_weight=w_all, ranking_sigma=ranking_sigma,
        )
        full_refit_nll = float(full_refit_nll.cpu())
        full_refit_ranking = float(full_refit_ranking.cpu())
        probs_all = F.softmax(logits_all, dim=1).detach().cpu().numpy()
    final_model = final_model.cpu()
    set_module_mode(final_model, False)
    entropy = -np.sum(probs_all * np.log(np.clip(probs_all, PMF_SURVIVAL_MIN_PROB, 1.0)), axis=1)
    return {
        "pmf_survival": True,
        "deephit_survival": True,
        "model": final_model,
        "center": x_mean,
        "scale": x_scale,
        "time_grid": time_grid,
        "pmf_grid": grid,
        "pmf_loss_weight": loss_weight,
        "deephit_ranking_weight": float(ranking_weight),
        "deephit_ranking_sigma": float(ranking_sigma),
        "pmf_outcome_weight": outcome_weight_label,
        "pmf_weight_cap": float(weight_cap),
        "pmf_weight_mean": float(np.mean(sample_weight)),
        "pmf_weight_sd": float(np.std(sample_weight, ddof=1)) if len(sample_weight) > 1 else 0.0,
        "pmf_weight_min": float(np.min(sample_weight)),
        "pmf_weight_max": float(np.max(sample_weight)),
        "pmf_weight_p90": float(np.quantile(sample_weight, 0.90)),
        "pmf_requested_intervals": int(n_intervals),
        "pmf_intervals": int(n_actual_intervals),
        "pmf_tail_mean": float(np.mean(probs_all[:, -1])),
        "pmf_entropy_mean": float(np.mean(entropy)),
        "best_epoch": best_epoch,
        "best_train_loss": best_train,
        "best_val_loss": best_val,
        "full_refit_loss": full_refit_loss,
        "full_refit_nll": full_refit_nll,
        "full_refit_ranking": full_refit_ranking,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_events": int(np.sum(Delta)),
    }


def fit_residual_projection(Phi, residual, ridge=RESIDUAL_ORTHOGONAL_RIDGE):
    Phi = np.asarray(Phi, dtype=float)
    residual = np.asarray(residual, dtype=float)
    Z = np.column_stack([np.ones(Phi.shape[0]), Phi])
    penalty = ridge * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(Z.T @ Z + penalty, Z.T @ residual)
    fitted = Z @ coef
    resid_orth = residual - fitted
    denom = float(np.sum((residual - residual.mean()) ** 2))
    r2 = 0.0 if denom <= 1e-12 else 1.0 - float(np.sum(resid_orth**2)) / denom
    return coef, resid_orth, float(np.clip(r2, 0.0, 1.0))


def apply_residual_projection(Phi, residual, coef):
    Phi = np.asarray(Phi, dtype=float)
    residual = np.asarray(residual, dtype=float)
    Z = np.column_stack([np.ones(Phi.shape[0]), Phi])
    return residual - Z @ coef


def select_gated_alpha(offset_val, residual_val, U_val, Delta_val):
    base_contrib = cox_event_loss_contributions(offset_val, U_val, Delta_val)
    losses = []
    contributions = []
    for alpha in GATED_ALPHA_GRID:
        eta_val = offset_val + alpha * residual_val
        contrib = cox_event_loss_contributions(eta_val, U_val, Delta_val)
        contributions.append(contrib)
        losses.append(float(np.mean(contrib)))
    losses = np.asarray(losses, dtype=float)
    best_idx = int(np.argmin(losses))
    best_alpha = float(GATED_ALPHA_GRID[best_idx])
    linear_loss = float(losses[0])
    best_loss = float(losses[best_idx])
    if len(base_contrib) == len(contributions[best_idx]) and len(base_contrib) > 1:
        gain_contrib = base_contrib - contributions[best_idx]
        gain_se = float(np.std(gain_contrib, ddof=1) / np.sqrt(len(gain_contrib)))
    else:
        _, gain_se = cox_event_loss_mean_se(offset_val + best_alpha * residual_val, U_val, Delta_val)
    validation_gain = float(linear_loss - best_loss)
    min_gain = max(GATED_SE_MULTIPLIER * gain_se, GATED_MIN_GAIN)
    if best_idx == 0 or validation_gain < min_gain:
        return {
            "alpha": 0.0,
            "linear_val_loss": linear_loss,
            "best_val_loss": linear_loss,
            "raw_best_alpha": best_alpha,
            "raw_best_val_loss": best_loss,
            "validation_gain": validation_gain,
            "validation_gain_se": gain_se,
        }
    one_se_limit = best_loss + gain_se
    chosen_idx = next(i for i, loss in enumerate(losses) if loss <= one_se_limit)
    alpha = float(GATED_ALPHA_GRID[chosen_idx])
    return {
        "alpha": alpha,
        "linear_val_loss": linear_loss,
        "best_val_loss": float(losses[chosen_idx]),
        "raw_best_alpha": best_alpha,
        "raw_best_val_loss": best_loss,
        "validation_gain": validation_gain,
        "validation_gain_se": gain_se,
    }


def fit_neural_cox(X, U, Delta, learner, seed, epochs=250, lr=1e-3, weight_decay=1e-4, patience=25):
    if np.sum(Delta) < 5:
        return None
    set_torch_seed(seed)
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    Xs = (X - x_mean) / x_scale
    train_idx, val_idx = validation_split_indices(Delta, seed)
    x_train = torch.as_tensor(Xs[train_idx], dtype=torch.float32, device=DEVICE)
    u_train = torch.as_tensor(U[train_idx], dtype=torch.float32, device=DEVICE)
    d_train = torch.as_tensor(Delta[train_idx].astype(float), dtype=torch.float32, device=DEVICE)
    x_val = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=DEVICE)
    u_val = torch.as_tensor(U[val_idx], dtype=torch.float32, device=DEVICE)
    d_val = torch.as_tensor(Delta[val_idx].astype(float), dtype=torch.float32, device=DEVICE)
    model = make_neural_cox_model(learner, X.shape[1], seed)
    model.to(DEVICE)
    maybe_update_coxkan_grid(model, x_train)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = np.inf
    best_train = np.inf
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(epochs):
        if learner == "coxkan" and epoch < COXKAN_GRID_UPDATE_EPOCHS and epoch % COXKAN_GRID_UPDATE_FREQ == 0:
            maybe_update_coxkan_grid(model, x_train)
        set_module_mode(model, True)
        opt.zero_grad()
        loss = cox_ph_loss(model_eta(model, x_train), u_train, d_train)
        loss.backward()
        opt.step()
        set_module_mode(model, False)
        with torch.no_grad():
            train_loss = float(cox_ph_loss(model_eta(model, x_train), u_train, d_train).cpu())
            val_loss = float(cox_ph_loss(model_eta(model, x_val), u_val, d_val).cpu())
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    full_epochs = max(1, int(best_epoch))
    set_torch_seed(seed + 7919)
    x_all = torch.as_tensor(Xs, dtype=torch.float32, device=DEVICE)
    u_all = torch.as_tensor(U, dtype=torch.float32, device=DEVICE)
    d_all = torch.as_tensor(Delta.astype(float), dtype=torch.float32, device=DEVICE)
    final_model = make_neural_cox_model(learner, X.shape[1], seed + 7919)
    final_model.to(DEVICE)
    maybe_update_coxkan_grid(final_model, x_all)
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(full_epochs):
        if learner == "coxkan" and epoch < COXKAN_GRID_UPDATE_EPOCHS and epoch % COXKAN_GRID_UPDATE_FREQ == 0:
            maybe_update_coxkan_grid(final_model, x_all)
        set_module_mode(final_model, True)
        final_opt.zero_grad()
        full_loss = cox_ph_loss(model_eta(final_model, x_all), u_all, d_all)
        full_loss.backward()
        final_opt.step()
    set_module_mode(final_model, False)
    with torch.no_grad():
        full_refit_loss = float(cox_ph_loss(model_eta(final_model, x_all), u_all, d_all).cpu())
    model = final_model
    with torch.no_grad():
        eta_raw = model_eta(model, x_all).detach().cpu().numpy()
    eta_offset = float(np.mean(eta_raw))
    eta_centered = eta_raw - eta_offset
    try:
        eta_scale = float(base.fit_cox_partial_likelihood(eta_centered[:, None], U, Delta, ridge=1e-4)[0])
        if not np.isfinite(eta_scale):
            eta_scale = 1.0
    except Exception:
        eta_scale = 1.0
    eta_scale = float(np.clip(eta_scale, 0.25, 4.0))
    model = model.cpu()
    set_module_mode(model, False)
    return {
        "model": model,
        "center": x_mean,
        "scale": x_scale,
        "eta_offset": eta_offset,
        "eta_scale": eta_scale,
        "best_epoch": best_epoch,
        "best_train_loss": best_train,
        "best_val_loss": best_val,
        "full_refit_loss": full_refit_loss,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_events": int(np.sum(Delta)),
    }


def fit_hybrid_deepsurv(X, U, Delta, seed, epochs=250, lr=1e-3, weight_decay=1e-4, patience=25):
    if np.sum(Delta) < 5:
        return None
    set_torch_seed(seed)
    Phi = base.phi_linear(X)
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    Xs = (X - x_mean) / x_scale
    train_idx, val_idx = validation_split_indices(Delta, seed)

    beta_train = base.fit_cox_partial_likelihood(Phi[train_idx], U[train_idx], Delta[train_idx], ridge=1e-4)
    offset_train_np = Phi[train_idx] @ beta_train
    offset_val_np = Phi[val_idx] @ beta_train

    x_train = torch.as_tensor(Xs[train_idx], dtype=torch.float32, device=DEVICE)
    u_train = torch.as_tensor(U[train_idx], dtype=torch.float32, device=DEVICE)
    d_train = torch.as_tensor(Delta[train_idx].astype(float), dtype=torch.float32, device=DEVICE)
    off_train = torch.as_tensor(offset_train_np, dtype=torch.float32, device=DEVICE)
    x_val = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=DEVICE)
    u_val = torch.as_tensor(U[val_idx], dtype=torch.float32, device=DEVICE)
    d_val = torch.as_tensor(Delta[val_idx].astype(float), dtype=torch.float32, device=DEVICE)
    off_val = torch.as_tensor(offset_val_np, dtype=torch.float32, device=DEVICE)

    model = DeepSurvNet(X.shape[1]).float().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = np.inf
    best_train = np.inf
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(epochs):
        set_module_mode(model, True)
        opt.zero_grad()
        loss = cox_ph_loss(off_train + model_eta(model, x_train), u_train, d_train)
        loss.backward()
        opt.step()
        set_module_mode(model, False)
        with torch.no_grad():
            train_loss = float(cox_ph_loss(off_train + model_eta(model, x_train), u_train, d_train).cpu())
            val_loss = float(cox_ph_loss(off_val + model_eta(model, x_val), u_val, d_val).cpu())
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    set_module_mode(model, False)
    with torch.no_grad():
        residual_train = model_eta(model, x_train).detach().cpu().numpy()
        residual_val = model_eta(model, x_val).detach().cpu().numpy()
    residual_offset_train = float(np.mean(residual_train))
    alpha_losses = []
    for alpha in HYBRID_ALPHA_GRID:
        eta_val = offset_val_np + alpha * (residual_val - residual_offset_train)
        alpha_losses.append(cox_ph_loss_numpy(eta_val, U[val_idx], Delta[val_idx]))
    best_alpha_idx = int(np.argmin(alpha_losses))
    alpha = float(HYBRID_ALPHA_GRID[best_alpha_idx])
    linear_val_loss = float(alpha_losses[0])
    hybrid_val_loss = float(alpha_losses[best_alpha_idx])

    beta_all = base.fit_cox_partial_likelihood(Phi, U, Delta, ridge=1e-4)
    offset_all_np = Phi @ beta_all
    off_all = torch.as_tensor(offset_all_np, dtype=torch.float32, device=DEVICE)
    x_all = torch.as_tensor(Xs, dtype=torch.float32, device=DEVICE)
    u_all = torch.as_tensor(U, dtype=torch.float32, device=DEVICE)
    d_all = torch.as_tensor(Delta.astype(float), dtype=torch.float32, device=DEVICE)
    full_epochs = max(1, int(best_epoch))
    set_torch_seed(seed + 7919)
    final_model = DeepSurvNet(X.shape[1]).float().to(DEVICE)
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(full_epochs):
        set_module_mode(final_model, True)
        final_opt.zero_grad()
        full_loss = cox_ph_loss(off_all + model_eta(final_model, x_all), u_all, d_all)
        full_loss.backward()
        final_opt.step()
    set_module_mode(final_model, False)
    with torch.no_grad():
        residual_all = model_eta(final_model, x_all).detach().cpu().numpy()
        full_refit_loss = float(cox_ph_loss(off_all + model_eta(final_model, x_all), u_all, d_all).cpu())
    final_model = final_model.cpu()
    set_module_mode(final_model, False)
    return {
        "hybrid": True,
        "model": final_model,
        "beta": beta_all,
        "center": x_mean,
        "scale": x_scale,
        "eta_offset": float(np.mean(residual_all)),
        "eta_scale": 1.0,
        "alpha": alpha,
        "best_epoch": best_epoch,
        "best_train_loss": best_train,
        "best_val_loss": hybrid_val_loss,
        "linear_val_loss": linear_val_loss,
        "raw_hybrid_val_loss": best_val,
        "full_refit_loss": full_refit_loss,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_events": int(np.sum(Delta)),
    }


def fit_hybrid_deepsurv_gated(X, U, Delta, seed, epochs=250, lr=1e-3, weight_decay=1e-4, patience=25):
    if np.sum(Delta) < 5:
        return None
    set_torch_seed(seed)
    Phi = base.phi_linear(X)
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    Xs = (X - x_mean) / x_scale
    train_idx, val_idx = validation_split_indices(Delta, seed)

    beta_train = base.fit_cox_partial_likelihood(Phi[train_idx], U[train_idx], Delta[train_idx], ridge=1e-4)
    offset_train_np = Phi[train_idx] @ beta_train
    offset_val_np = Phi[val_idx] @ beta_train

    x_train = torch.as_tensor(Xs[train_idx], dtype=torch.float32, device=DEVICE)
    u_train = torch.as_tensor(U[train_idx], dtype=torch.float32, device=DEVICE)
    d_train = torch.as_tensor(Delta[train_idx].astype(float), dtype=torch.float32, device=DEVICE)
    off_train = torch.as_tensor(offset_train_np, dtype=torch.float32, device=DEVICE)
    x_val = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=DEVICE)
    u_val = torch.as_tensor(U[val_idx], dtype=torch.float32, device=DEVICE)
    d_val = torch.as_tensor(Delta[val_idx].astype(float), dtype=torch.float32, device=DEVICE)
    off_val = torch.as_tensor(offset_val_np, dtype=torch.float32, device=DEVICE)

    model = DeepSurvNet(X.shape[1]).float().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = np.inf
    best_train = np.inf
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(epochs):
        set_module_mode(model, True)
        opt.zero_grad()
        loss = cox_ph_loss(off_train + model_eta(model, x_train), u_train, d_train)
        loss.backward()
        opt.step()
        set_module_mode(model, False)
        with torch.no_grad():
            train_loss = float(cox_ph_loss(off_train + model_eta(model, x_train), u_train, d_train).cpu())
            val_loss = float(cox_ph_loss(off_val + model_eta(model, x_val), u_val, d_val).cpu())
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    set_module_mode(model, False)
    with torch.no_grad():
        residual_train = model_eta(model, x_train).detach().cpu().numpy()
        residual_val = model_eta(model, x_val).detach().cpu().numpy()
    projection_coef_train, residual_train_orth, projection_r2 = fit_residual_projection(
        Phi[train_idx],
        residual_train,
        ridge=RESIDUAL_ORTHOGONAL_RIDGE,
    )
    residual_val_orth = apply_residual_projection(Phi[val_idx], residual_val, projection_coef_train)
    alpha_info = select_gated_alpha(offset_val_np, residual_val_orth, U[val_idx], Delta[val_idx])

    beta_all = base.fit_cox_partial_likelihood(Phi, U, Delta, ridge=1e-4)
    offset_all_np = Phi @ beta_all
    off_all = torch.as_tensor(offset_all_np, dtype=torch.float32, device=DEVICE)
    x_all = torch.as_tensor(Xs, dtype=torch.float32, device=DEVICE)
    u_all = torch.as_tensor(U, dtype=torch.float32, device=DEVICE)
    d_all = torch.as_tensor(Delta.astype(float), dtype=torch.float32, device=DEVICE)
    full_epochs = max(1, int(best_epoch))
    set_torch_seed(seed + 7919)
    final_model = DeepSurvNet(X.shape[1]).float().to(DEVICE)
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(full_epochs):
        set_module_mode(final_model, True)
        final_opt.zero_grad()
        full_loss = cox_ph_loss(off_all + model_eta(final_model, x_all), u_all, d_all)
        full_loss.backward()
        final_opt.step()
    set_module_mode(final_model, False)
    with torch.no_grad():
        residual_all = model_eta(final_model, x_all).detach().cpu().numpy()
        full_refit_loss = float(cox_ph_loss(off_all + model_eta(final_model, x_all), u_all, d_all).cpu())
    projection_coef_all, _, _ = fit_residual_projection(Phi, residual_all, ridge=RESIDUAL_ORTHOGONAL_RIDGE)
    final_model = final_model.cpu()
    set_module_mode(final_model, False)
    return {
        "hybrid": True,
        "gated": True,
        "model": final_model,
        "beta": beta_all,
        "projection_coef": projection_coef_all,
        "center": x_mean,
        "scale": x_scale,
        "eta_offset": 0.0,
        "eta_scale": 1.0,
        "alpha": float(alpha_info["alpha"]),
        "raw_best_alpha": float(alpha_info["raw_best_alpha"]),
        "best_epoch": best_epoch,
        "best_train_loss": best_train,
        "best_val_loss": float(alpha_info["best_val_loss"]),
        "linear_val_loss": float(alpha_info["linear_val_loss"]),
        "raw_best_val_loss": float(alpha_info["raw_best_val_loss"]),
        "validation_gain": float(alpha_info["validation_gain"]),
        "validation_gain_se": float(alpha_info["validation_gain_se"]),
        "residual_projection_r2": projection_r2,
        "full_refit_loss": full_refit_loss,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_events": int(np.sum(Delta)),
    }


def predict_eta_neural(X, fitted):
    Xs = (X - fitted["center"]) / fitted["scale"]
    with torch.no_grad():
        eta = model_eta(fitted["model"], torch.as_tensor(Xs, dtype=torch.float32)).cpu().numpy()
    if fitted.get("hybrid", False):
        eta_linear = base.phi_linear(X) @ fitted["beta"]
        if fitted.get("gated", False):
            eta = apply_residual_projection(base.phi_linear(X), eta, fitted["projection_coef"])
            eta = eta_linear + fitted["alpha"] * eta
            return np.clip(eta, -8.0, 8.0)
        eta = eta_linear + fitted["alpha"] * (eta - fitted["eta_offset"])
        return np.clip(eta, -8.0, 8.0)
    eta = fitted["eta_scale"] * (eta - fitted["eta_offset"])
    return np.clip(eta, -8.0, 8.0)


def predict_discrete_hazard(X, fitted):
    Xs = (X - fitted["center"]) / fitted["scale"]
    with torch.no_grad():
        logits = fitted["model"](torch.as_tensor(Xs, dtype=torch.float32))
        logits_np = logits.cpu().numpy()
    if fitted.get("shared_arm_discrete_hazard", False):
        arm = int(fitted["arm"])
        logits_np = logits_np[:, arm, :]
        if "logit_offsets" in fitted:
            logits_np = logits_np + fitted["logit_offsets"][arm][None, :]
        hazard = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -40.0, 40.0)))
        return np.clip(hazard, DISCRETE_HAZARD_MIN, DISCRETE_HAZARD_MAX)
    if "logit_offsets" in fitted:
        logits_np = logits_np + fitted["logit_offsets"][None, :]
    hazard = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -40.0, 40.0)))
    return np.clip(hazard, DISCRETE_HAZARD_MIN, DISCRETE_HAZARD_MAX)


def predict_dlambda_discrete(X, fitted):
    hazard = predict_discrete_hazard(X, fitted)
    return -np.log1p(-hazard)


def predict_eta_weibull(X, fitted):
    Xs = (X - fitted["center"]) / fitted["scale"]
    with torch.no_grad():
        eta = model_eta(fitted["model"], torch.as_tensor(Xs, dtype=torch.float32)).cpu().numpy()
    return np.clip(eta - fitted["eta_offset"], -8.0, 8.0)


def predict_dlambda_weibull(X, fitted):
    eta = predict_eta_weibull(X, fitted)
    time_grid = fitted["time_grid"]
    left = np.asarray(time_grid[:-1], dtype=float)
    right = np.asarray(time_grid[1:], dtype=float)
    shape = float(fitted["weibull_shape"])
    scale = float(fitted["weibull_scale"])
    base_left = np.where(left > 0.0, (left / scale) ** shape, 0.0)
    base_right = (right / scale) ** shape
    base_inc = np.maximum(base_right - base_left, 0.0)
    return np.exp(np.clip(eta, -30.0, 30.0))[:, None] * base_inc[None, :]


def predict_survival_pmf(X, fitted):
    Xs = (X - fitted["center"]) / fitted["scale"]
    with torch.no_grad():
        logits = fitted["model"](torch.as_tensor(Xs, dtype=torch.float32))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    event_probs = np.clip(probs[:, :-1], PMF_SURVIVAL_MIN_PROB, 1.0)
    tail = np.clip(probs[:, -1], PMF_SURVIVAL_MIN_PROB, 1.0)
    return np.cumsum(event_probs[:, ::-1], axis=1)[:, ::-1] + tail[:, None], tail


def predict_dlambda_pmf_survival(X, fitted):
    survival_before, tail = predict_survival_pmf(X, fitted)
    survival_after = np.c_[survival_before[:, 1:], tail]
    ratio = np.clip(survival_after / np.clip(survival_before, PMF_SURVIVAL_MIN_PROB, 1.0), PMF_SURVIVAL_MIN_PROB, 1.0)
    return -np.log(ratio)


def predict_dlambda_matrix(X, fitted):
    if fitted.get("shared_arm_discrete_hazard", False):
        return predict_dlambda_discrete(X, fitted)
    if fitted.get("discrete_hazard", False):
        return predict_dlambda_discrete(X, fitted)
    if fitted.get("weibull_ph", False):
        return predict_dlambda_weibull(X, fitted)
    if fitted.get("pmf_survival", False):
        return predict_dlambda_pmf_survival(X, fitted)
    raise ValueError("Fitted object does not expose a hazard-matrix prediction path.")


def plugin_cnb_g_hazard_matrix(dL0, dL1):
    dL0 = np.asarray(dL0, dtype=float)
    dL1 = np.asarray(dL1, dtype=float)
    H0_before = np.cumsum(dL0, axis=1) - dL0
    H1_before = np.cumsum(dL1, axis=1) - dL1
    kernel = np.exp(-H0_before - H1_before)
    return np.sum(kernel * (dL0 - dL1), axis=1)


def fitted_right_grid(fitted):
    return np.asarray(fitted["time_grid"][1:], dtype=float)


def align_dlambda_to_grid(source_times, source_dlambda, target_times):
    source_times = np.asarray(source_times, dtype=float)
    source_dlambda = np.asarray(source_dlambda, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    aligned = np.zeros((source_dlambda.shape[0], len(target_times)), dtype=float)
    target_index = {round(float(t), 12): j for j, t in enumerate(target_times)}
    for k, t in enumerate(source_times):
        j = target_index.get(round(float(t), 12))
        if j is not None:
            aligned[:, j] = source_dlambda[:, k]
    return aligned


def predict_dlambda_pair_on_union_grid(X, fitted0, fitted1, tau=TAU):
    t0 = fitted_right_grid(fitted0)
    t1 = fitted_right_grid(fitted1)
    keep0 = t0 <= tau + 1e-12
    keep1 = t1 <= tau + 1e-12
    jump_times = np.sort(np.unique(np.r_[t0[keep0], t1[keep1]]))
    dL0 = predict_dlambda_matrix(X, fitted0)[:, keep0]
    dL1 = predict_dlambda_matrix(X, fitted1)[:, keep1]
    return jump_times, align_dlambda_to_grid(t0[keep0], dL0, jump_times), align_dlambda_to_grid(t1[keep1], dL1, jump_times)


def plugin_cnb_g_fitted_pair(X, fitted0, fitted1, tau=TAU):
    _, dL0, dL1 = predict_dlambda_pair_on_union_grid(X, fitted0, fitted1, tau=tau)
    return plugin_cnb_g_hazard_matrix(dL0, dL1)


def true_survival_arm(X, arm, tau=TAU):
    scale, shape = (16.0, 1.35) if int(arm) == 0 else (18.0, 1.25)
    eta = base.eta_true(X, int(arm))
    return np.exp(-((float(tau) / scale) ** shape) * np.exp(np.clip(eta, -8.0, 8.0)))


def predict_survival_tau(X, fitted, tau=TAU):
    if fitted.get("pmf_survival", False):
        _, pred_tau = predict_survival_pmf(X, fitted)
        return pred_tau
    dL = predict_dlambda_matrix(X, fitted)
    right = fitted_right_grid(fitted)
    keep = right <= tau + 1e-12
    return np.exp(-np.sum(dL[:, keep], axis=1))


def survival_tau_diagnostics(X, fitted, arm, tau=TAU):
    pred_tau = predict_survival_tau(X, fitted, tau=tau)
    true_tau = true_survival_arm(X, arm, tau=tau)
    err = pred_tau - true_tau
    return {
        f"s{arm}_tau_mean": float(np.mean(pred_tau)),
        f"s{arm}_tau_true_mean": float(np.mean(true_tau)),
        f"s{arm}_tau_bias": float(np.mean(err)),
        f"s{arm}_tau_mae": float(np.mean(np.abs(err))),
    }


def vector_distribution(values, prefix):
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p10": float(np.quantile(values, 0.10)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
    }


def source_residual_diagnostics(d, weights, A, Delta):
    d = np.asarray(d, dtype=float)
    weights = np.asarray(weights, dtype=float)
    A = np.asarray(A)
    Delta = np.asarray(Delta)
    wd = weights * d

    def masked_mean(x, mask):
        if not np.any(mask):
            return np.nan
        return float(np.mean(x[mask]))

    wd_sum = float(np.sum(wd))
    q90 = np.quantile(weights, 0.90)
    q95 = np.quantile(weights, 0.95)
    q99 = np.quantile(weights, 0.99)
    return {
        "rd_source_mean": float(np.mean(wd)),
        "rd_source_sd": float(np.std(wd, ddof=1)),
        "d_source_arm0_mean": masked_mean(d, A == 0),
        "d_source_arm1_mean": masked_mean(d, A == 1),
        "rd_source_arm0_mean": masked_mean(wd, A == 0),
        "rd_source_arm1_mean": masked_mean(wd, A == 1),
        "d_source_event_mean": masked_mean(d, Delta == 1),
        "d_source_censor_mean": masked_mean(d, Delta == 0),
        "rd_source_event_mean": masked_mean(wd, Delta == 1),
        "rd_source_censor_mean": masked_mean(wd, Delta == 0),
        "d_source_w_top10_mean": masked_mean(d, weights >= q90),
        "d_source_w_top5_mean": masked_mean(d, weights >= q95),
        "d_source_w_top1_mean": masked_mean(d, weights >= q99),
        "rd_source_w_top10_mean": masked_mean(wd, weights >= q90),
        "rd_source_w_top5_mean": masked_mean(wd, weights >= q95),
        "rd_source_w_top1_mean": masked_mean(wd, weights >= q99),
        "rd_source_w_top10_share": float(np.sum(wd[weights >= q90]) / wd_sum) if abs(wd_sum) > 1e-12 else np.nan,
        "rd_source_w_top5_share": float(np.sum(wd[weights >= q95]) / wd_sum) if abs(wd_sum) > 1e-12 else np.nan,
        "rd_source_w_top1_share": float(np.sum(wd[weights >= q99]) / wd_sum) if abs(wd_sum) > 1e-12 else np.nan,
    }


def mean_or_nan(values):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def compute_pmf_outcome_weights(X_source, X_target=None, method=PMF_OUTCOME_WEIGHT, cap=PMF_WEIGHT_CAP):
    if method == "none" or X_target is None or len(X_target) == 0:
        return np.ones(X_source.shape[0], dtype=float)
    if method == "density_ratio":
        alpha, prior_ratio = base.fit_density_ratio_logistic(X_source, X_target)
        weights = base.predict_density_ratio_logistic(X_source, alpha, prior_ratio)
    elif method == "calibration":
        model = base.fit_calibration_weights(X_source, X_target)
        weights = base.predict_calibration_ratio(X_source, model)
    else:
        raise ValueError(f"Unknown PMF outcome weighting: {method}")
    return base.stabilize_ratio(weights, cap=cap)


def ratio_validation_split(y, seed):
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    val_idx = np.r_[
        idx0[: max(1, int(0.20 * len(idx0)))],
        idx1[: max(1, int(0.20 * len(idx1)))],
    ]
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.flatnonzero(train_mask)
    return train_idx, val_idx


def fit_neural_density_ratio(X_source, X_target, seed, epochs=250, lr=1e-3, weight_decay=1e-4, patience=25):
    set_torch_seed(seed)
    X_all = np.vstack([X_source, X_target])
    y = np.r_[np.zeros(X_source.shape[0]), np.ones(X_target.shape[0])].astype(np.float32)
    x_mean = X_all.mean(axis=0)
    x_scale = X_all.std(axis=0, ddof=1)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    Xs = (X_all - x_mean) / x_scale
    train_idx, val_idx = ratio_validation_split(y, seed)

    x_train = torch.as_tensor(Xs[train_idx], dtype=torch.float32, device=DEVICE)
    y_train = torch.as_tensor(y[train_idx], dtype=torch.float32, device=DEVICE)
    x_val = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = torch.as_tensor(y[val_idx], dtype=torch.float32, device=DEVICE)

    model = DensityRatioNet(X_source.shape[1]).float().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = np.inf
    best_train = np.inf
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(epochs):
        set_module_mode(model, True)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x_train), y_train)
        loss.backward()
        opt.step()
        set_module_mode(model, False)
        with torch.no_grad():
            train_loss = float(F.binary_cross_entropy_with_logits(model(x_train), y_train).cpu())
            val_loss = float(F.binary_cross_entropy_with_logits(model(x_val), y_val).cpu())
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    full_epochs = max(1, int(best_epoch))
    set_torch_seed(seed + 7919)
    x_all = torch.as_tensor(Xs, dtype=torch.float32, device=DEVICE)
    y_all = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    final_model = DensityRatioNet(X_source.shape[1]).float().to(DEVICE)
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(full_epochs):
        set_module_mode(final_model, True)
        final_opt.zero_grad()
        full_loss = F.binary_cross_entropy_with_logits(final_model(x_all), y_all)
        full_loss.backward()
        final_opt.step()
    set_module_mode(final_model, False)
    with torch.no_grad():
        logits_all = final_model(x_all)
        full_refit_loss = float(F.binary_cross_entropy_with_logits(logits_all, y_all).cpu())
        p_all = torch.sigmoid(logits_all).cpu().numpy()
    final_model = final_model.cpu()
    set_module_mode(final_model, False)
    return {
        "learner": "neural",
        "model": final_model,
        "center": x_mean,
        "scale": x_scale,
        "prior_ratio": float(X_source.shape[0] / X_target.shape[0]),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train),
        "best_val_loss": float(best_val),
        "full_refit_loss": float(full_refit_loss),
        "p_target_mean": float(np.mean(p_all)),
        "p_target_source_mean": float(np.mean(p_all[: X_source.shape[0]])),
        "p_target_target_mean": float(np.mean(p_all[X_source.shape[0] :])),
    }


def fit_ratio_nuisance(X_source, X_target, seed, learner=None):
    learner = RATIO_LEARNER if learner is None else learner
    if learner not in RATIO_LEARNER_CHOICES:
        raise ValueError(f"Unknown ratio learner: {learner}")
    if learner == "calibration":
        return {
            "learner": "calibration",
            "model": base.fit_calibration_weights(X_source, X_target, ridge=EIF_CALIBRATION_RIDGE),
            "diagnostics": [],
        }
    if learner == "logistic":
        alpha, prior_ratio = base.fit_density_ratio_logistic(X_source, X_target)
        return {
            "learner": "logistic",
            "alpha": alpha,
            "prior_ratio": prior_ratio,
            "diagnostics": [],
        }
    fitted = fit_neural_density_ratio(X_source, X_target, seed)
    return {"learner": "neural", "model": fitted, "diagnostics": [fitted]}


def predict_ratio(X, ratio_nuisance):
    learner = ratio_nuisance["learner"]
    if learner == "calibration":
        return base.predict_calibration_ratio(X, ratio_nuisance["model"])
    if learner == "logistic":
        return base.predict_density_ratio_logistic(X, ratio_nuisance["alpha"], ratio_nuisance["prior_ratio"])
    if learner == "neural":
        fitted = ratio_nuisance["model"]
        Xs = (X - fitted["center"]) / fitted["scale"]
        with torch.no_grad():
            logits = fitted["model"](torch.as_tensor(Xs, dtype=torch.float32))
            p_target = torch.sigmoid(logits).cpu().numpy()
        p_target = np.clip(p_target, 0.01, 0.99)
        return fitted["prior_ratio"] * p_target / (1.0 - p_target)
    raise ValueError(f"Unknown ratio learner: {learner}")


def configure_pmf_options(
    pmf_grid=None,
    pmf_intervals=None,
    pmf_loss_weight=None,
    pmf_outcome_weight=None,
    pmf_weight_cap=None,
    shared_hazard_event_weight=None,
    shared_hazard_event_weight_arm0=None,
    shared_hazard_event_weight_arm1=None,
    shared_hazard_epochs=None,
    eif_weight_cap=None,
    eif_calibration_ridge=None,
    ci_method=None,
    dgp_scenario=None,
    dgp_target_shift_scale=None,
    censor_learner=None,
    ratio_learner=None,
):
    global PMF_SURVIVAL_GRID, PMF_SURVIVAL_INTERVALS, PMF_SURVIVAL_LOSS_WEIGHT, PMF_OUTCOME_WEIGHT, PMF_WEIGHT_CAP
    global SHARED_HAZARD_EVENT_WEIGHT, SHARED_HAZARD_EVENT_WEIGHT_ARM0, SHARED_HAZARD_EVENT_WEIGHT_ARM1
    global SHARED_HAZARD_EPOCHS
    global EIF_WEIGHT_CAP, EIF_CALIBRATION_RIDGE
    global CI_METHOD
    global DGP_SCENARIO, DGP_TARGET_SHIFT_SCALE
    global CENSOR_LEARNER, RATIO_LEARNER
    if pmf_grid is not None:
        if pmf_grid not in PMF_GRID_CHOICES:
            raise ValueError(f"Unknown PMF grid: {pmf_grid}")
        PMF_SURVIVAL_GRID = pmf_grid
    if pmf_intervals is not None:
        PMF_SURVIVAL_INTERVALS = int(pmf_intervals)
    if pmf_loss_weight is not None:
        if pmf_loss_weight not in PMF_LOSS_WEIGHT_CHOICES:
            raise ValueError(f"Unknown PMF loss weight: {pmf_loss_weight}")
        PMF_SURVIVAL_LOSS_WEIGHT = pmf_loss_weight
    if pmf_outcome_weight is not None:
        if pmf_outcome_weight not in PMF_OUTCOME_WEIGHT_CHOICES:
            raise ValueError(f"Unknown PMF outcome weight: {pmf_outcome_weight}")
        PMF_OUTCOME_WEIGHT = pmf_outcome_weight
    if pmf_weight_cap is not None:
        PMF_WEIGHT_CAP = float(pmf_weight_cap)
    if shared_hazard_event_weight is not None:
        SHARED_HAZARD_EVENT_WEIGHT = float(shared_hazard_event_weight)
    if shared_hazard_event_weight_arm0 is not None:
        SHARED_HAZARD_EVENT_WEIGHT_ARM0 = float(shared_hazard_event_weight_arm0)
    if shared_hazard_event_weight_arm1 is not None:
        SHARED_HAZARD_EVENT_WEIGHT_ARM1 = float(shared_hazard_event_weight_arm1)
    if shared_hazard_epochs is not None:
        SHARED_HAZARD_EPOCHS = int(shared_hazard_epochs)
    if eif_weight_cap is not None:
        EIF_WEIGHT_CAP = float(eif_weight_cap)
    if eif_calibration_ridge is not None:
        EIF_CALIBRATION_RIDGE = float(eif_calibration_ridge)
    if ci_method is not None:
        if ci_method not in CI_METHOD_CHOICES:
            raise ValueError(f"Unknown CI method: {ci_method}")
        CI_METHOD = ci_method
    if dgp_scenario is not None:
        if dgp_scenario not in DGP_SCENARIO_CHOICES:
            raise ValueError(f"Unknown DGP scenario: {dgp_scenario}")
        DGP_SCENARIO = dgp_scenario
    if dgp_target_shift_scale is not None:
        DGP_TARGET_SHIFT_SCALE = float(dgp_target_shift_scale)
    if censor_learner is not None:
        if censor_learner not in CENSOR_LEARNER_CHOICES:
            raise ValueError(f"Unknown censoring learner: {censor_learner}")
        CENSOR_LEARNER = censor_learner
    if ratio_learner is not None:
        if ratio_learner not in RATIO_LEARNER_CHOICES:
            raise ValueError(f"Unknown ratio learner: {ratio_learner}")
        RATIO_LEARNER = ratio_learner


def breslow_from_eta(eta, U, Delta):
    exp_eta = np.exp(np.clip(eta, -30, 30))
    event_times = np.sort(np.unique(U[Delta == 1]))
    increments = np.empty(len(event_times))
    for k, t in enumerate(event_times):
        increments[k] = np.sum((U == t) & (Delta == 1)) / max(np.sum(exp_eta[U >= t]), 1e-12)
    return event_times, increments


def fit_event_nuisance(X, A, U, Delta, learner, seed, X_target_for_weight=None):
    if learner == "linear_cox":
        Phi = base.phi_linear(X)
        return {"learner": learner, "cox": base.fit_event_nuisance(Phi, A, U, Delta), "diagnostics": []}
    if learner == "spline_cox":
        Phi = phi_spline(X)
        return {"learner": learner, "cox": base.fit_event_nuisance(Phi, A, U, Delta), "diagnostics": []}
    direct_survival_learners = {"deepsurv", "neural_discrete_hazard", "neural_weibull_ph", "neural_pmf_survival"}
    if learner in direct_survival_learners:
        fits = {}
        pmf_weights = compute_pmf_outcome_weights(
            X,
            X_target=X_target_for_weight,
            method=PMF_OUTCOME_WEIGHT if learner in {"deepsurv", "neural_pmf_survival"} else "none",
            cap=PMF_WEIGHT_CAP,
        )
        if learner == "deepsurv":
            shared = fit_shared_arm_discrete_hazard(
                X,
                A,
                U,
                Delta,
                seed + 53,
                n_intervals=PMF_SURVIVAL_INTERVALS,
                grid=PMF_SURVIVAL_GRID,
                loss_weight=PMF_SURVIVAL_LOSS_WEIGHT,
                event_weight=SHARED_HAZARD_EVENT_WEIGHT,
                event_weight_arm0=SHARED_HAZARD_EVENT_WEIGHT_ARM0,
                event_weight_arm1=SHARED_HAZARD_EVENT_WEIGHT_ARM1,
                sample_weight=pmf_weights,
                outcome_weight_label=PMF_OUTCOME_WEIGHT,
                weight_cap=PMF_WEIGHT_CAP,
                epochs=SHARED_HAZARD_EPOCHS,
            )
            fits = {
                0: [{"neural": shared_arm_fit_for_prediction(shared, 0)}],
                1: [{"neural": shared_arm_fit_for_prediction(shared, 1)}],
            }
            diagnostics = [
                {
                    "arm": a,
                    "model": 0,
                    "best_epoch": shared["best_epoch"],
                    "best_train_loss": shared["best_train_loss"],
                    "best_val_loss": shared["best_val_loss"],
                    "alpha": np.nan,
                    "linear_val_loss": np.nan,
                    "validation_gain": np.nan,
                    "residual_projection_r2": np.nan,
                    "alpha_zero": np.nan,
                    "discrete_hazard_mean": shared.get(f"discrete_hazard_mean_arm{a}", np.nan),
                    "discrete_hazard_max": shared.get(f"discrete_hazard_max_arm{a}", np.nan),
                    "weibull_shape": np.nan,
                    "weibull_scale": np.nan,
                    "weibull_eta_sd": np.nan,
                    "pmf_grid": shared.get("pmf_grid", ""),
                    "pmf_loss_weight": shared.get("pmf_loss_weight", ""),
                    "shared_hazard_event_weight": shared.get("shared_hazard_event_weight", np.nan),
                    "shared_hazard_event_weight_arm0": shared.get("shared_hazard_event_weight_arm0", np.nan),
                    "shared_hazard_event_weight_arm1": shared.get("shared_hazard_event_weight_arm1", np.nan),
                    "pmf_outcome_weight": shared.get("pmf_outcome_weight", ""),
                    "pmf_weight_cap": shared.get("pmf_weight_cap", np.nan),
                    "pmf_weight_mean": shared.get("pmf_weight_mean", np.nan),
                    "pmf_weight_sd": shared.get("pmf_weight_sd", np.nan),
                    "pmf_weight_min": shared.get("pmf_weight_min", np.nan),
                    "pmf_weight_max": shared.get("pmf_weight_max", np.nan),
                    "pmf_weight_p90": shared.get("pmf_weight_p90", np.nan),
                    "pmf_requested_intervals": shared.get("pmf_requested_intervals", np.nan),
                    "pmf_intervals": shared.get("pmf_intervals", np.nan),
                    "pmf_tail_mean": np.nan,
                    "pmf_entropy_mean": np.nan,
                    "n_events": shared.get(f"n_events_arm{a}", shared["n_events"]),
                }
                for a in [0, 1]
            ]
            return {
                "learner": learner,
                "fits": fits,
                "diagnostics": diagnostics,
                "time_grid": shared["time_grid"],
            }
        for a in [0, 1]:
            idx = A == a
            if learner == "neural_discrete_hazard":
                neural = fit_discrete_hazard(X[idx], U[idx], Delta[idx], seed + 17 * (a + 1))
            elif learner == "neural_weibull_ph":
                neural = fit_neural_weibull_ph(X[idx], U[idx], Delta[idx], seed + 17 * (a + 1))
            else:
                # Temporarily use a direct survival-function network for "deepsurv":
                # the PMF head estimates P(T in interval k | X) and the tail mass,
                # then S(t | X) is obtained by summing remaining mass over the grid.
                neural = fit_neural_pmf_survival(
                    X[idx],
                    U[idx],
                    Delta[idx],
                    seed + 17 * (a + 1),
                    n_intervals=PMF_SURVIVAL_INTERVALS,
                    grid=PMF_SURVIVAL_GRID,
                    loss_weight=PMF_SURVIVAL_LOSS_WEIGHT,
                    sample_weight=pmf_weights[idx],
                    outcome_weight_label=PMF_OUTCOME_WEIGHT,
                    weight_cap=PMF_WEIGHT_CAP,
                )
            fits[a] = [{"neural": neural}]
        diagnostics = [
            {
                "arm": a,
                "model": 0,
                "best_epoch": fits[a][0]["neural"]["best_epoch"],
                "best_train_loss": fits[a][0]["neural"]["best_train_loss"],
                "best_val_loss": fits[a][0]["neural"]["best_val_loss"],
                "alpha": np.nan,
                "linear_val_loss": np.nan,
                "validation_gain": np.nan,
                "residual_projection_r2": np.nan,
                "alpha_zero": np.nan,
                "discrete_hazard_mean": fits[a][0]["neural"].get("discrete_hazard_mean", np.nan),
                "discrete_hazard_max": fits[a][0]["neural"].get("discrete_hazard_max", np.nan),
                "weibull_shape": fits[a][0]["neural"].get("weibull_shape", np.nan),
                "weibull_scale": fits[a][0]["neural"].get("weibull_scale", np.nan),
                "weibull_eta_sd": fits[a][0]["neural"].get("weibull_eta_sd", np.nan),
                "pmf_grid": fits[a][0]["neural"].get("pmf_grid", ""),
                "pmf_loss_weight": fits[a][0]["neural"].get("pmf_loss_weight", ""),
                "pmf_outcome_weight": fits[a][0]["neural"].get("pmf_outcome_weight", ""),
                "pmf_weight_cap": fits[a][0]["neural"].get("pmf_weight_cap", np.nan),
                "pmf_weight_mean": fits[a][0]["neural"].get("pmf_weight_mean", np.nan),
                "pmf_weight_sd": fits[a][0]["neural"].get("pmf_weight_sd", np.nan),
                "pmf_weight_min": fits[a][0]["neural"].get("pmf_weight_min", np.nan),
                "pmf_weight_max": fits[a][0]["neural"].get("pmf_weight_max", np.nan),
                "pmf_weight_p90": fits[a][0]["neural"].get("pmf_weight_p90", np.nan),
                "pmf_requested_intervals": fits[a][0]["neural"].get("pmf_requested_intervals", np.nan),
                "pmf_intervals": fits[a][0]["neural"].get("pmf_intervals", np.nan),
                "pmf_tail_mean": fits[a][0]["neural"].get("pmf_tail_mean", np.nan),
                "pmf_entropy_mean": fits[a][0]["neural"].get("pmf_entropy_mean", np.nan),
                "n_events": fits[a][0]["neural"]["n_events"],
            }
            for a in [0, 1]
        ]
        return {
            "learner": learner,
            "fits": fits,
            "diagnostics": diagnostics,
            "time_grid": fits[0][0]["neural"]["time_grid"],
        }
    if learner not in {"deepsurv_ensemble", "coxkan", "hybrid_deepsurv", "hybrid_deepsurv_gated"}:
        raise ValueError(f"Unknown learner: {learner}")
    n_models = ENSEMBLE_SIZE if learner == "deepsurv_ensemble" else 1
    neural_learner = "deepsurv" if learner == "deepsurv_ensemble" else learner
    linear_nuisance = base.fit_event_nuisance(base.phi_linear(X), A, U, Delta) if learner == "hybrid_deepsurv_gated" else None
    fits = {}
    for a in [0, 1]:
        idx = A == a
        arm_fits = []
        for m in range(n_models):
            model_seed = seed + 1009 * m + 17 * (a + 1)
            if neural_learner == "hybrid_deepsurv":
                neural = fit_hybrid_deepsurv(X[idx], U[idx], Delta[idx], model_seed)
            elif neural_learner == "hybrid_deepsurv_gated":
                neural = fit_hybrid_deepsurv_gated(X[idx], U[idx], Delta[idx], model_seed)
            else:
                neural = fit_neural_cox(X[idx], U[idx], Delta[idx], neural_learner, model_seed)
            eta_train = predict_eta_neural(X[idx], neural)
            t, dH = breslow_from_eta(eta_train, U[idx], Delta[idx])
            arm_fits.append({"neural": neural, "t": t, "dH": dH})
        fits[a] = arm_fits
    diagnostics = [
        {
            "arm": a,
            "model": m,
            "best_epoch": fits[a][m]["neural"]["best_epoch"],
            "best_train_loss": fits[a][m]["neural"]["best_train_loss"],
            "best_val_loss": fits[a][m]["neural"]["best_val_loss"],
            "alpha": fits[a][m]["neural"].get("alpha", np.nan),
            "linear_val_loss": fits[a][m]["neural"].get("linear_val_loss", np.nan),
            "validation_gain": fits[a][m]["neural"].get("validation_gain", np.nan),
            "residual_projection_r2": fits[a][m]["neural"].get("residual_projection_r2", np.nan),
            "alpha_zero": float(fits[a][m]["neural"].get("alpha", np.nan) == 0.0),
            "n_events": fits[a][m]["neural"]["n_events"],
        }
        for a in [0, 1]
        for m in range(n_models)
    ]
    return {"learner": learner, "fits": fits, "diagnostics": diagnostics, "linear_nuisance": linear_nuisance}


def evaluate_event_nuisance(X, nuisance):
    if nuisance["learner"] == "linear_cox":
        return base.evaluate_event_nuisance(base.phi_linear(X), nuisance["cox"])
    if nuisance["learner"] == "spline_cox":
        return base.evaluate_event_nuisance(phi_spline(X), nuisance["cox"])
    if nuisance["learner"] in {"deepsurv", "neural_discrete_hazard", "neural_weibull_ph", "neural_pmf_survival"}:
        out = np.zeros(X.shape[0])
        fits0, fits1 = nuisance["fits"][0], nuisance["fits"][1]
        for fit0, fit1 in zip(fits0, fits1):
            out += plugin_cnb_g_fitted_pair(X, fit0["neural"], fit1["neural"])
        return out / len(fits0)
    out = np.zeros(X.shape[0])
    fits0, fits1 = nuisance["fits"][0], nuisance["fits"][1]
    for fit0, fit1 in zip(fits0, fits1):
        eta0 = predict_eta_neural(X, fit0["neural"])
        eta1 = predict_eta_neural(X, fit1["neural"])
        out += plugin_cnb_g_eta(eta0, fit0["t"], fit0["dH"], eta1, fit1["t"], fit1["dH"])
    return out / len(fits0)


def plugin_cnb_g_eta(eta0, t0, dH0, eta1, t1, dH1, tau=TAU):
    jump_times = np.sort(np.unique(np.r_[t0[t0 <= tau], t1[t1 <= tau]]))
    exp0 = np.exp(np.clip(eta0, -30, 30))
    exp1 = np.exp(np.clip(eta1, -30, 30))
    H0 = np.zeros(len(eta0))
    H1 = np.zeros(len(eta1))
    g_hat = np.zeros(len(eta0))
    inc0 = {float(t): h for t, h in zip(t0, dH0) if t <= tau}
    inc1 = {float(t): h for t, h in zip(t1, dH1) if t <= tau}
    for t in jump_times:
        kernel = np.exp(-H0 - H1)
        dL0 = exp0 * inc0.get(float(t), 0.0)
        dL1 = exp1 * inc1.get(float(t), 0.0)
        g_hat += kernel * (dL0 - dL1)
        H0 += dL0
        H1 += dL1
    return g_hat


def cnb_eif_eta(X, A, U, Delta, nuisance, censor_nuisance, pi1, tau=TAU):
    if nuisance["learner"] in {"linear_cox", "spline_cox"}:
        censor_model = censor_nuisance.get("model") if isinstance(censor_nuisance, dict) else censor_nuisance
        Phi = base.phi_linear(X) if nuisance["learner"] == "linear_cox" else phi_spline(X)
        return base.cox_cnb_eif_K(
            X, Phi, A, U, Delta, nuisance["cox"], censor_model, pi1, tau=tau
        )
    if nuisance["learner"] in {"deepsurv", "neural_discrete_hazard", "neural_weibull_ph", "neural_pmf_survival"}:
        corrections = []
        time_grid = nuisance.get("time_grid", make_discrete_time_grid(tau=tau))
        for fit0, fit1 in zip(nuisance["fits"][0], nuisance["fits"][1]):
            corrections.append(
                cnb_eif_hazard_matrix(X, A, U, Delta, fit0, fit1, time_grid, censor_nuisance, pi1, tau=tau)
            )
        return np.mean(corrections, axis=0)
    corrections = []
    for fit0, fit1 in zip(nuisance["fits"][0], nuisance["fits"][1]):
        corrections.append(cnb_eif_eta_single(X, A, U, Delta, fit0, fit1, censor_nuisance, pi1, tau=tau))
    return np.mean(corrections, axis=0)


def _full_cnb_eif_from_dlambda_grid(X, A, U, Delta, jump_times, dL0_all, dL1_all, censor_nuisance, pi1, tau=TAU):
    """Shared full EIF core for hazard-matrix and DeepSurv-Cox nuisance fits."""
    eps = 1e-12
    n = X.shape[0]
    jump_times = np.asarray(jump_times, dtype=float)
    if len(jump_times) == 0:
        return np.zeros(n)

    A = np.asarray(A, dtype=int)
    U = np.asarray(U, dtype=float)
    Delta = np.asarray(Delta, dtype=int)
    pi1 = np.clip(np.asarray(pi1, dtype=float), 0.02, 0.98)
    pi0 = 1.0 - pi1
    K = censor_survival_matrix(X, A, censor_nuisance, jump_times)
    correction = np.zeros(n)

    def left_survival(dLambda):
        H_left = np.r_[0.0, np.cumsum(dLambda[:-1])]
        return np.exp(-np.clip(H_left, 0.0, 700.0))

    def left_survival_at(time, grid, dLambda):
        idx = np.searchsorted(grid, float(time), side="left")
        H_left = np.sum(dLambda[:idx]) if idx > 0 else 0.0
        return float(np.exp(-min(max(H_left, 0.0), 700.0)))

    def censor_survival_at(i, time):
        return float(censor_survival_matrix(
            X[i : i + 1],
            A[i : i + 1],
            censor_nuisance,
            np.array([float(time)]),
        )[0, 0])

    for i in range(n):
        ui = float(U[i])
        ai = int(A[i])
        di = int(Delta[i])
        upper = min(ui, tau)
        at_risk = jump_times <= upper + 1e-12

        dL0 = dL0_all[i]
        dL1 = dL1_all[i]
        S0_left = left_survival(dL0)
        S1_left = left_survival(dL1)
        K_grid = np.maximum(K[i], eps)
        K_inv = 1.0 / K_grid

        if ai == 1:
            dLA = dL1
            SA_left = S1_left
            piA = pi1[i]
        else:
            dLA = dL0
            SA_left = S0_left
            piA = pi0[i]
        G_A = 1.0 / piA

        denom_tilde = np.maximum(K_grid * SA_left, eps)
        tilde_increments = np.zeros_like(jump_times, dtype=float)
        tilde_increments[at_risk] = -dLA[at_risk] / denom_tilde[at_risk]
        M_tilde = np.cumsum(tilde_increments)

        event_in_window = (di == 1 and ui <= tau)
        if event_in_window:
            K_U = max(censor_survival_at(i, ui), eps)
            S_A_U_left = max(left_survival_at(ui, jump_times, dLA), eps)
            M_tilde[jump_times >= ui] += 1.0 / (K_U * S_A_U_left)
        else:
            K_U = eps

        common_kernel = S1_left * S0_left * M_tilde
        first10 = -G_A * np.sum(common_kernel * dL0)
        first01 = -G_A * np.sum(common_kernel * dL1)

        second10 = 0.0
        if ai == 0:
            mart10 = -np.sum(S1_left[at_risk] * dL0[at_risk] * K_inv[at_risk])
            if event_in_window:
                mart10 += left_survival_at(ui, jump_times, dL1) / K_U
            second10 = mart10 / pi0[i]

        second01 = 0.0
        if ai == 1:
            mart01 = -np.sum(S0_left[at_risk] * dL1[at_risk] * K_inv[at_risk])
            if event_in_window:
                mart01 += left_survival_at(ui, jump_times, dL0) / K_U
            second01 = mart01 / pi1[i]

        correction[i] = (first10 + second10) - (first01 + second01)
    return correction


def cnb_eif_hazard_matrix(X, A, U, Delta, fit0, fit1, time_grid, censor_nuisance, pi1, tau=TAU):
    jump_times, dL0_all, dL1_all = predict_dlambda_pair_on_union_grid(X, fit0["neural"], fit1["neural"], tau=tau)
    return _full_cnb_eif_from_dlambda_grid(
        X, A, U, Delta, jump_times, dL0_all, dL1_all, censor_nuisance, pi1, tau=tau
    )


def cnb_eif_eta_single(X, A, U, Delta, fit0, fit1, censor_nuisance, pi1, tau=TAU):
    t0, dH0 = fit0["t"], fit0["dH"]
    t1, dH1 = fit1["t"], fit1["dH"]
    jump_times = np.sort(np.unique(np.r_[t0[t0 <= tau], t1[t1 <= tau]]))
    inc0 = {float(t): float(h) for t, h in zip(t0, dH0) if t <= tau}
    inc1 = {float(t): float(h) for t, h in zip(t1, dH1) if t <= tau}
    base0 = np.array([inc0.get(float(t), 0.0) for t in jump_times])
    base1 = np.array([inc1.get(float(t), 0.0) for t in jump_times])
    exp0 = np.exp(np.clip(predict_eta_neural(X, fit0["neural"]), -30, 30))
    exp1 = np.exp(np.clip(predict_eta_neural(X, fit1["neural"]), -30, 30))
    dL0_all = exp0[:, None] * base0[None, :]
    dL1_all = exp1[:, None] * base1[None, :]
    return _full_cnb_eif_from_dlambda_grid(
        X, A, U, Delta, jump_times, dL0_all, dL1_all, censor_nuisance, pi1, tau=tau
    )


def make_folds(n, k, rng):
    perm = rng.permutation(n)
    folds = np.empty(n, dtype=int)
    for fold_id, idx in enumerate(np.array_split(perm, k)):
        folds[idx] = fold_id
    return folds


def one_replicate_on_data(learner, X_S, A, U, Delta, X_T, fold_S, fold_T, rep):
    g_S_cf = np.empty(N_S)
    g_T_cf = np.empty(N_T)
    d_S_cf = np.empty(N_S)
    r_cal_cf = np.empty(N_S)
    plugin_shift_vs_linear = np.full(N_T, np.nan)
    neural_diagnostics = []
    censor_diagnostics = []
    ratio_diagnostics = []
    survival_diagnostics = []
    direct_survival_learners = {"deepsurv", "neural_discrete_hazard", "neural_weibull_ph", "neural_pmf_survival"}
    for fold in range(N_FOLDS):
        s_eval = fold_S == fold
        t_eval = fold_T == fold
        s_train = ~s_eval
        t_train = ~t_eval
        event_nuisance = fit_event_nuisance(
            X_S[s_train],
            A[s_train],
            U[s_train],
            Delta[s_train],
            learner,
            SEED + rep * 101 + fold,
            X_target_for_weight=X_T[t_train],
        )
        censor_nuisance = fit_censoring_nuisance(
            X_S[s_train],
            A[s_train],
            U[s_train],
            Delta[s_train],
            SEED + rep * 101 + fold + 50021,
            learner="cox" if learner in {"linear_cox", "spline_cox"} else CENSOR_LEARNER,
        )
        alpha_pi = base.fit_logistic_regression(X_S[s_train], A[s_train])
        ratio_nuisance = fit_ratio_nuisance(
            X_S[s_train],
            X_T[t_train],
            SEED + rep * 101 + fold + 70001,
            learner=RATIO_LEARNER,
        )
        neural_diagnostics.extend(event_nuisance.get("diagnostics", []))
        censor_diagnostics.extend(censor_nuisance.get("diagnostics", []))
        ratio_diagnostics.extend(ratio_nuisance.get("diagnostics", []))
        g_S_cf[s_eval] = evaluate_event_nuisance(X_S[s_eval], event_nuisance)
        g_T_cf[t_eval] = evaluate_event_nuisance(X_T[t_eval], event_nuisance)
        pi_eval = base.predict_prob(X_S[s_eval], alpha_pi)
        d_S_cf[s_eval] = cnb_eif_eta(
            X_S[s_eval],
            A[s_eval],
            U[s_eval],
            Delta[s_eval],
            event_nuisance,
            censor_nuisance,
            pi_eval,
            tau=TAU,
        )
        r_cal_cf[s_eval] = predict_ratio(X_S[s_eval], ratio_nuisance)
        if learner in direct_survival_learners:
            for arm in [0, 1]:
                fitted = event_nuisance["fits"][arm][0]["neural"]
                survival_diagnostics.append(survival_tau_diagnostics(X_T[t_eval], fitted, arm, tau=TAU))
        if learner == "hybrid_deepsurv_gated":
            linear_g_T = base.evaluate_event_nuisance(base.phi_linear(X_T[t_eval]), event_nuisance["linear_nuisance"])
            plugin_shift_vs_linear[t_eval] = g_T_cf[t_eval] - linear_g_T
    r_cal_cf = base.stabilize_ratio(r_cal_cf, cap=EIF_WEIGHT_CAP)
    plugin_T = float(g_T_cf.mean())
    correction_calibration = float(np.mean(r_cal_cf * d_S_cf))
    one_step_calibration = float(plugin_T + correction_calibration)
    residual_diag = source_residual_diagnostics(d_S_cf, r_cal_cf, A, Delta)
    target_if = g_T_cf - plugin_T
    source_if_calibration = r_cal_cf * d_S_cf - correction_calibration
    target_if_sd = float(np.std(target_if, ddof=1))
    source_if_sd = float(np.std(source_if_calibration, ddof=1))
    target_se_component = float(np.sqrt(np.var(target_if, ddof=1) / N_T))
    source_se_component = float(np.sqrt(np.var(source_if_calibration, ddof=1) / N_S))
    se_eif_calibration = float(np.sqrt(
        np.var(target_if, ddof=1) / N_T + np.var(source_if_calibration, ddof=1) / N_S
    ))
    fold_est_calibration = []
    for fold in range(N_FOLDS):
        t_eval = fold_T == fold
        s_eval = fold_S == fold
        fold_est_calibration.append(g_T_cf[t_eval].mean() + np.mean(r_cal_cf[s_eval] * d_S_cf[s_eval]))
    se_fold_calibration = float(base.FOLD_SE_MULTIPLIER * np.std(fold_est_calibration, ddof=1))
    se_max_calibration = float(max(se_eif_calibration, se_fold_calibration))
    se_by_method = {
        "eif": se_eif_calibration,
        "fold": se_fold_calibration,
        "max": se_max_calibration,
    }
    se_calibration = float(se_by_method[CI_METHOD])
    ci_low_eif_calibration = float(one_step_calibration - NORMAL_CRITICAL_95 * se_eif_calibration)
    ci_high_eif_calibration = float(one_step_calibration + NORMAL_CRITICAL_95 * se_eif_calibration)
    ci_low_fold_calibration = float(one_step_calibration - NORMAL_CRITICAL_95 * se_fold_calibration)
    ci_high_fold_calibration = float(one_step_calibration + NORMAL_CRITICAL_95 * se_fold_calibration)
    ci_low_max_calibration = float(one_step_calibration - NORMAL_CRITICAL_95 * se_max_calibration)
    ci_high_max_calibration = float(one_step_calibration + NORMAL_CRITICAL_95 * se_max_calibration)
    ci_low_calibration = {
        "eif": ci_low_eif_calibration,
        "fold": ci_low_fold_calibration,
        "max": ci_low_max_calibration,
    }[CI_METHOD]
    ci_high_calibration = {
        "eif": ci_high_eif_calibration,
        "fold": ci_high_fold_calibration,
        "max": ci_high_max_calibration,
    }[CI_METHOD]
    balance_l2_calibration, balance_max_calibration = base.balance_diagnostics(X_S, X_T, r_cal_cf)
    g_source_dist = vector_distribution(g_S_cf, "source_g_hat")
    g_target_dist = vector_distribution(g_T_cf, "target_g_hat")
    arm0_surv = {k: mean_or_nan(d.get(k, np.nan) for d in survival_diagnostics) for k in ["s0_tau_mean", "s0_tau_true_mean", "s0_tau_bias", "s0_tau_mae"]}
    arm1_surv = {k: mean_or_nan(d.get(k, np.nan) for d in survival_diagnostics) for k in ["s1_tau_mean", "s1_tau_true_mean", "s1_tau_bias", "s1_tau_mae"]}
    return {
        "S_only": float(g_S_cf.mean()),
        "plugin_T": plugin_T,
        "one_step_calibration": one_step_calibration,
        "se_calibration": se_calibration,
        "se_eif_calibration": se_eif_calibration,
        "se_fold_calibration": se_fold_calibration,
        "se_max_calibration": se_max_calibration,
        "ci_method": CI_METHOD,
        "target_if_sd": target_if_sd,
        "source_if_sd": source_if_sd,
        "target_se_component": target_se_component,
        "source_se_component": source_se_component,
        "ci_low_calibration": ci_low_calibration,
        "ci_high_calibration": ci_high_calibration,
        "ci_low_eif_calibration": ci_low_eif_calibration,
        "ci_high_eif_calibration": ci_high_eif_calibration,
        "ci_low_fold_calibration": ci_low_fold_calibration,
        "ci_high_fold_calibration": ci_high_fold_calibration,
        "ci_low_max_calibration": ci_low_max_calibration,
        "ci_high_max_calibration": ci_high_max_calibration,
        "correction_calibration": correction_calibration,
        "w_calibration_mean": float(np.mean(r_cal_cf)),
        "w_calibration_p99": float(np.quantile(r_cal_cf, 0.99)),
        "w_calibration_max": float(np.max(r_cal_cf)),
        "ess_calibration": float(base.effective_sample_size(r_cal_cf)),
        "balance_l2_calibration": float(balance_l2_calibration),
        "balance_max_calibration": float(balance_max_calibration),
        "d_source_mean": float(np.mean(d_S_cf)),
        "d_source_sd": float(np.std(d_S_cf, ddof=1)),
        "event_rate": float(Delta.mean()),
        "censoring_rate": float(1.0 - Delta.mean()),
        **residual_diag,
        "censor_learner": "cox" if learner in {"linear_cox", "spline_cox"} else CENSOR_LEARNER,
        "ratio_learner": RATIO_LEARNER,
        "mean_ratio_best_epoch": mean_or_nan([
            d.get("best_epoch", np.nan)
            for d in ratio_diagnostics
        ]),
        "mean_ratio_val_loss": mean_or_nan([
            d.get("best_val_loss", np.nan)
            for d in ratio_diagnostics
        ]),
        "mean_ratio_p_source": mean_or_nan([
            d.get("p_target_source_mean", np.nan)
            for d in ratio_diagnostics
        ]),
        "mean_ratio_p_target": mean_or_nan([
            d.get("p_target_target_mean", np.nan)
            for d in ratio_diagnostics
        ]),
        "mean_censor_best_epoch": mean_or_nan([
            d.get("censor_best_epoch", np.nan)
            for d in censor_diagnostics
        ]),
        "mean_censor_val_loss": mean_or_nan([
            d.get("censor_best_val_loss", np.nan)
            for d in censor_diagnostics
        ]),
        "mean_censor_hazard_arm0": mean_or_nan([
            d.get("censor_hazard_mean_arm0", np.nan)
            for d in censor_diagnostics
        ]),
        "mean_censor_hazard_arm1": mean_or_nan([
            d.get("censor_hazard_mean_arm1", np.nan)
            for d in censor_diagnostics
        ]),
        "mean_censor_intervals": mean_or_nan([
            d.get("censor_intervals", np.nan)
            for d in censor_diagnostics
        ]),
        **g_source_dist,
        **g_target_dist,
        **arm0_surv,
        **arm1_surv,
        "mean_best_epoch": float(np.mean([
            d["best_epoch"]
            for d in neural_diagnostics
        ])) if learner not in {"linear_cox", "spline_cox"} and neural_diagnostics else np.nan,
        "mean_val_loss": float(np.mean([
            d["best_val_loss"]
            for d in neural_diagnostics
        ])) if learner not in {"linear_cox", "spline_cox"} and neural_diagnostics else np.nan,
        "mean_hybrid_alpha": float(np.nanmean([
            d["alpha"]
            for d in neural_diagnostics
        ])) if learner == "hybrid_deepsurv" else np.nan,
        "mean_hybrid_linear_val_loss": float(np.nanmean([
            d["linear_val_loss"]
            for d in neural_diagnostics
        ])) if learner == "hybrid_deepsurv" else np.nan,
        "mean_gated_alpha": float(np.nanmean([
            d["alpha"]
            for d in neural_diagnostics
        ])) if learner == "hybrid_deepsurv_gated" else np.nan,
        "alpha_zero_rate": float(np.nanmean([
            d["alpha_zero"]
            for d in neural_diagnostics
        ])) if learner == "hybrid_deepsurv_gated" else np.nan,
        "mean_validation_gain": float(np.nanmean([
            d["validation_gain"]
            for d in neural_diagnostics
        ])) if learner == "hybrid_deepsurv_gated" else np.nan,
        "mean_residual_projection_r2": float(np.nanmean([
            d["residual_projection_r2"]
            for d in neural_diagnostics
        ])) if learner == "hybrid_deepsurv_gated" else np.nan,
        "mean_plugin_shift_vs_linear": float(np.nanmean(plugin_shift_vs_linear)) if learner == "hybrid_deepsurv_gated" else np.nan,
        "mean_discrete_hazard_mean": float(np.nanmean([
            d.get("discrete_hazard_mean", np.nan)
            for d in neural_diagnostics
        ])) if learner in {"deepsurv", "neural_discrete_hazard"} else np.nan,
        "mean_discrete_hazard_max": float(np.nanmean([
            d.get("discrete_hazard_max", np.nan)
            for d in neural_diagnostics
        ])) if learner in {"deepsurv", "neural_discrete_hazard"} else np.nan,
        "mean_weibull_shape": float(np.nanmean([
            d.get("weibull_shape", np.nan)
            for d in neural_diagnostics
        ])) if learner == "neural_weibull_ph" else np.nan,
        "mean_weibull_scale": float(np.nanmean([
            d.get("weibull_scale", np.nan)
            for d in neural_diagnostics
        ])) if learner == "neural_weibull_ph" else np.nan,
        "mean_weibull_eta_sd": float(np.nanmean([
            d.get("weibull_eta_sd", np.nan)
            for d in neural_diagnostics
        ])) if learner == "neural_weibull_ph" else np.nan,
        "mean_pmf_tail": mean_or_nan([
            d.get("pmf_tail_mean", np.nan)
            for d in neural_diagnostics
        ]) if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
        "mean_pmf_tail_arm0": mean_or_nan(
            d.get("pmf_tail_mean", np.nan)
            for d in neural_diagnostics
            if d.get("arm") == 0
        ) if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
        "mean_pmf_tail_arm1": mean_or_nan(
            d.get("pmf_tail_mean", np.nan)
            for d in neural_diagnostics
            if d.get("arm") == 1
        ) if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
        "mean_pmf_entropy": mean_or_nan([
            d.get("pmf_entropy_mean", np.nan)
            for d in neural_diagnostics
        ]) if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
        "mean_pmf_intervals": mean_or_nan([
            d.get("pmf_intervals", np.nan)
            for d in neural_diagnostics
        ]) if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
        "mean_pmf_weight_sd": mean_or_nan([
            d.get("pmf_weight_sd", np.nan)
            for d in neural_diagnostics
        ]) if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
        "mean_pmf_weight_max": mean_or_nan([
            d.get("pmf_weight_max", np.nan)
            for d in neural_diagnostics
        ]) if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
        "mean_pmf_weight_p90": mean_or_nan([
            d.get("pmf_weight_p90", np.nan)
            for d in neural_diagnostics
        ]) if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
        "pmf_grid": PMF_SURVIVAL_GRID if learner in {"deepsurv", "neural_pmf_survival"} else "",
        "pmf_loss_weight": PMF_SURVIVAL_LOSS_WEIGHT if learner in {"deepsurv", "neural_pmf_survival"} else "",
        "shared_hazard_event_weight": SHARED_HAZARD_EVENT_WEIGHT if learner == "deepsurv" else np.nan,
        "shared_hazard_event_weight_arm0": (
            SHARED_HAZARD_EVENT_WEIGHT if SHARED_HAZARD_EVENT_WEIGHT_ARM0 is None else SHARED_HAZARD_EVENT_WEIGHT_ARM0
        ) if learner == "deepsurv" else np.nan,
        "shared_hazard_event_weight_arm1": (
            SHARED_HAZARD_EVENT_WEIGHT if SHARED_HAZARD_EVENT_WEIGHT_ARM1 is None else SHARED_HAZARD_EVENT_WEIGHT_ARM1
        ) if learner == "deepsurv" else np.nan,
        "pmf_outcome_weight": PMF_OUTCOME_WEIGHT if learner in {"deepsurv", "neural_pmf_survival"} else "",
        "pmf_weight_cap": PMF_WEIGHT_CAP if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
        "pmf_requested_intervals": PMF_SURVIVAL_INTERVALS if learner in {"deepsurv", "neural_pmf_survival"} else np.nan,
    }


def one_replicate(learner, rng, rep):
    X_S, A, U, Delta = generate_source_data_dgp(N_S, rng)
    X_T = sample_x_dgp(N_T, current_target_mu(), rng)
    fold_S = make_folds(N_S, N_FOLDS, rng)
    fold_T = make_folds(N_T, N_FOLDS, rng)
    return one_replicate_on_data(learner, X_S, A, U, Delta, X_T, fold_S, fold_T, rep)


def summarize(raw):
    rows = []
    for (learner, estimator), sub in raw.groupby(["learner", "estimator"]):
        vals = sub["estimate"].to_numpy()
        truth = float(sub["truth"].iloc[0])
        empirical_sd = vals.std(ddof=1) if len(vals) > 1 else np.nan
        row = {
            "learner": learner,
            "estimator": estimator,
            "truth": truth,
            "mean": vals.mean(),
            "bias": vals.mean() - truth,
            "sd": empirical_sd,
            "rmse": np.sqrt(np.mean((vals - truth) ** 2)),
        }
        row["abs_bias"] = abs(row["bias"])
        if "ci_low" in sub.columns and sub["ci_low"].notna().any():
            covered = (sub["ci_low"].to_numpy() <= truth) & (truth <= sub["ci_high"].to_numpy())
            row["mean_se"] = float(np.nanmean(sub["se"].to_numpy()))
            row["coverage"] = float(np.nanmean(covered.astype(float)))
            if np.isfinite(empirical_sd) and empirical_sd > 0:
                row["se_ratio_to_empirical"] = row["mean_se"] / empirical_sd
        for method in ["eif", "fold", "max"]:
            se_col = f"se_{method}"
            low_col = f"ci_low_{method}"
            high_col = f"ci_high_{method}"
            if se_col in sub.columns and sub[se_col].notna().any():
                row[f"mean_se_{method}"] = float(np.nanmean(sub[se_col].to_numpy()))
                if np.isfinite(empirical_sd) and empirical_sd > 0:
                    row[f"se_{method}_ratio_to_empirical"] = row[f"mean_se_{method}"] / empirical_sd
            if low_col in sub.columns and sub[low_col].notna().any():
                covered = (sub[low_col].to_numpy() <= truth) & (truth <= sub[high_col].to_numpy())
                row[f"coverage_{method}"] = float(np.nanmean(covered.astype(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def run_simulation(
    n_rep=N_REP,
    learners=("linear_cox", "deepsurv"),
    pmf_grid=None,
    pmf_intervals=None,
    pmf_loss_weight=None,
    pmf_outcome_weight=None,
    pmf_weight_cap=None,
    shared_hazard_event_weight=None,
    shared_hazard_event_weight_arm0=None,
    shared_hazard_event_weight_arm1=None,
    shared_hazard_epochs=None,
    eif_weight_cap=None,
    eif_calibration_ridge=None,
    ci_method=None,
    dgp_scenario=None,
    dgp_target_shift_scale=None,
    censor_learner=None,
    ratio_learner=None,
):
    configure_pmf_options(
        pmf_grid=pmf_grid,
        pmf_intervals=pmf_intervals,
        pmf_loss_weight=pmf_loss_weight,
        pmf_outcome_weight=pmf_outcome_weight,
        pmf_weight_cap=pmf_weight_cap,
        shared_hazard_event_weight=shared_hazard_event_weight,
        shared_hazard_event_weight_arm0=shared_hazard_event_weight_arm0,
        shared_hazard_event_weight_arm1=shared_hazard_event_weight_arm1,
        shared_hazard_epochs=shared_hazard_epochs,
        eif_weight_cap=eif_weight_cap,
        eif_calibration_ridge=eif_calibration_ridge,
        ci_method=ci_method,
        dgp_scenario=dgp_scenario,
        dgp_target_shift_scale=dgp_target_shift_scale,
        censor_learner=censor_learner,
        ratio_learner=ratio_learner,
    )
    rng = np.random.default_rng(SEED)
    x_big = sample_x_dgp(120000, current_target_mu(), np.random.default_rng(SEED + 900))
    truth = float(true_cnb_g_dgp(x_big).mean())
    raw_rows = []
    diag_rows = []
    for rep in range(n_rep):
        X_S, A, U, Delta = generate_source_data_dgp(N_S, rng)
        X_T = sample_x_dgp(N_T, current_target_mu(), rng)
        fold_S = make_folds(N_S, N_FOLDS, rng)
        fold_T = make_folds(N_T, N_FOLDS, rng)
        for learner in learners:
            out = one_replicate_on_data(learner, X_S, A, U, Delta, X_T, fold_S, fold_T, rep)
            for estimator in ["S_only", "plugin_T", "one_step_calibration"]:
                raw_rows.append({
                    "rep": rep,
                    "learner": learner,
                    "estimator": estimator,
                    "estimate": out[estimator],
                    "truth": truth,
                    "dgp_scenario": DGP_SCENARIO,
                    "dgp_target_shift_scale": DGP_TARGET_SHIFT_SCALE,
                    "se": out["se_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "ci_low": out["ci_low_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "ci_high": out["ci_high_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "ci_method": out["ci_method"] if estimator == "one_step_calibration" else "",
                    "se_eif": out["se_eif_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "se_fold": out["se_fold_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "se_max": out["se_max_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "ci_low_eif": out["ci_low_eif_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "ci_high_eif": out["ci_high_eif_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "ci_low_fold": out["ci_low_fold_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "ci_high_fold": out["ci_high_fold_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "ci_low_max": out["ci_low_max_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "ci_high_max": out["ci_high_max_calibration"] if estimator == "one_step_calibration" else np.nan,
                    "pmf_grid": PMF_SURVIVAL_GRID,
                    "pmf_intervals": PMF_SURVIVAL_INTERVALS,
                    "pmf_loss_weight": PMF_SURVIVAL_LOSS_WEIGHT,
                    "shared_hazard_event_weight": SHARED_HAZARD_EVENT_WEIGHT,
                    "shared_hazard_event_weight_arm0": SHARED_HAZARD_EVENT_WEIGHT if SHARED_HAZARD_EVENT_WEIGHT_ARM0 is None else SHARED_HAZARD_EVENT_WEIGHT_ARM0,
                    "shared_hazard_event_weight_arm1": SHARED_HAZARD_EVENT_WEIGHT if SHARED_HAZARD_EVENT_WEIGHT_ARM1 is None else SHARED_HAZARD_EVENT_WEIGHT_ARM1,
                    "shared_hazard_epochs": SHARED_HAZARD_EPOCHS,
                    "pmf_outcome_weight": PMF_OUTCOME_WEIGHT,
                    "pmf_weight_cap": PMF_WEIGHT_CAP,
                    "eif_weight_cap": EIF_WEIGHT_CAP,
                    "eif_calibration_ridge": EIF_CALIBRATION_RIDGE,
                    "censor_learner": out["censor_learner"],
                    "ratio_learner": out["ratio_learner"],
                })
            diag_rows.append({
                "rep": rep,
                "learner": learner,
                "dgp_scenario": DGP_SCENARIO,
                "dgp_target_shift_scale": DGP_TARGET_SHIFT_SCALE,
                "censor_learner": out["censor_learner"],
                "ratio_learner": out["ratio_learner"],
                "pmf_grid": out["pmf_grid"],
                "pmf_loss_weight": out["pmf_loss_weight"],
                "shared_hazard_event_weight": out["shared_hazard_event_weight"],
                "shared_hazard_event_weight_arm0": out["shared_hazard_event_weight_arm0"],
                "shared_hazard_event_weight_arm1": out["shared_hazard_event_weight_arm1"],
                "shared_hazard_epochs": SHARED_HAZARD_EPOCHS,
                "pmf_outcome_weight": out["pmf_outcome_weight"],
                "pmf_weight_cap": out["pmf_weight_cap"],
                "eif_weight_cap": EIF_WEIGHT_CAP,
                "eif_calibration_ridge": EIF_CALIBRATION_RIDGE,
                "ci_method": out["ci_method"],
                "pmf_requested_intervals": out["pmf_requested_intervals"],
                "correction_calibration": out["correction_calibration"],
                "se_calibration": out["se_calibration"],
                "se_eif_calibration": out["se_eif_calibration"],
                "se_fold_calibration": out["se_fold_calibration"],
                "se_max_calibration": out["se_max_calibration"],
                "target_if_sd": out["target_if_sd"],
                "source_if_sd": out["source_if_sd"],
                "target_se_component": out["target_se_component"],
                "source_se_component": out["source_se_component"],
                "w_calibration_mean": out["w_calibration_mean"],
                "w_calibration_p99": out["w_calibration_p99"],
                "w_calibration_max": out["w_calibration_max"],
                "ess_calibration": out["ess_calibration"],
                "balance_l2_calibration": out["balance_l2_calibration"],
                "balance_max_calibration": out["balance_max_calibration"],
                "d_source_mean": out["d_source_mean"],
                "d_source_sd": out["d_source_sd"],
                "rd_source_mean": out["rd_source_mean"],
                "rd_source_sd": out["rd_source_sd"],
                "d_source_arm0_mean": out["d_source_arm0_mean"],
                "d_source_arm1_mean": out["d_source_arm1_mean"],
                "rd_source_arm0_mean": out["rd_source_arm0_mean"],
                "rd_source_arm1_mean": out["rd_source_arm1_mean"],
                "d_source_event_mean": out["d_source_event_mean"],
                "d_source_censor_mean": out["d_source_censor_mean"],
                "rd_source_event_mean": out["rd_source_event_mean"],
                "rd_source_censor_mean": out["rd_source_censor_mean"],
                "d_source_w_top10_mean": out["d_source_w_top10_mean"],
                "d_source_w_top5_mean": out["d_source_w_top5_mean"],
                "d_source_w_top1_mean": out["d_source_w_top1_mean"],
                "rd_source_w_top10_mean": out["rd_source_w_top10_mean"],
                "rd_source_w_top5_mean": out["rd_source_w_top5_mean"],
                "rd_source_w_top1_mean": out["rd_source_w_top1_mean"],
                "rd_source_w_top10_share": out["rd_source_w_top10_share"],
                "rd_source_w_top5_share": out["rd_source_w_top5_share"],
                "rd_source_w_top1_share": out["rd_source_w_top1_share"],
                "event_rate": out["event_rate"],
                "censoring_rate": out["censoring_rate"],
                "mean_ratio_best_epoch": out["mean_ratio_best_epoch"],
                "mean_ratio_val_loss": out["mean_ratio_val_loss"],
                "mean_ratio_p_source": out["mean_ratio_p_source"],
                "mean_ratio_p_target": out["mean_ratio_p_target"],
                "mean_censor_best_epoch": out["mean_censor_best_epoch"],
                "mean_censor_val_loss": out["mean_censor_val_loss"],
                "mean_censor_hazard_arm0": out["mean_censor_hazard_arm0"],
                "mean_censor_hazard_arm1": out["mean_censor_hazard_arm1"],
                "mean_censor_intervals": out["mean_censor_intervals"],
                "source_g_hat_mean": out["source_g_hat_mean"],
                "source_g_hat_p10": out["source_g_hat_p10"],
                "source_g_hat_p50": out["source_g_hat_p50"],
                "source_g_hat_p90": out["source_g_hat_p90"],
                "target_g_hat_mean": out["target_g_hat_mean"],
                "target_g_hat_p10": out["target_g_hat_p10"],
                "target_g_hat_p50": out["target_g_hat_p50"],
                "target_g_hat_p90": out["target_g_hat_p90"],
                "mean_best_epoch": out["mean_best_epoch"],
                "mean_val_loss": out["mean_val_loss"],
                "mean_hybrid_alpha": out["mean_hybrid_alpha"],
                "mean_hybrid_linear_val_loss": out["mean_hybrid_linear_val_loss"],
                "mean_gated_alpha": out["mean_gated_alpha"],
                "alpha_zero_rate": out["alpha_zero_rate"],
                "mean_validation_gain": out["mean_validation_gain"],
                "mean_residual_projection_r2": out["mean_residual_projection_r2"],
                "mean_plugin_shift_vs_linear": out["mean_plugin_shift_vs_linear"],
                "mean_discrete_hazard_mean": out["mean_discrete_hazard_mean"],
                "mean_discrete_hazard_max": out["mean_discrete_hazard_max"],
                "mean_weibull_shape": out["mean_weibull_shape"],
                "mean_weibull_scale": out["mean_weibull_scale"],
                "mean_weibull_eta_sd": out["mean_weibull_eta_sd"],
                "mean_pmf_tail": out["mean_pmf_tail"],
                "mean_pmf_tail_arm0": out["mean_pmf_tail_arm0"],
                "mean_pmf_tail_arm1": out["mean_pmf_tail_arm1"],
                "mean_pmf_entropy": out["mean_pmf_entropy"],
                "mean_pmf_intervals": out["mean_pmf_intervals"],
                "mean_pmf_weight_sd": out["mean_pmf_weight_sd"],
                "mean_pmf_weight_max": out["mean_pmf_weight_max"],
                "mean_pmf_weight_p90": out["mean_pmf_weight_p90"],
                "s0_tau_mean": out["s0_tau_mean"],
                "s0_tau_true_mean": out["s0_tau_true_mean"],
                "s0_tau_bias": out["s0_tau_bias"],
                "s0_tau_mae": out["s0_tau_mae"],
                "s1_tau_mean": out["s1_tau_mean"],
                "s1_tau_true_mean": out["s1_tau_true_mean"],
                "s1_tau_bias": out["s1_tau_bias"],
                "s1_tau_mae": out["s1_tau_mae"],
            })
            print(
                f"rep {rep + 1:03d}/{n_rep} {learner}: "
                f"S_only={out['S_only']:.4f}, plugin_T={out['plugin_T']:.4f}, "
                f"one_step={out['one_step_calibration']:.4f}",
                flush=True,
            )
    raw = pd.DataFrame(raw_rows)
    diagnostics = pd.DataFrame(diag_rows)
    summary = summarize(raw)
    diagnostics_summary = diagnostics.groupby("learner").mean(numeric_only=True).reset_index()
    config_cols = [c for c in ["dgp_scenario", "censor_learner", "ratio_learner", "ci_method", "pmf_grid", "pmf_loss_weight", "pmf_outcome_weight"] if c in diagnostics.columns]
    if config_cols:
        configs = diagnostics.groupby("learner")[config_cols].first().reset_index()
        diagnostics_summary = diagnostics_summary.merge(configs, on="learner", how="left")
    one_step_summary = summary[summary["estimator"] == "one_step_calibration"].set_index("learner")
    for learner in diagnostics_summary["learner"]:
        if learner not in one_step_summary.index:
            continue
        row = one_step_summary.loc[learner]
        mask = diagnostics_summary["learner"] == learner
        for col in ["se_ratio_to_empirical", "se_eif_ratio_to_empirical", "se_fold_ratio_to_empirical", "se_max_ratio_to_empirical"]:
            if col in row.index:
                diagnostics_summary.loc[mask, col] = row[col]
    return raw, summary, diagnostics_summary


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text, execution_count=None, outputs=None):
    return {"cell_type": "code", "execution_count": execution_count, "metadata": {}, "outputs": outputs or [], "source": text.splitlines(True)}


def dataframe_output(df):
    return [{"name": "stdout", "output_type": "stream", "text": (df.to_string(index=False) + "\n").splitlines(True)}]


def build_notebook(summary, diagnostics, raw_path=RAW_PATH, summary_path=SUMMARY_PATH, diagnostics_path=DIAGNOSTICS_PATH):
    summary_cols = [
        c for c in [
            "learner",
            "estimator",
            "truth",
            "mean",
            "bias",
            "sd",
            "rmse",
            "mean_se",
            "coverage",
            "mean_se_eif",
            "coverage_eif",
            "mean_se_fold",
            "coverage_fold",
            "mean_se_max",
            "coverage_max",
            "se_ratio_to_empirical",
        ]
        if c in summary.columns
    ]
    summary_view = summary[summary_cols].round(4)
    diag_cols = [
        c for c in [
            "learner",
            "dgp_scenario",
            "dgp_target_shift_scale",
            "censor_learner",
            "ratio_learner",
            "pmf_grid",
            "pmf_loss_weight",
            "shared_hazard_event_weight",
            "shared_hazard_event_weight_arm0",
            "shared_hazard_event_weight_arm1",
            "shared_hazard_epochs",
            "pmf_outcome_weight",
            "pmf_weight_cap",
            "eif_weight_cap",
            "eif_calibration_ridge",
            "ci_method",
            "pmf_requested_intervals",
            "correction_calibration",
            "se_calibration",
            "se_eif_calibration",
            "se_fold_calibration",
            "se_max_calibration",
            "target_if_sd",
            "source_if_sd",
            "target_se_component",
            "source_se_component",
            "se_ratio_to_empirical",
            "se_eif_ratio_to_empirical",
            "se_fold_ratio_to_empirical",
            "se_max_ratio_to_empirical",
            "w_calibration_mean",
            "w_calibration_p99",
            "w_calibration_max",
            "ess_calibration",
            "balance_l2_calibration",
            "balance_max_calibration",
            "d_source_mean",
            "d_source_sd",
            "rd_source_mean",
            "rd_source_sd",
            "d_source_arm0_mean",
            "d_source_arm1_mean",
            "rd_source_arm0_mean",
            "rd_source_arm1_mean",
            "d_source_event_mean",
            "d_source_censor_mean",
            "rd_source_event_mean",
            "rd_source_censor_mean",
            "d_source_w_top10_mean",
            "d_source_w_top5_mean",
            "d_source_w_top1_mean",
            "rd_source_w_top10_mean",
            "rd_source_w_top5_mean",
            "rd_source_w_top1_mean",
            "rd_source_w_top10_share",
            "rd_source_w_top5_share",
            "rd_source_w_top1_share",
            "mean_pmf_intervals",
            "mean_pmf_weight_sd",
            "mean_pmf_weight_max",
            "mean_pmf_weight_p90",
            "event_rate",
            "censoring_rate",
            "mean_ratio_best_epoch",
            "mean_ratio_val_loss",
            "mean_ratio_p_source",
            "mean_ratio_p_target",
            "mean_censor_best_epoch",
            "mean_censor_val_loss",
            "mean_censor_hazard_arm0",
            "mean_censor_hazard_arm1",
            "mean_censor_intervals",
            "mean_best_epoch",
            "mean_val_loss",
            "mean_pmf_tail",
            "mean_pmf_tail_arm0",
            "mean_pmf_tail_arm1",
            "mean_pmf_entropy",
            "s0_tau_mean",
            "s0_tau_bias",
            "s0_tau_mae",
            "s1_tau_mean",
            "s1_tau_bias",
            "s1_tau_mae",
            "source_g_hat_mean",
            "target_g_hat_mean",
            "target_g_hat_p10",
            "target_g_hat_p50",
            "target_g_hat_p90",
        ]
        if c in diagnostics.columns
    ]
    diag_view = diagnostics[diag_cols].round(4)
    cells = [
        md("# Direct-survival DeepSurv one-step pilot\n\nThis notebook compares the existing linear Cox nuisance with a neural direct survival-function nuisance. It reports the plain target plug-in, the calibration-weighted EIF one-step estimator, and multiple CI/SE diagnostics."),
        md("For `deepsurv`, the neural nuisance directly estimates treatment-specific survival through a shared trunk with arm-specific discrete hazard heads. The one-step estimator uses cross-fitted censoring, treatment propensity, and source-to-target ratio nuisances. The EIF point estimator is unchanged; `ci_method` only controls which SE is used for the primary confidence interval."),
        code(f"import pandas as pd\nfrom pathlib import Path\n\nraw = pd.read_csv(Path(r\"{raw_path}\"))\nsummary = pd.read_csv(Path(r\"{summary_path}\"))\ndiagnostics = pd.read_csv(Path(r\"{diagnostics_path}\"))", 1),
        code("summary[[c for c in [\"learner\", \"estimator\", \"truth\", \"mean\", \"bias\", \"sd\", \"rmse\", \"mean_se\", \"coverage\", \"mean_se_eif\", \"coverage_eif\", \"mean_se_fold\", \"coverage_fold\", \"mean_se_max\", \"coverage_max\", \"se_ratio_to_empirical\"] if c in summary.columns]].round(4)", 2, dataframe_output(summary_view)),
        code("diagnostics[[c for c in [\"learner\", \"dgp_scenario\", \"dgp_target_shift_scale\", \"censor_learner\", \"ratio_learner\", \"ci_method\", \"pmf_grid\", \"pmf_loss_weight\", \"shared_hazard_event_weight\", \"shared_hazard_event_weight_arm0\", \"shared_hazard_event_weight_arm1\", \"shared_hazard_epochs\", \"pmf_outcome_weight\", \"pmf_weight_cap\", \"eif_weight_cap\", \"eif_calibration_ridge\", \"pmf_requested_intervals\", \"correction_calibration\", \"se_calibration\", \"se_eif_calibration\", \"se_fold_calibration\", \"se_max_calibration\", \"target_if_sd\", \"source_if_sd\", \"target_se_component\", \"source_se_component\", \"se_ratio_to_empirical\", \"se_eif_ratio_to_empirical\", \"se_fold_ratio_to_empirical\", \"se_max_ratio_to_empirical\", \"w_calibration_mean\", \"w_calibration_p99\", \"w_calibration_max\", \"ess_calibration\", \"balance_l2_calibration\", \"balance_max_calibration\", \"d_source_mean\", \"d_source_sd\", \"rd_source_mean\", \"rd_source_sd\", \"d_source_arm0_mean\", \"d_source_arm1_mean\", \"rd_source_arm0_mean\", \"rd_source_arm1_mean\", \"d_source_event_mean\", \"d_source_censor_mean\", \"rd_source_event_mean\", \"rd_source_censor_mean\", \"d_source_w_top10_mean\", \"d_source_w_top5_mean\", \"d_source_w_top1_mean\", \"rd_source_w_top10_mean\", \"rd_source_w_top5_mean\", \"rd_source_w_top1_mean\", \"rd_source_w_top10_share\", \"rd_source_w_top5_share\", \"rd_source_w_top1_share\", \"mean_pmf_intervals\", \"mean_pmf_weight_sd\", \"mean_pmf_weight_max\", \"mean_pmf_weight_p90\", \"event_rate\", \"censoring_rate\", \"mean_ratio_best_epoch\", \"mean_ratio_val_loss\", \"mean_ratio_p_source\", \"mean_ratio_p_target\", \"mean_censor_best_epoch\", \"mean_censor_val_loss\", \"mean_censor_hazard_arm0\", \"mean_censor_hazard_arm1\", \"mean_censor_intervals\", \"mean_best_epoch\", \"mean_val_loss\", \"mean_pmf_tail\", \"mean_pmf_tail_arm0\", \"mean_pmf_tail_arm1\", \"mean_pmf_entropy\", \"s0_tau_mean\", \"s0_tau_bias\", \"s0_tau_mae\", \"s1_tau_mean\", \"s1_tau_bias\", \"s1_tau_mae\", \"source_g_hat_mean\", \"target_g_hat_mean\", \"target_g_hat_p10\", \"target_g_hat_p50\", \"target_g_hat_p90\"] if c in diagnostics.columns]].round(4)", 3, dataframe_output(diag_view)),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rep", type=int, default=N_REP)
    parser.add_argument("--learners", nargs="+", default=["linear_cox", "deepsurv"])
    parser.add_argument("--output-label", default="", help="Optional suffix inserted before output file extensions.")
    parser.add_argument("--pmf-grid", choices=PMF_GRID_CHOICES, default=PMF_SURVIVAL_GRID)
    parser.add_argument("--pmf-intervals", type=int, default=PMF_SURVIVAL_INTERVALS)
    parser.add_argument("--pmf-loss-weight", choices=PMF_LOSS_WEIGHT_CHOICES, default=PMF_SURVIVAL_LOSS_WEIGHT)
    parser.add_argument("--pmf-outcome-weight", choices=PMF_OUTCOME_WEIGHT_CHOICES, default=PMF_OUTCOME_WEIGHT)
    parser.add_argument("--pmf-weight-cap", type=float, default=PMF_WEIGHT_CAP)
    parser.add_argument("--shared-hazard-event-weight", type=float, default=SHARED_HAZARD_EVENT_WEIGHT)
    parser.add_argument("--shared-hazard-event-weight-arm0", type=float, default=None)
    parser.add_argument("--shared-hazard-event-weight-arm1", type=float, default=None)
    parser.add_argument("--shared-hazard-epochs", type=int, default=SHARED_HAZARD_EPOCHS)
    parser.add_argument("--eif-weight-cap", type=float, default=EIF_WEIGHT_CAP)
    parser.add_argument("--eif-calibration-ridge", type=float, default=EIF_CALIBRATION_RIDGE)
    parser.add_argument("--ci-method", choices=CI_METHOD_CHOICES, default=CI_METHOD)
    parser.add_argument("--dgp-scenario", choices=DGP_SCENARIO_CHOICES, default=DGP_SCENARIO)
    parser.add_argument("--dgp-target-shift-scale", type=float, default=DGP_TARGET_SHIFT_SCALE)
    parser.add_argument("--censor-learner", choices=CENSOR_LEARNER_CHOICES, default=CENSOR_LEARNER)
    parser.add_argument("--ratio-learner", choices=RATIO_LEARNER_CHOICES, default=RATIO_LEARNER)
    args = parser.parse_args()
    raw, summary, diagnostics = run_simulation(
        n_rep=args.n_rep,
        learners=tuple(args.learners),
        pmf_grid=args.pmf_grid,
        pmf_intervals=args.pmf_intervals,
        pmf_loss_weight=args.pmf_loss_weight,
        pmf_outcome_weight=args.pmf_outcome_weight,
        pmf_weight_cap=args.pmf_weight_cap,
        shared_hazard_event_weight=args.shared_hazard_event_weight,
        shared_hazard_event_weight_arm0=args.shared_hazard_event_weight_arm0,
        shared_hazard_event_weight_arm1=args.shared_hazard_event_weight_arm1,
        shared_hazard_epochs=args.shared_hazard_epochs,
        eif_weight_cap=args.eif_weight_cap,
        eif_calibration_ridge=args.eif_calibration_ridge,
        ci_method=args.ci_method,
        dgp_scenario=args.dgp_scenario,
        dgp_target_shift_scale=args.dgp_target_shift_scale,
        censor_learner=args.censor_learner,
        ratio_learner=args.ratio_learner,
    )
    raw_path = safe_to_csv(raw, label_path(RAW_PATH, args.output_label))
    summary_path = safe_to_csv(summary, label_path(SUMMARY_PATH, args.output_label))
    diagnostics_path = safe_to_csv(diagnostics, label_path(DIAGNOSTICS_PATH, args.output_label))
    label_path(NB_PATH, args.output_label).write_text(json.dumps(build_notebook(summary, diagnostics, raw_path, summary_path, diagnostics_path), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWrote raw: {raw_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote diagnostics: {diagnostics_path}")
    print("\nSummary:")
    print(summary.round(4).to_string(index=False))
    print("\nDiagnostics:")
    print(diagnostics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
