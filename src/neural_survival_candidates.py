"""Small PyTorch survival-function candidates for the CNB screening study."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_PROB = 1e-6


def _seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _split(delta, seed):
    if torch.is_tensor(delta):
        delta = delta.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    e = np.flatnonzero(delta == 1)
    c = np.flatnonzero(delta == 0)
    rng.shuffle(e); rng.shuffle(c)
    vi = np.r_[e[:max(1, int(.2 * len(e)))], c[:max(1, int(.2 * len(c)))]]
    tr = np.ones(len(delta), dtype=bool); tr[vi] = False
    ti = np.flatnonzero(tr)
    if delta[ti].sum() < 5 or delta[vi].sum() < 2:
        ti = np.arange(len(delta)); vi = np.arange(len(delta))
    return ti, vi


class _MLP(nn.Module):
    def __init__(self, p, out, hidden_dims=None):
        super().__init__()
        hidden_dims = tuple(hidden_dims or (64, 32, 16))
        layers = []
        width = int(p)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, int(hidden)), nn.ReLU(), nn.Dropout(.05)])
            width = int(hidden)
        layers.append(nn.Linear(width, int(out)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _fit_model(model, loss_fn, x, u, d, seed, epochs, lr, patience,
               weight_decay=1e-4, grad_clip=5.0):
    _seed(seed)
    ti, vi = _split(d, seed)
    xt, ut, dt = x[ti], u[ti], d[ti]
    xv, uv, dv = x[vi], u[vi], d[vi]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = np.inf; state = None; stale = 0; best_epoch = 0
    for ep in range(int(epochs)):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(xt), ut, dt)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        model.eval()
        with torch.no_grad(): val = float(loss_fn(model(xv), uv, dv).cpu())
        if val + 1e-5 < best:
            best = val; best_epoch = ep + 1; stale = 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(patience): break
    if state is not None: model.load_state_dict(state)
    model.eval()
    return model, best_epoch, best


def _prepare(X, U, D, seed):
    center = X.mean(0); scale = np.maximum(X.std(0, ddof=1), 1e-6)
    xs = (X - center) / scale
    # Keep the standardized NumPy covariates and the PyTorch network on the
    # same dtype; the simulation covariates are otherwise float64 by default.
    ti = torch.as_tensor(xs, dtype=torch.float32, device=DEVICE)
    ut = torch.as_tensor(U, dtype=torch.float32, device=DEVICE)
    dt = torch.as_tensor(D.astype(float), dtype=torch.float32, device=DEVICE)
    return ti, ut, dt, center, scale


def _grid(tau, k):
    return np.linspace(0.0, float(tau), int(k) + 1)


def _hazard_loss(logits, u, d, right):
    logh = F.logsigmoid(logits); log1 = F.logsigmoid(-logits)
    bins = torch.searchsorted(right, torch.clamp(u, max=right[-1]), right=False)
    bins = torch.clamp(bins, 0, logits.shape[1] - 1)
    idx = torch.arange(logits.shape[1], device=logits.device)[None, :]
    before = idx < bins[:, None]
    through = idx <= bins[:, None]
    event_term = logh.gather(1, bins[:, None]).squeeze(1)
    contribution = torch.where(d > .5, -log1.masked_fill(~before, 0.0).sum(1) - event_term,
                               -log1.masked_fill(~through, 0.0).sum(1))
    return contribution.mean()


def fit_hazard_network(X, U, D, seed, tau=12.0, n_intervals=48, epochs=20, lr=2e-3, patience=8):
    x, u, d, center, scale = _prepare(X, U, D, seed)
    right = torch.as_tensor(_grid(tau, n_intervals)[1:], dtype=torch.float32, device=DEVICE)
    model = _MLP(X.shape[1], n_intervals).float().to(DEVICE)
    model, be, vl = _fit_model(model, lambda z, a, e: _hazard_loss(z, a, e, right), x, u, d, seed, epochs, lr, patience)
    return {"kind": "hazard_net", "model": model.cpu(), "center": center, "scale": scale,
            "tau": tau, "n_intervals": n_intervals, "best_epoch": be, "best_val_loss": vl}


def predict_hazard_network(X, fitted):
    xs = (X - fitted["center"]) / fitted["scale"]
    with torch.no_grad():
        z = fitted["model"](torch.as_tensor(xs, dtype=torch.float32))
        h = torch.sigmoid(z).cpu().numpy()
    return -np.log(np.clip(1.0 - h, MIN_PROB, 1.0))


def _pmf_loss(logits, u, d, right):
    lp = F.log_softmax(logits, dim=1); p = torch.exp(lp)
    bins = torch.searchsorted(right, torch.clamp(u, max=right[-1]), right=False)
    bins = torch.clamp(bins, 0, logits.shape[1] - 2)
    ii = torch.arange(len(u), device=logits.device)
    event_term = -lp[ii, bins]
    suffix = torch.flip(torch.cumsum(torch.flip(p, dims=[1]), dim=1), dims=[1])
    censor_term = -torch.log(torch.clamp(suffix[ii, bins + 1], min=MIN_PROB))
    return torch.where(d > .5, event_term, censor_term).mean()


def fit_mtlr(X, U, D, seed, tau=12.0, n_intervals=48, epochs=20, lr=2e-3,
             patience=8, time_grid=None, hidden_dims=None,
             weight_decay=1e-4, grad_clip=5.0, ensemble_seeds=1):
    x, u, d, center, scale = _prepare(X, U, D, seed)
    if time_grid is None:
        time_grid = _grid(tau, n_intervals)
    time_grid = np.asarray(time_grid, dtype=float)
    right = torch.as_tensor(time_grid[1:], dtype=torch.float32, device=DEVICE)
    n_ensemble = max(1, int(ensemble_seeds))
    members = []
    for member_id in range(n_ensemble):
        member_seed = int(seed) + 100003 * member_id
        _seed(member_seed)
        model = _MLP(X.shape[1], n_intervals + 1, hidden_dims=hidden_dims).float().to(DEVICE)
        model, be, vl = _fit_model(
            model, lambda z, a, e: _pmf_loss(z, a, e, right),
            x, u, d, member_seed, epochs, lr, patience,
            weight_decay=weight_decay, grad_clip=grad_clip,
        )
        members.append({"model": model.cpu(), "best_epoch": be, "best_val_loss": vl})
    if n_ensemble == 1:
        return {"kind": "mtlr", "model": members[0]["model"], "center": center, "scale": scale,
                "tau": tau, "n_intervals": n_intervals, "time_grid": time_grid,
                "best_epoch": members[0]["best_epoch"], "best_val_loss": members[0]["best_val_loss"]}
    return {"kind": "mtlr_ensemble", "models": members, "center": center, "scale": scale,
            "tau": tau, "n_intervals": n_intervals, "time_grid": time_grid,
            "best_epoch": float(np.mean([m["best_epoch"] for m in members])),
            "best_val_loss": float(np.mean([m["best_val_loss"] for m in members])),
            "ensemble_size": n_ensemble}


def _predict_mtlr_single(X, fitted, model):
    xs = (X - fitted["center"]) / fitted["scale"]
    with torch.no_grad():
        p = F.softmax(model(torch.as_tensor(xs, dtype=torch.float32)), dim=1).cpu().numpy()
    ev, tail = np.clip(p[:, :-1], MIN_PROB, 1.0), np.clip(p[:, -1], MIN_PROB, 1.0)
    sb = np.cumsum(ev[:, ::-1], axis=1)[:, ::-1] + tail[:, None]
    sa = np.c_[sb[:, 1:], tail]
    return -np.log(np.clip(sa / np.clip(sb, MIN_PROB, 1.0), MIN_PROB, 1.0))


def predict_mtlr(X, fitted):
    if fitted["kind"] == "mtlr":
        return _predict_mtlr_single(X, fitted, fitted["model"])
    if fitted["kind"] != "mtlr_ensemble":
        raise ValueError(fitted["kind"])
    hazards = [_predict_mtlr_single(X, fitted, member["model"]) for member in fitted["models"]]
    survival = []
    for hazard in hazards:
        survival.append(np.c_[np.ones(len(X)), np.exp(-np.cumsum(hazard, axis=1))])
    mean_survival = np.mean(survival, axis=0)
    return -np.log(np.clip(
        mean_survival[:, 1:] / np.maximum(mean_survival[:, :-1], MIN_PROB),
        MIN_PROB, 1.0,
    ))


def _dsm_loss(out, u, d, tau):
    m = out.shape[1] // 3
    logits, log_scale, log_shape = out[:, :m], out[:, m:2*m], out[:, 2*m:]
    pi = F.softmax(logits, dim=1)
    scale = F.softplus(log_scale) + .05
    shape = F.softplus(log_shape) + .2
    t = torch.clamp(u[:, None], min=1e-3, max=float(tau))
    z = (t / scale) ** shape
    log_s = torch.logsumexp(torch.log(torch.clamp(pi, MIN_PROB)) - z, dim=1)
    log_f = torch.logsumexp(torch.log(torch.clamp(pi, MIN_PROB)) + torch.log(shape) - torch.log(scale) + (shape - 1) * (torch.log(t) - torch.log(scale)) - z, dim=1)
    return torch.where(d > .5, -log_f, -log_s).mean()


def fit_dsm(X, U, D, seed, tau=12.0, n_intervals=48, mixtures=3, epochs=20, lr=1e-3, patience=8):
    x, u, d, center, scale = _prepare(X, U, D, seed)
    model = _MLP(X.shape[1], 3 * mixtures).float().to(DEVICE)
    model, be, vl = _fit_model(model, lambda z, a, e: _dsm_loss(z, a, e, tau), x, u, d, seed, epochs, lr, patience)
    return {"kind": "dsm", "model": model.cpu(), "center": center, "scale": scale,
            "tau": tau, "n_intervals": n_intervals, "mixtures": mixtures,
            "best_epoch": be, "best_val_loss": vl}


def predict_dsm(X, fitted):
    xs = (X - fitted["center"]) / fitted["scale"]
    m = fitted["mixtures"]
    with torch.no_grad(): out = fitted["model"](torch.as_tensor(xs, dtype=torch.float32)).cpu().numpy()
    logits, ls, lshape = out[:, :m], out[:, m:2*m], out[:, 2*m:]
    pi = np.exp(logits - logits.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    scale = np.log1p(np.exp(ls)) + .05; shape = np.log1p(np.exp(lshape)) + .2
    edges = _grid(fitted["tau"], fitted["n_intervals"])
    def surv(t):
        z = (float(t) / scale) ** shape
        return np.sum(pi * np.exp(-z), axis=1)
    before = np.column_stack([surv(t) for t in edges[:-1]])
    after = np.column_stack([surv(t) for t in edges[1:]])
    return -np.log(np.clip(after / np.clip(before, MIN_PROB, 1.0), MIN_PROB, 1.0))


def fit_candidate(X, U, D, kind, seed, **kwargs):
    if kind == "hazard_net": return fit_hazard_network(X, U, D, seed, **kwargs)
    if kind == "mtlr": return fit_mtlr(X, U, D, seed, **kwargs)
    if kind == "dsm": return fit_dsm(X, U, D, seed, **kwargs)
    raise ValueError(kind)


def predict_dlambda(X, fitted):
    if fitted["kind"] == "hazard_net": return predict_hazard_network(X, fitted)
    if fitted["kind"] in ("mtlr", "mtlr_ensemble"): return predict_mtlr(X, fitted)
    if fitted["kind"] == "dsm": return predict_dsm(X, fitted)
    raise ValueError(fitted["kind"])
