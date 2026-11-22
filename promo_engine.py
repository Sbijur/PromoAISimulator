"""
PromoAI engine: synthetic retailer data -> LightGBM forecast -> piecewise-linear (PWL) demand
-> promotion-calendar MILP (HiGHS) -> independent evaluation, greedy heuristic, reality simulation.

Re-implementation of the PromoAI system in Llenas et al., "PepsiCo Deploys AI-Driven Pricing and
Promotion Optimization at Scale" (INFORMS J. Applied Analytics; arXiv:2606.17941), section 2.1 and
Appendix A. All data are synthetic.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import highspy
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

# --------------------------------------------------------------------------- defaults
DEFAULT_PPG = pd.DataFrame([
    # ppg,              brand,         category, segment, pepsico_flag, base_units, shelf_price, sell_in, unit_cost, elasticity
    ("PEP_COLA_2L",     "Pepsi",       "CSD",    "2L",    True,  900, 2.49, 1.70, 0.95, 2.6),
    ("PEP_MAX_2L",      "Pepsi Max",   "CSD",    "2L",    True,  600, 2.49, 1.70, 0.95, 2.9),
    ("7UP_2L",          "7UP",         "CSD",    "2L",    True,  350, 2.29, 1.55, 0.90, 3.1),
    ("COMP_COLA_2L",    "Rival Cola",  "CSD",    "2L",    False, 1000, 2.39, 1.65, 1.00, 2.5),
    ("PEP_CAN_6PK",     "Pepsi",       "CSD",    "6pk",   True,  700, 4.99, 3.40, 1.90, 2.2),
    ("MIRINDA_6PK",     "Mirinda",     "CSD",    "6pk",   True,  300, 4.49, 3.05, 1.75, 2.8),
    ("COMP_CAN_6PK",    "Rival Cola",  "CSD",    "6pk",   False, 800, 4.79, 3.30, 1.95, 2.3),
    ("LAYS_150G",       "Lay's",       "Snacks", "150g",  True,  800, 2.99, 2.00, 1.05, 2.4),
    ("COMP_CHIPS_150G", "Rival Chips", "Snacks", "150g",  False, 700, 2.79, 1.90, 1.05, 2.2),
], columns=["ppg", "brand", "category", "segment", "pepsico_flag", "base_units",
            "shelf_price", "sell_in", "unit_cost", "elasticity"])

DEFAULT_OPTIONS = pd.DataFrame([
    ("NONE",  "None",     0.00, 0),
    ("TPR10", "Price",    0.10, 0),
    ("TPR20", "Price",    0.20, 0),
    ("TPR30", "Price",    0.30, 1),
    ("MB25",  "Multibuy", 0.25, 1),
], columns=["option", "mechanic_type", "depth", "placement"])

HOLIDAYS = {6: "Super Bowl", 14: "Easter", 27: "Independence Day", 47: "Thanksgiving", 51: "Christmas", 52: "New Year"}

DEFAULT_RIVAL_PLAN = {"COMP_COLA_2L": {4: "TPR20", 5: "TPR20"}, "COMP_CAN_6PK": {8: "MB25", 9: "MB25"}}

DEFAULT_RULES = {
    "Objective": {"w_pepsico": 0.7, "w_pep_margin": 0.5, "w_ret_margin": 0.5},
    "MarginFloor": {"pepsico": 0.40, "retailer": 0.28},
    "MaxPromoTotal": 30,
    "MaxPromoPepsiCo": 24,
    "MinMarketShare": 0.55,
    "WeeklyMaxPromo": [{"data": {"mechanic_type": ["Price"]}, "week_max_promo": 3}],
    "MinMaxPromoDuration": {"data": [{"pepsico_flag": True}], "max_promo_block": 2, "min_promo_block": 1},
    "MinSpacing": {"data": [{"pepsico_flag": True}], "weeks": 2},
    "MaxPromoPerPPG": {"data": [{"pepsico_flag": True}], "max_weeks": 4},
    "Solver": {"mip_gap": 0.01, "time_limit": 60, "formulation": "hull"},
}

FEATURES = ["ppg", "brand", "category", "segment", "year", "week_of_year", "month", "quarter", "holiday",
            "mechanic_type", "depth", "base_price", "actual_price", "placement", "discount_pressure"]
CAT_COLS = ["ppg", "brand", "category", "segment", "mechanic_type"]


def with_changes(cfg: dict, **changes) -> dict:
    c = json.loads(json.dumps(cfg))
    for k, v in changes.items():
        c[k] = v
    return c


# --------------------------------------------------------------------------- data
def calendar_frame(start, n_weeks: int) -> pd.DataFrame:
    d = pd.date_range(start, periods=n_weeks, freq="W-MON")
    iso = d.isocalendar()
    out = pd.DataFrame({"date": d, "year": iso.year.values.astype(int), "week_of_year": iso.week.values.astype(int)})
    out["month"] = out.date.dt.month
    out["quarter"] = out.date.dt.quarter
    out["holiday"] = out.week_of_year.isin(HOLIDAYS).astype(int)
    return out


def segment_pressure(depth_by_ppg_week: pd.DataFrame, ppg: pd.DataFrame) -> pd.DataFrame:
    """beta[p,w] = mean discount of the OTHER PPGs in p's segment (paper eq. 5)."""
    out = depth_by_ppg_week.copy() * 0.0
    for _, members in ppg.groupby("segment").ppg:
        members = list(members)
        for p in members:
            others = [q for q in members if q != p]
            out[p] = depth_by_ppg_week[others].mean(axis=1) if others else 0.0
    return out


def true_units_mean(df: pd.DataFrame, ppg: pd.DataFrame, elast_mult: float = 1.0) -> np.ndarray:
    """Hidden data-generating process (the forecast never sees this formula)."""
    a = ppg.set_index("ppg").loc[df.ppg.values]
    csd = a.category.values == "CSD"
    season = 1 + np.where(csd, 0.12, 0.04) * np.sin(2 * np.pi * (df.week_of_year.values - 10) / 52)
    hol = 1 + df.holiday.values * np.where(csd, 0.25, 0.35)
    trend = 1.03 ** (df.year.values - 2024)
    mech = np.where(df.mechanic_type.values == "Multibuy", 1.15, 1.0)
    lift = np.exp(elast_mult * a.elasticity.values * df.depth.values * mech)
    place = 1 + 0.12 * df.placement.values
    press = np.exp(-(0.8 + 3 * df.depth.values) * df.discount_pressure.values)
    return a.base_units.values * season * hol * trend * lift * place * press


def simulate_history(ppg: pd.DataFrame, options: pd.DataFrame, n_weeks: int = 156, start="2024-01-01",
                     seed: int = 7, noise: float = 0.08, promo_prob: float = 0.13, elast_mult: float = 1.0,
                     inject_errors: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cal = calendar_frame(start, n_weeks)
    opt_by = {}
    for p in ppg.ppg:                       # random promo blocks (1-3 weeks), gap >= 2 weeks
        o, w = np.zeros(n_weeks, int), 0
        while w < n_weeks:
            if rng.random() < promo_prob:
                L, r = rng.integers(1, 4), rng.integers(1, len(options))
                o[w:w + L] = r
                w += L + 2
            else:
                w += 1
        opt_by[p] = o
    depth = pd.DataFrame({p: options.depth.values[o] for p, o in opt_by.items()})
    press = segment_pressure(depth, ppg)
    rows = []
    for p in ppg.ppg:
        r = options.iloc[opt_by[p]].reset_index(drop=True)
        d = cal.copy()
        d["ppg"] = p
        d = pd.concat([d, r], axis=1)
        d["discount_pressure"] = press[p].values
        rows.append(d)
    h = pd.concat(rows, ignore_index=True).merge(ppg[["ppg", "brand", "category", "segment", "pepsico_flag"]], on="ppg")
    h["base_price"] = ppg.set_index("ppg").loc[h.ppg, "shelf_price"].values
    h["actual_price"] = h.base_price * (1 - h.depth)
    h["units"] = true_units_mean(h, ppg, elast_mult) * rng.lognormal(0, noise, len(h))
    if inject_errors:                       # data problems described in paper 2.1.1
        miss = rng.random(len(h)) < 0.02
        stale = (h.depth > 0) & (rng.random(len(h)) < 0.04)
        h.loc[stale, "actual_price"] = h.loc[stale, "base_price"]
        spike = rng.random(len(h)) < 0.003
        h.loc[spike, "units"] *= 20
        h = h.loc[~miss].reset_index(drop=True)
    return h


def clean_history(h: pd.DataFrame):
    h = h.copy()
    report = {}
    full = h.groupby("ppg").date.agg(["min", "max", "count"])
    full["expected"] = (full["max"] - full["min"]).dt.days // 7 + 1
    report["Missing PPG-weeks (flagged)"] = int((full.expected - full["count"]).sum())
    bad = (h.depth > 0) & np.isclose(h.actual_price, h.base_price)
    report["Stale promotion prices corrected"] = int(bad.sum())
    h.loc[bad, "actual_price"] = h.loc[bad, "base_price"] * (1 - h.loc[bad, "depth"])
    med = h.groupby(["ppg", "option"]).units.transform("median")
    spike = h.units > 4 * med
    report["Implausible volume rows dropped"] = int(spike.sum())
    h = h.loc[~spike].reset_index(drop=True)
    report["Rows kept for training"] = len(h)
    return h, report


# --------------------------------------------------------------------------- forecast
def _cats(ppg, options):
    return {c: sorted(set(ppg[c]) if c in ppg else set(options[c])) for c in CAT_COLS}


def to_X(df: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c, cs in cats.items():
        X[c] = pd.Categorical(X[c], categories=cs)
    return X


def mape(y, yhat) -> float:
    y = np.asarray(y, float)
    return float(np.mean(np.abs(y - yhat) / np.maximum(y, 1e-9)))


@dataclass
class Forecast:
    model: object
    cats: dict
    params: dict
    best_iter: int
    val_mape: float
    test_mape: float
    legacy_mape: float
    test: pd.DataFrame
    importance: pd.Series
    trials: pd.DataFrame
    split: dict

    def predict(self, df):
        return np.maximum(self.model.predict(to_X(df, self.cats)), 0.0)


def train_forecast(hist: pd.DataFrame, ppg, options, n_trials: int = 15, seed: int = 7,
                   valid_weeks: int = 13, test_weeks: int = 13) -> Forecast:
    """Global LightGBM, MAPE objective, randomised search on a temporal hold-out, early stopping 50."""
    cats = _cats(ppg, options)
    dates = np.sort(hist.date.unique())
    test_start, valid_start = dates[-test_weeks], dates[-test_weeks - valid_weeks]
    train = hist[hist.date < valid_start]
    valid = hist[(hist.date >= valid_start) & (hist.date < test_start)]
    test = hist[hist.date >= test_start].copy()
    rng = np.random.default_rng(seed)

    def sample():
        # paper Table 4 ranges, rescaled: MAPE gradients are ~1/|y| so penalties are searched on a log scale
        return dict(objective="mape", metric="mape", verbosity=-1, seed=seed,
                    num_leaves=int(rng.integers(8, 64)), learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
                    min_split_gain=float(10 ** rng.uniform(-8, -5)), max_depth=int(rng.integers(4, 13)),
                    reg_alpha=float(10 ** rng.uniform(-7, -3)), reg_lambda=float(10 ** rng.uniform(-7, -3)),
                    min_child_samples=int(rng.integers(5, 30)))

    import warnings
    warnings.filterwarnings("ignore", message="Overriding the parameters")
    dtr = lgb.Dataset(to_X(train, cats), train.units.values, free_raw_data=False, params={"feature_pre_filter": False})
    dva = lgb.Dataset(to_X(valid, cats), valid.units.values, reference=dtr)
    trials = []
    for _ in range(n_trials):
        prm = sample()
        m = lgb.train(prm, dtr, num_boost_round=600, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        trials.append({"val_mape": m.best_score["valid_0"]["mape"], "trees": m.best_iteration, **prm})
    tr = pd.DataFrame(trials).sort_values("val_mape").reset_index(drop=True)
    best = tr.iloc[0].to_dict()
    params = {k: best[k] for k in ["objective", "metric", "verbosity", "seed", "num_leaves", "learning_rate",
                                   "min_split_gain", "max_depth", "reg_alpha", "reg_lambda", "min_child_samples"]}
    params = {k: (int(v) if isinstance(v, (np.integer,)) or k in ("num_leaves", "max_depth", "min_child_samples", "seed", "verbosity") else v)
              for k, v in params.items()}
    trval = pd.concat([train, valid])
    model = lgb.train(params, lgb.Dataset(to_X(trval, cats), trval.units.values), num_boost_round=max(int(best["trees"]), 50))
    test["forecast"] = np.maximum(model.predict(to_X(test, cats)), 0)
    legacy = trval.groupby(["ppg", "option"]).units.mean()
    test["legacy"] = legacy.reindex(pd.MultiIndex.from_frame(test[["ppg", "option"]])).values
    test["legacy"] = test["legacy"].fillna(pd.Series(trval.groupby("ppg").units.mean().reindex(test.ppg).values, index=test.index))
    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    return Forecast(model, cats, params, int(best["trees"]), float(best["val_mape"]),
                    mape(test.units, test.forecast), mape(test.units, test.legacy), test, imp / imp.sum(), tr,
                    {"train": len(train), "valid": len(valid), "test": len(test)})


def scenario_rows(plan, ppg, options, p_idx, w_idx, r_idx, beta) -> pd.DataFrame:
    d = plan.iloc[w_idx].reset_index(drop=True).copy()
    d["ppg"] = ppg.ppg.values[p_idx]
    d = pd.concat([d, options.iloc[r_idx].reset_index(drop=True)], axis=1)
    a = ppg.set_index("ppg").loc[d.ppg]
    for c in ["brand", "category", "segment"]:
        d[c] = a[c].values
    d["base_price"] = a.shelf_price.values
    d["actual_price"] = d.base_price * (1 - d.depth)
    d["discount_pressure"] = beta
    return d


def make_predictor(fc: Forecast, plan, ppg, options) -> Callable:
    def f(p_idx, w_idx, r_idx, beta):
        return fc.predict(scenario_rows(plan, ppg, options, np.asarray(p_idx), np.asarray(w_idx),
                                        np.asarray(r_idx), np.asarray(beta, float)))
    return f


def forecast_curves(predictor, P, T, R, grid):
    pp, ww, rr, gg = np.meshgrid(range(P), range(T), range(R), range(len(grid)), indexing="ij")
    return predictor(pp.ravel(), ww.ravel(), rr.ravel(), grid[gg.ravel()]).reshape(P, T, R, len(grid))


# --------------------------------------------------------------------------- PWL
def hat_basis(xs, grid):
    K = len(xs)
    A = np.zeros((len(grid), K))
    for k in range(K):
        e = np.zeros(K); e[k] = 1.0
        A[:, k] = np.interp(grid, xs, e)
    return A


def fit_pwl(curves, grid, K):
    """Shared breakpoint positions (SLSQP) for a (R, G) block; demand values by non-negative least squares."""
    lo, hi = grid[0], grid[-1]
    scale = np.maximum(curves.mean(axis=1, keepdims=True), 1e-9)

    def solve_y(xs):
        A = hat_basis(xs, grid)
        Y = np.array([nnls(A, c)[0] for c in curves])
        return Y, float(np.mean(((Y @ A.T - curves) / scale) ** 2))

    if K == 2:
        xs = np.array([lo, hi]); Y, err = solve_y(xs); return xs, Y, err
    gap = 0.01
    cons = [{"type": "ineq", "fun": (lambda z, i=i: z[i + 1] - z[i] - gap)} for i in range(K - 3)]
    res = minimize(lambda z: solve_y(np.r_[lo, np.sort(z), hi])[1], np.linspace(lo, hi, K)[1:-1], method="SLSQP",
                   bounds=[(lo + gap, hi - gap)] * (K - 2), constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    xs = np.r_[lo, np.sort(res.x), hi]
    Y, err = solve_y(xs)
    return xs, Y, err


def knee_point(ks, errs):
    x = (np.asarray(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.asarray(errs) - min(errs)) / max(max(errs) - min(errs), 1e-18)
    return int(ks[int(np.argmax((1 - x) - y))])


def fit_all_pwl(curves, grid, k_range=(2, 3, 4, 5, 6), k_fixed: Optional[int] = None):
    P, T = curves.shape[:2]
    sample = [(p, w) for p in range(P) for w in sorted({0, T // 2, T - 1})]
    err_by_k = {k: float(np.mean([fit_pwl(curves[p, w], grid, k)[2] for p, w in sample])) for k in k_range}
    K = k_fixed or knee_point(list(k_range), list(err_by_k.values()))
    R = curves.shape[2]
    bp, dem = np.zeros((P, T, K)), np.zeros((P, T, R, K))
    for p in range(P):
        for w in range(T):
            bp[p, w], dem[p, w], _ = fit_pwl(curves[p, w], grid, K)
    approx = np.einsum("ptrk,ptgk->ptrg", dem, np.stack([[hat_basis(bp[p, w], grid) for w in range(T)] for p in range(P)]))
    rel = np.abs(approx - curves) / np.maximum(curves, 1e-9)
    return bp, dem, K, err_by_k, rel


# --------------------------------------------------------------------------- MILP
@dataclass
class Instance:
    ppg: pd.DataFrame
    opts: pd.DataFrame
    bp: np.ndarray                 # [P, T, K]
    dem: np.ndarray                # [P, T, R, K]
    rhoP: np.ndarray
    muP: np.ndarray
    rhoR: np.ndarray
    muR: np.ndarray
    locks: list = field(default_factory=list)
    week_labels: Optional[list] = None
    lgbm: Optional[Callable] = None
    funding: float = 0.5

    @property
    def dims(self):
        return self.dem.shape[:3]

    def peers(self):
        seg = self.ppg.segment.values
        return [[q for q in range(len(seg)) if q != p and seg[q] == seg[p]] for p in range(len(seg))]


def unit_economics(ppg, opts, T, funding=0.5):
    shelf = ppg.shelf_price.values[:, None]; sell = ppg.sell_in.values[:, None]
    cost = ppg.unit_cost.values[:, None]; pep = ppg.pepsico_flag.values.astype(bool)[:, None]
    dlt = opts.depth.values[None, :]
    pep_rev = sell - funding * shelf * dlt
    rhoP = np.where(pep, pep_rev, 0.0); muP = np.where(pep, pep_rev - cost, 0.0)
    rhoR = shelf * (1 - dlt) + 0 * pep; muR = rhoR - pep_rev
    rep = lambda a: np.repeat(a[:, None, :], T, axis=1)
    return rep(rhoP), rep(muP), rep(rhoR), rep(muR)


def _match(inst, filters):
    Pn, _, Rn = inst.dims
    if not filters:
        return np.ones((Pn, Rn), bool)
    if isinstance(filters, dict):
        filters = [filters]
    mask = np.zeros((Pn, Rn), bool)
    for f in filters:
        m = np.ones((Pn, Rn), bool)
        for k, vals in f.items():
            vals = vals if isinstance(vals, list) else [vals]
            if k in inst.ppg.columns:
                m &= np.isin(inst.ppg[k].to_numpy(dtype=object), vals)[:, None]
            elif k in inst.opts.columns:
                m &= np.isin(inst.opts[k].to_numpy(dtype=object), vals)[None, :]
            else:
                raise KeyError(f"unknown filter column {k}")
        mask |= m
    return mask


@dataclass
class Solution:
    status: str
    objective: Optional[float] = None
    dual_bound: Optional[float] = None
    calendar: Optional[np.ndarray] = None
    kpis: dict = field(default_factory=dict)
    gap: Optional[float] = None
    runtime: float = 0.0
    nodes: int = 0
    size: dict = field(default_factory=dict)
    slack: dict = field(default_factory=dict)
    q: Optional[np.ndarray] = None
    qhat: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None

    @property
    def ok(self):
        return self.calendar is not None


def solve_promoai(inst: Instance, cfg: dict, elastic: bool = False) -> Solution:
    """Appendix A MILP: objective (2), constraints (3)-(13) + A.6.12 templates from JSON."""
    Pn, Tn, Rn = inst.dims
    if Pn == 0 or Tn == 0:
        return Solution("Optimal (empty input)", 0.0, 0.0, np.zeros((Pn, Tn), int), {"pepsico_sales": 0.0})
    K = inst.bp.shape[2]
    h = highspy.Highs(); h.silent()
    ow = cfg.get("Objective", {})
    w1 = ow.get("w_pepsico", 1.0); w2 = 1 - w1
    w3 = ow.get("w_pep_margin", 0.0); w4 = ow.get("w_ret_margin", 0.0)
    pep = inst.ppg.pepsico_flag.values.astype(bool)
    delta = inst.opts.depth.values
    active = delta > 0
    peers = inst.peers()
    form = cfg.get("Solver", {}).get("formulation", "hull")
    n_bin = 0

    x = np.empty((Pn, Tn, Rn), object)
    for i in np.ndindex(Pn, Tn, Rn):
        x[i] = h.addBinary(); n_bin += 1
    alpha = np.empty((Pn, Tn), object)
    for p, w in np.ndindex(Pn, Tn):
        h.addConstr(sum(x[p, w, r] for r in range(Rn)) == 1)                                  # (3)
        alpha[p, w] = h.addVariable(lb=0.0)
        terms = [float(delta[r]) * x[p, w, r] for r in range(Rn) if delta[r] > 0]
        h.addConstr(alpha[p, w] == (sum(terms) if terms else 0))                              # (4)

    q = np.empty((Pn, Tn, Rn), object); qhat = np.empty((Pn, Tn, Rn), object)
    for p, w in np.ndindex(Pn, Tn):
        beta = h.addVariable(lb=0.0)
        if peers[p]:
            h.addConstr(beta == (1.0 / len(peers[p])) * sum(alpha[o, w] for o in peers[p]))    # (5)
        else:
            h.addConstr(beta == 0.0)
        z = None
        if K > 2:
            z = [h.addBinary() for _ in range(K - 1)]; n_bin += K - 1
            h.addConstr(sum(z) == 1)

        def sos2(lam_k):
            if z is None:
                return
            for k in range(K):
                h.addConstr(lam_k[k] <= sum(z[j] for j in (k - 1, k) if 0 <= j < K - 1))

        if form == "bigm":                                                                     # paper (6)+(7)
            lam = [h.addVariable(lb=0.0, ub=1.0) for _ in range(K)]
            h.addConstr(sum(lam) == 1)
            h.addConstr(beta == sum(float(inst.bp[p, w, k]) * lam[k] for k in range(K)))
            sos2(lam)
            for r in range(Rn):
                qhat[p, w, r] = h.addVariable(lb=0.0)
                h.addConstr(qhat[p, w, r] == sum(float(inst.dem[p, w, r, k]) * lam[k] for k in range(K)))
                M = float(inst.dem[p, w, r].max())
                q[p, w, r] = h.addVariable(lb=0.0, ub=M)
                h.addConstr(qhat[p, w, r] - q[p, w, r] <= M * (1 - x[p, w, r]))
                h.addConstr(q[p, w, r] <= M * x[p, w, r])
                h.addConstr(q[p, w, r] <= qhat[p, w, r])
        else:                                                                                  # convex hull
            lam = [[h.addVariable(lb=0.0, ub=1.0) for _ in range(K)] for _ in range(Rn)]
            for r in range(Rn):
                h.addConstr(sum(lam[r]) == x[p, w, r])
                q[p, w, r] = h.addVariable(lb=0.0)
                h.addConstr(q[p, w, r] == sum(float(inst.dem[p, w, r, k]) * lam[r][k] for k in range(K)))
            h.addConstr(beta == sum(float(inst.bp[p, w, k]) * lam[r][k] for r in range(Rn) for k in range(K)))
            sos2([sum(lam[r][k] for r in range(Rn)) for k in range(K)])

    def total(coef, ppg_mask=None):
        terms = [float(coef[p, w, r]) * q[p, w, r] for p, w, r in np.ndindex(Pn, Tn, Rn)
                 if coef[p, w, r] != 0 and (ppg_mask is None or ppg_mask[p])]
        return sum(terms) if terms else 0
    SP, PiP, SR, PiR = [h.addVariable(lb=-highspy.kHighsInf) for _ in range(4)]
    h.addConstr(SP == total(inst.rhoP)); h.addConstr(PiP == total(inst.muP))                  # (8)-(9)
    h.addConstr(SR == total(inst.rhoR)); h.addConstr(PiR == total(inst.muR))

    slacks = {}

    def soft(name, expr, scale):
        if elastic:
            s = h.addVariable(lb=0.0); slacks[name] = (s, scale)
            h.addConstr(expr + s >= 0)
        else:
            h.addConstr(expr >= 0)

    mf = cfg.get("MarginFloor") or {}
    if mf.get("pepsico") is not None:
        soft("PepsiCo margin floor (currency)", PiP - mf["pepsico"] * SP, 1.0)                 # (10a)
    if mf.get("retailer") is not None:
        soft("Retailer margin floor (currency)", PiR - mf["retailer"] * SR, 1.0)               # (10b)
    act = lambda mask: [x[p, w, r] for p, w, r in np.ndindex(Pn, Tn, Rn) if active[r] and mask[p]]
    if cfg.get("MaxPromoTotal") is not None:
        soft("Max promotions total (count)", cfg["MaxPromoTotal"] - sum(act(np.ones(Pn, bool))), 1e-3)   # (11a)
    if cfg.get("MaxPromoPepsiCo") is not None and pep.any():
        soft("Max PepsiCo promotions (count)", cfg["MaxPromoPepsiCo"] - sum(act(pep)), 1e-3)            # (11b)
    if cfg.get("MinMarketShare") is not None and pep.any():
        soft("PepsiCo market share (currency)", total(inst.rhoR, pep) - cfg["MinMarketShare"] * SR, 1.0)  # (12)
    for (p, w, r) in inst.locks:
        h.addConstr(x[p, w, r] == 1)                                                           # (13)

    for rule in cfg.get("WeeklyMaxPromo", []) or []:
        mask = _match(inst, rule.get("data"))
        for w in range(Tn):
            terms = [x[p, w, r] for p in range(Pn) for r in range(Rn) if active[r] and mask[p, r]]
            if terms:
                soft(f"Weekly mechanic cap wk{w + 1} (count)", rule["week_max_promo"] - sum(terms), 1e-3)
    a = {}

    def a_(p, w):
        if (p, w) not in a:
            a[p, w] = sum(x[p, w, r] for r in range(Rn) if active[r])
        return a[p, w]
    dur, spc = cfg.get("MinMaxPromoDuration"), cfg.get("MinSpacing")
    if dur or spc:
        dmask = _match(inst, dur.get("data")).any(1) if dur else np.zeros(Pn, bool)
        smask = _match(inst, spc.get("data")).any(1) if spc else np.zeros(Pn, bool)
        for p in range(Pn):
            if not (dmask[p] or smask[p]):
                continue
            st = [h.addVariable(lb=0, ub=1) for _ in range(Tn)]
            en = [h.addVariable(lb=0, ub=1) for _ in range(Tn)]
            for w in range(Tn):
                prev = a_(p, w - 1) if w > 0 else 0
                nxt = a_(p, w + 1) if w < Tn - 1 else 0
                h.addConstr(st[w] >= a_(p, w) - prev); h.addConstr(st[w] <= a_(p, w))
                h.addConstr(en[w] >= a_(p, w) - nxt); h.addConstr(en[w] <= a_(p, w))
                if w > 0: h.addConstr(st[w] <= 1 - a_(p, w - 1))
                if w < Tn - 1: h.addConstr(en[w] <= 1 - a_(p, w + 1))
            if dmask[p]:
                mn, mx = dur.get("min_promo_block", 1), dur.get("max_promo_block", Tn)
                for w in range(Tn):
                    for j in range(1, mn):
                        if w + j < Tn: h.addConstr(a_(p, w + j) >= st[w])
                for w in range(Tn - mx):
                    h.addConstr(sum(a_(p, w + j) for j in range(mx + 1)) <= mx)
            if smask[p]:
                for w in range(Tn):
                    for j in range(1, spc["weeks"] + 1):
                        if w + j < Tn: h.addConstr(a_(p, w + j) <= 1 - en[w])
    mpp = cfg.get("MaxPromoPerPPG")
    if mpp:
        mask = _match(inst, mpp.get("data")).any(1)
        for p in range(Pn):
            if mask[p]:
                h.addConstr(sum(a_(p, w) for w in range(Tn)) <= mpp["max_weeks"])

    sol_cfg = cfg.get("Solver", {})
    h.setOptionValue("mip_rel_gap", float(sol_cfg.get("mip_gap", 0.0)))
    h.setOptionValue("time_limit", float(sol_cfg.get("time_limit", 60)))
    t0 = time.time()
    if elastic:
        h.minimize(sum((1.0 / sc) * s for s, sc in slacks.values()) if slacks else 0 * SP)
    else:
        h.maximize(w1 * ((1 - w3) * SP + w3 * PiP) + w2 * ((1 - w4) * SR + w4 * PiR))           # (2)
    rt = time.time() - t0
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    size = {"binaries": n_bin, "columns": h.getNumCol(), "rows": h.getNumRow()}
    if info.primal_solution_status != 2:
        return Solution(status, runtime=rt, size=size)
    vals = np.array(h.allVariableValues())
    get = np.vectorize(lambda v: vals[v.index])
    xv = get(x)
    cal = xv.argmax(axis=2)
    k = {"pepsico_sales": vals[SP.index], "pepsico_margin": vals[PiP.index],
         "retailer_sales": vals[SR.index], "retailer_margin": vals[PiR.index]}
    sl = {n: vals[s.index] for n, (s, sc) in slacks.items() if vals[s.index] > 1e-6}
    return Solution(status, h.getObjectiveValue(), info.mip_dual_bound, cal, k, info.mip_gap, rt,
                    int(info.mip_node_count), size, sl, get(q), get(qhat) if form == "bigm" else None, xv)


# --------------------------------------------------------------------------- evaluation
def evaluate(inst: Instance, cfg: dict, cal: np.ndarray, demand: str = "pwl") -> dict:
    """Independent re-computation of one calendar: KPIs, objective and every rule."""
    Pn, Tn, Rn = inst.dims
    cal = np.asarray(cal, int)
    delta = inst.opts.depth.values[cal]
    peers = inst.peers()
    beta = np.array([[delta[peers[p], w].mean() if peers[p] else 0.0 for w in range(Tn)] for p in range(Pn)])
    if demand == "pwl":
        units = np.array([[np.interp(beta[p, w], inst.bp[p, w], inst.dem[p, w, cal[p, w]]) for w in range(Tn)]
                          for p in range(Pn)])
    else:
        pw = np.array(list(np.ndindex(Pn, Tn)))
        units = inst.lgbm(pw[:, 0], pw[:, 1], cal[pw[:, 0], pw[:, 1]], beta[pw[:, 0], pw[:, 1]]).reshape(Pn, Tn)
    return kpis_from_units(inst, cfg, cal, units, beta)


def kpis_from_units(inst, cfg, cal, units, beta=None) -> dict:
    Pn, Tn, Rn = inst.dims
    pick = lambda arr: np.take_along_axis(arr, cal[..., None], 2)[..., 0]
    pep = inst.ppg.pepsico_flag.values.astype(bool)
    delta = inst.opts.depth.values[cal]
    k = {"pepsico_sales": float((units * pick(inst.rhoP)).sum()), "pepsico_margin": float((units * pick(inst.muP)).sum()),
         "retailer_sales": float((units * pick(inst.rhoR)).sum()), "retailer_margin": float((units * pick(inst.muR)).sum())}
    k["pepsico_share"] = float((units * pick(inst.rhoR))[pep].sum() / max(k["retailer_sales"], 1e-9))
    k["pepsico_margin_ratio"] = k["pepsico_margin"] / max(k["pepsico_sales"], 1e-9)
    k["retailer_margin_ratio"] = k["retailer_margin"] / max(k["retailer_sales"], 1e-9)
    k["units"] = float(units.sum())
    act = delta > 0
    k["promotions"] = int(act.sum()); k["pepsico_promotions"] = int(act[pep].sum())
    k["trade_spend"] = float((units * inst.funding * inst.ppg.shelf_price.values[:, None] * delta)[pep].sum())
    o = cfg.get("Objective", {}); w1 = o.get("w_pepsico", 1.0)
    k["objective"] = (w1 * ((1 - o.get("w_pep_margin", 0)) * k["pepsico_sales"] + o.get("w_pep_margin", 0) * k["pepsico_margin"])
                      + (1 - w1) * ((1 - o.get("w_ret_margin", 0)) * k["retailer_sales"] + o.get("w_ret_margin", 0) * k["retailer_margin"]))
    v, tol = [], 1e-6
    mf = cfg.get("MarginFloor") or {}
    if mf.get("pepsico") is not None and k["pepsico_margin"] < mf["pepsico"] * k["pepsico_sales"] - tol:
        v.append(f"PepsiCo margin {k['pepsico_margin_ratio']:.1%} is below the {mf['pepsico']:.0%} floor")
    if mf.get("retailer") is not None and k["retailer_margin"] < mf["retailer"] * k["retailer_sales"] - tol:
        v.append(f"Retailer margin {k['retailer_margin_ratio']:.1%} is below the {mf['retailer']:.0%} floor")
    if cfg.get("MaxPromoTotal") is not None and k["promotions"] > cfg["MaxPromoTotal"]:
        v.append(f"{k['promotions']} promotions exceed the total cap of {cfg['MaxPromoTotal']}")
    if cfg.get("MaxPromoPepsiCo") is not None and k["pepsico_promotions"] > cfg["MaxPromoPepsiCo"]:
        v.append(f"{k['pepsico_promotions']} PepsiCo promotions exceed the cap of {cfg['MaxPromoPepsiCo']}")
    if cfg.get("MinMarketShare") is not None and pep.any() and k["pepsico_share"] < cfg["MinMarketShare"] - tol:
        v.append(f"PepsiCo share {k['pepsico_share']:.1%} is below the {cfg['MinMarketShare']:.0%} target")
    for (p, w, r) in inst.locks:
        if cal[p, w] != r:
            v.append(f"Locked rival slot changed: {inst.ppg.ppg.values[p]} week {w + 1}")
    for rule in cfg.get("WeeklyMaxPromo", []) or []:
        m = _match(inst, rule.get("data"))
        for w in range(Tn):
            n = sum(1 for p in range(Pn) if act[p, w] and m[p, cal[p, w]])
            if n > rule["week_max_promo"]:
                v.append(f"Week {w + 1}: {n} {rule.get('data')} promotions exceed the cap of {rule['week_max_promo']}")
    dur, spc, mpp = cfg.get("MinMaxPromoDuration"), cfg.get("MinSpacing"), cfg.get("MaxPromoPerPPG")
    for p in range(Pn):
        blocks, w = [], 0
        while w < Tn:
            if act[p, w]:
                s = w
                while w < Tn and act[p, w]:
                    w += 1
                blocks.append((s, w - 1))
            else:
                w += 1
        name = inst.ppg.ppg.values[p]
        if dur and _match(inst, dur.get("data")).any(1)[p]:
            for s, e in blocks:
                L = e - s + 1
                if L > dur.get("max_promo_block", Tn) or (L < dur.get("min_promo_block", 1) and e < Tn - 1):
                    v.append(f"{name}: promotion block of {L} weeks outside "
                             f"{dur.get('min_promo_block', 1)}-{dur.get('max_promo_block')} weeks")
        if spc and _match(inst, spc.get("data")).any(1)[p]:
            for (s1, e1), (s2, e2) in zip(blocks, blocks[1:]):
                if s2 - e1 - 1 < spc["weeks"]:
                    v.append(f"{name}: only {s2 - e1 - 1} week(s) between promotions (min {spc['weeks']})")
        if mpp and _match(inst, mpp.get("data")).any(1)[p] and act[p].sum() > mpp["max_weeks"]:
            v.append(f"{name}: {int(act[p].sum())} promotion weeks exceed the max of {mpp['max_weeks']}")
    k["violations"] = v
    k["feasible"] = not v
    k["units_grid"] = units
    return k


def baseline_calendar(inst: Instance) -> np.ndarray:
    cal = np.zeros(inst.dims[:2], int)
    for (p, w, r) in inst.locks:
        cal[p, w] = r
    return cal


def greedy(inst: Instance, cfg: dict, start: Optional[np.ndarray] = None) -> np.ndarray:
    """Build heuristic: repeatedly apply the single feasible change with the largest objective gain."""
    cal = baseline_calendar(inst) if start is None else start.copy()
    cur = evaluate(inst, cfg, cal)["objective"]
    free = [(p, w) for p in range(inst.dims[0]) for w in range(inst.dims[1]) if inst.ppg.pepsico_flag.values[p]]
    while True:
        best = None
        for p, w in free:
            for r in range(1, inst.dims[2]):
                if cal[p, w] == r:
                    continue
                c2 = cal.copy(); c2[p, w] = r
                k = evaluate(inst, cfg, c2)
                if k["feasible"] and k["objective"] > cur + 1e-6 and (best is None or k["objective"] > best[0]):
                    best = (k["objective"], c2)
        if best is None:
            return cal
        cur, cal = best


# --------------------------------------------------------------------------- reality simulation
def simulate_reality(inst: Instance, cfg: dict, ppg_true: pd.DataFrame, plan: pd.DataFrame, cals: dict,
                     n_runs: int = 200, noise: float = 0.08, elast_mult: float = 1.0, seed: int = 11) -> pd.DataFrame:
    """Play each calendar against the hidden 'true' demand process with random noise (Monte Carlo)."""
    Pn, Tn, _ = inst.dims
    rng = np.random.default_rng(seed)
    shocks = rng.lognormal(0, noise, (n_runs, Pn, Tn))       # same shocks for every calendar (fair comparison)
    peers = inst.peers()
    out = []
    for name, cal in cals.items():
        cal = np.asarray(cal, int)
        delta = inst.opts.depth.values[cal]
        beta = np.array([[delta[peers[p], w].mean() if peers[p] else 0.0 for w in range(Tn)] for p in range(Pn)])
        pw = np.array(list(np.ndindex(Pn, Tn)))
        d = scenario_rows(plan, inst.ppg, inst.opts, pw[:, 0], pw[:, 1], cal[pw[:, 0], pw[:, 1]], beta[pw[:, 0], pw[:, 1]])
        mean_units = true_units_mean(d, ppg_true, elast_mult).reshape(Pn, Tn)
        for i in range(n_runs):
            k = kpis_from_units(inst, cfg, cal, mean_units * shocks[i])
            out.append({"calendar": name, "run": i, "objective": k["objective"], "pepsico_sales": k["pepsico_sales"],
                        "pepsico_margin": k["pepsico_margin"], "pepsico_margin_ratio": k["pepsico_margin_ratio"],
                        "retailer_margin": k["retailer_margin"]})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- full pipeline
def build_instance(ppg, options, plan, bp, dem, rival_plan: dict, funding: float, predictor=None) -> Instance:
    T = len(plan)
    rho_p, mu_p, rho_r, mu_r = unit_economics(ppg, options, T, funding)
    oi = {o: i for i, o in enumerate(options.option)}
    idx = {p: i for i, p in enumerate(ppg.ppg)}
    locks = []
    for p in ppg.ppg[~ppg.pepsico_flag.astype(bool)]:
        for w in range(T):
            opt = rival_plan.get(p, {}).get(w, "NONE")
            locks.append((idx[p], w, oi.get(opt, 0)))
    labels = [f"wk{w}" + ("*" if h else "") for w, h in zip(plan.week_of_year, plan.holiday)]
    return Instance(ppg.reset_index(drop=True), options.reset_index(drop=True), bp, dem, rho_p, mu_p, rho_r, mu_r,
                    locks, labels, predictor, funding)


# --------------------------------------------------------------------------- built-in tests
def toy_instance(rival=False) -> Instance:
    names = ["A", "B"] + (["RIVAL"] if rival else [])
    ppg = pd.DataFrame({"ppg": names, "segment": ["sA", "sB"] + (["sA"] if rival else []),
                        "pepsico_flag": [True, True] + ([False] if rival else []), "shelf_price": [10.0] * len(names)})
    opts = pd.DataFrame({"option": ["NONE", "P15"], "mechanic_type": ["None", "Price"], "depth": [0.0, 0.15]})
    Pn, Tn, Rn, K = len(names), 4, 2, 2
    bp = np.tile([0.0, 0.15], (Pn, Tn, 1))
    units = {"A": (100, 160), "B": (80, 110), "RIVAL": (100, 150)}
    dem = np.zeros((Pn, Tn, Rn, K))
    for i, n in enumerate(names):
        dem[i, :, 0, :] = units[n][0]; dem[i, :, 1, :] = units[n][1]
    if rival:
        dem[0, :, 1, 1] = 130; dem[0, :, 0, 1] = 85
    price = np.array([10.0, 8.5])
    pepm = ppg.pepsico_flag.values[:, None, None]
    rhoP = np.where(pepm, np.tile(price, (Pn, Tn, 1)), 0); muP = np.where(pepm, np.tile(price - 6.0, (Pn, Tn, 1)), 0)
    z = np.zeros((Pn, Tn, Rn))
    return Instance(ppg, opts, bp, dem, rhoP, muP, z, z.copy())


TOY_CFG = {"Objective": {"w_pepsico": 1.0, "w_pep_margin": 0.0}, "MarginFloor": {"pepsico": 0.30},
           "MaxPromoTotal": 3, "MinMaxPromoDuration": {"data": [{"pepsico_flag": True}], "max_promo_block": 1},
           "MinSpacing": {"data": [{"pepsico_flag": True}], "weeks": 1},
           "MaxPromoPerPPG": {"data": [{"pepsico_flag": True}], "max_weeks": 2}}


def run_toy_tests() -> pd.DataFrame:
    toy = toy_instance()
    run = lambda **c: solve_promoai(toy, with_changes(TOY_CFG, **c))
    rows = []

    def t(tid, desc, expected, fn):
        t0 = time.time()
        try:
            obs = fn()
            rows.append((tid, desc, expected, obs, "PASS", round(time.time() - t0, 2)))
        except AssertionError as e:
            rows.append((tid, desc, expected, str(e), "FAIL", round(time.time() - t0, 2)))

    def tc01():
        s = run(MaxPromoTotal=0); assert abs(s.objective - 7200) < 1e-3 and s.calendar.sum() == 0, s.objective
        return f"revenue {s.objective:,.0f}, empty calendar"
    def tc02():
        s = run(); assert abs(s.objective - 8055) < 1e-3, s.objective; return f"revenue {s.objective:,.0f}"
    def tc03():
        s = run(MarginFloor={"pepsico": 0.36}); assert abs(s.objective - 7920) < 1e-3 and s.calendar[1].sum() == 0, s.objective
        return f"revenue {s.objective:,.0f}, B never promoted"
    def tc04():
        s = run(MarginFloor={"pepsico": 0.40}); assert abs(s.objective - 7200) < 1e-3, s.objective; return f"revenue {s.objective:,.0f}"
    def tc05():
        c = with_changes(TOY_CFG, MarginFloor={"pepsico": 0.41})
        s = solve_promoai(toy, c); assert not s.ok, "should be infeasible"
        e = solve_promoai(toy, c, elastic=True); sl = e.slack.get("PepsiCo margin floor (currency)")
        assert sl is not None and abs(sl - 72) < 1e-4, sl; return f"{s.status}; slack {sl:.1f}"
    def tc06():
        s = run(); a = s.calendar > 0; assert not (a[:, 1:] & a[:, :-1]).any(); return "no adjacent promotion weeks"
    def tc07():
        s = run(Objective={"w_pepsico": 1.0, "w_pep_margin": 1.0}); assert abs(s.objective - 2880) < 1e-3 and s.calendar[1].sum() == 0
        return f"margin {s.objective:,.0f}, no B promotions"
    def tc08():
        tt = toy_instance(); tt.locks = [(1, 1, 1)]; s = solve_promoai(tt, TOY_CFG)
        assert s.calendar[1, 1] == 1 and abs(s.objective - 8055) < 1e-3; return "B locked in week 2, revenue 8,055"
    def tc09():
        e = Instance(toy.ppg.iloc[:0], toy.opts, np.zeros((0, 4, 2)), np.zeros((0, 4, 2, 2)), *[np.zeros((0, 4, 2))] * 4)
        s = solve_promoai(e, TOY_CFG); assert s.objective == 0.0; return s.status
    def tc10():
        tt = toy_instance(rival=True); tt.locks = [(2, w, 1) for w in range(4)]
        s = solve_promoai(tt, with_changes(TOY_CFG, MaxPromoTotal=7))
        u = [s.q[0, w, 1] for w in range(4) if s.calendar[0, w] == 1]
        assert u and max(u) < 160 - 1e-6; return f"A promo units {[round(x) for x in u]} vs 160 without rival"

    t("TC_001", "No promotions allowed (N = 0)", "Revenue 7,200", tc01)
    t("TC_002", "Loose margin floor 0.30", "Revenue 8,055", tc02)
    t("TC_003", "Margin floor binds 0.36", "Revenue 7,920, B never", tc03)
    t("TC_004", "Tight floor 0.40", "Revenue 7,200", tc04)
    t("TC_005", "Impossible floor 0.41", "Infeasible, slack 72", tc05)
    t("TC_006", "Spacing 1 week", "No adjacent promo weeks", tc06)
    t("TC_007", "Margin objective", "Margin 2,880, no B", tc07)
    t("TC_008", "Locked slot B week 2", "Revenue 8,055", tc08)
    t("TC_009", "Empty input", "Revenue 0", tc09)
    t("TC_010", "Rival pressure", "A's promo units < 160", tc10)
    return pd.DataFrame(rows, columns=["ID", "Test", "Expected", "Observed", "Result", "Seconds"])


def brute_force(inst: Instance, cfg: dict, max_calendars: int = 20000):
    Pn, Tn, Rn = inst.dims
    free = [p for p in range(Pn) if inst.ppg.pepsico_flag.values[p]]
    n = Rn ** (len(free) * Tn)
    if n > max_calendars:
        raise ValueError(f"{n:,} calendars is too many for brute force")
    base = baseline_calendar(inst)
    best, n_feas = None, 0
    for combo in itertools.product(range(Rn), repeat=len(free) * Tn):
        cal = base.copy(); cal[free] = np.array(combo).reshape(len(free), Tn)
        k = evaluate(inst, cfg, cal)
        if k["feasible"]:
            n_feas += 1
            if best is None or k["objective"] > best[0] + 1e-9:
                best = (k["objective"], cal)
    return best, n, n_feas


def tiny_slice(inst: Instance, weeks=(3, 4, 5, 6), opt_names=("NONE", "TPR20", "TPR30")) -> Optional[Instance]:
    """First segment that has >=2 PepsiCo PPGs (max 2) plus its rivals, 4 weeks, 3 options."""
    for seg in inst.ppg.segment.unique():
        m = inst.ppg.segment.values == seg
        pep_idx = [i for i in np.where(m)[0] if inst.ppg.pepsico_flag.values[i]][:2]
        riv_idx = [i for i in np.where(m)[0] if not inst.ppg.pepsico_flag.values[i]]
        if len(pep_idx) >= 2:
            break
    else:
        return None
    keep_p = pep_idx + riv_idx
    keep_w = [w for w in weeks if w < inst.dims[1]]
    oi = list(inst.opts.option)
    keep_r = [oi.index(o) for o in opt_names if o in oi]
    if len(keep_w) < 2 or len(keep_r) < 2:
        return None
    sub3 = lambda a: a[np.ix_(keep_p, keep_w, keep_r)]
    locks = [(keep_p.index(p), keep_w.index(w), keep_r.index(r)) for (p, w, r) in inst.locks
             if p in keep_p and w in keep_w and r in keep_r]
    locks += [(keep_p.index(p), keep_w.index(w), 0) for (p, w, r) in inst.locks
              if p in keep_p and w in keep_w and r not in keep_r]
    return Instance(inst.ppg.iloc[keep_p].reset_index(drop=True), inst.opts.iloc[keep_r].reset_index(drop=True),
                    inst.bp[np.ix_(keep_p, keep_w)], inst.dem[np.ix_(keep_p, keep_w)][..., keep_r, :],
                    sub3(inst.rhoP), sub3(inst.muP), sub3(inst.rhoR), sub3(inst.muR), locks, funding=inst.funding)
