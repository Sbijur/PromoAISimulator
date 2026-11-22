"""
PromoAI simulator (Streamlit).  Run:  streamlit run app.py
Interactive front end for promo_engine.py: synthetic retailer data -> LightGBM forecast -> PWL demand
-> promotion calendar MILP (HiGHS) -> KPIs, manual edits, scenarios, Monte Carlo reality check, tests.
"""
import hashlib
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import promo_engine as pe

st.set_page_config(page_title="PromoAI simulator", page_icon="🥤", layout="wide")

# Colour roles: promotion depth is a magnitude -> one blue ramp (light = shallow, dark = deep);
# multibuy is a different mechanic -> its own hue; no promotion -> neutral surface grey.
OPTION_COLORS = {"NONE": "#ecebe8", "TPR10": "#b7d3f2", "TPR20": "#6fa6e8", "TPR30": "#2a5fab", "MB25": "#eb6834"}
CAL_COLORS = {"No PepsiCo promotion": "#8f8e8a", "Greedy heuristic": "#eb6834",
              "PromoAI MILP": "#2a78d6", "Your edited calendar": "#1baf7a"}
ALL_ACTIVE = ["TPR10", "TPR20", "TPR30", "MB25"]
OPTION_HELP = "NONE = no promotion · TPR10/20/30 = temporary price cut of 10/20/30% · MB25 = multibuy worth ≈25%"


# =============================================================================== rules <-> widgets
def state_from_cfg(cfg: dict):
    """Push a rules JSON into the sidebar widgets (used by the JSON upload)."""
    o = cfg.get("Objective", {})
    ss = st.session_state
    ss.w_pepsico = float(o.get("w_pepsico", 0.7)); ss.w_pep_margin = float(o.get("w_pep_margin", 0.5))
    ss.w_ret_margin = float(o.get("w_ret_margin", 0.5))
    mf = cfg.get("MarginFloor") or {}
    ss.floor_pep_on = mf.get("pepsico") is not None; ss.floor_pep = int(round(100 * (mf.get("pepsico") or 0.40)))
    ss.floor_ret_on = mf.get("retailer") is not None; ss.floor_ret = int(round(100 * (mf.get("retailer") or 0.28)))
    ss.max_total_on = cfg.get("MaxPromoTotal") is not None; ss.max_total = int(cfg.get("MaxPromoTotal") or 30)
    ss.max_pep_on = cfg.get("MaxPromoPepsiCo") is not None; ss.max_pep = int(cfg.get("MaxPromoPepsiCo") or 24)
    ss.share_on = cfg.get("MinMarketShare") is not None; ss.share = int(round(100 * (cfg.get("MinMarketShare") or 0.55)))
    wk = cfg.get("WeeklyMaxPromo") or []
    ss.weekly_on = bool(wk); ss.weekly_cap = int(wk[0]["week_max_promo"]) if wk else 3
    d = cfg.get("MinMaxPromoDuration")
    ss.dur_on = bool(d); ss.block = (int((d or {}).get("min_promo_block", 1)), int((d or {}).get("max_promo_block", 2)))
    s = cfg.get("MinSpacing")
    ss.spacing_on = bool(s); ss.spacing = int((s or {}).get("weeks", 2))
    m = cfg.get("MaxPromoPerPPG")
    ss.maxw_on = bool(m); ss.max_weeks = int((m or {}).get("max_weeks", 4))
    sv = cfg.get("Solver", {})
    ss.gap = {0.0: "0% (prove optimal)", 0.005: "0.5%", 0.01: "1%", 0.02: "2%", 0.05: "5%"}.get(float(sv.get("mip_gap", 0.01)), "1%")
    ss.time_limit = int(sv.get("time_limit", 60))
    ss.formulation = "Convex hull (tight)" if sv.get("formulation", "hull") == "hull" else "Big-M (paper eq. 7)"


def cfg_from_state() -> dict:
    ss = st.session_state
    cfg = {"Objective": {"w_pepsico": ss.w_pepsico, "w_pep_margin": ss.w_pep_margin, "w_ret_margin": ss.w_ret_margin},
           "MarginFloor": {}}
    if ss.floor_pep_on: cfg["MarginFloor"]["pepsico"] = ss.floor_pep / 100
    if ss.floor_ret_on: cfg["MarginFloor"]["retailer"] = ss.floor_ret / 100
    cfg["MaxPromoTotal"] = int(ss.max_total) if ss.max_total_on else None
    cfg["MaxPromoPepsiCo"] = int(ss.max_pep) if ss.max_pep_on else None
    cfg["MinMarketShare"] = ss.share / 100 if ss.share_on else None
    cfg["WeeklyMaxPromo"] = [{"data": {"mechanic_type": ["Price"]}, "week_max_promo": int(ss.weekly_cap)}] if ss.weekly_on else []
    cfg["MinMaxPromoDuration"] = ({"data": [{"pepsico_flag": True}], "min_promo_block": int(ss.block[0]),
                                   "max_promo_block": int(ss.block[1])} if ss.dur_on else None)
    cfg["MinSpacing"] = {"data": [{"pepsico_flag": True}], "weeks": int(ss.spacing)} if ss.spacing_on else None
    cfg["MaxPromoPerPPG"] = {"data": [{"pepsico_flag": True}], "max_weeks": int(ss.max_weeks)} if ss.maxw_on else None
    gap = {"0% (prove optimal)": 0.0, "0.5%": 0.005, "1%": 0.01, "2%": 0.02, "5%": 0.05}[ss.gap]
    cfg["Solver"] = {"mip_gap": gap, "time_limit": int(ss.time_limit),
                     "formulation": "hull" if ss.formulation.startswith("Convex") else "bigm"}
    return cfg


if "w_pepsico" not in st.session_state:
    state_from_cfg(pe.DEFAULT_RULES)


# =============================================================================== cached pipeline steps
@st.cache_resource(show_spinner=False)
def get_forecast(ppg: pd.DataFrame, seed: int, hist_weeks: int, noise: float, promo_prob: float,
                 elast_mult: float, n_trials: int):
    raw = pe.simulate_history(ppg, pe.DEFAULT_OPTIONS, n_weeks=hist_weeks, seed=seed, noise=noise,
                              promo_prob=promo_prob, elast_mult=elast_mult)
    hist, dq = pe.clean_history(raw)
    fc = pe.train_forecast(hist, ppg, pe.DEFAULT_OPTIONS, n_trials=n_trials, seed=seed)
    return hist, dq, fc


@st.cache_resource(show_spinner=False)
def get_pwl(_fc, fc_key: str, ppg: pd.DataFrame, opts: pd.DataFrame, start: str, n_weeks: int, k_choice: str):
    plan = pe.calendar_frame(start, n_weeks)
    pred = pe.make_predictor(_fc, plan, ppg, opts)
    grid = np.linspace(0.0, 0.30, 13)
    curves = pe.forecast_curves(pred, len(ppg), n_weeks, len(opts), grid)
    k_fixed = None if k_choice == "Auto (knee point)" else int(k_choice)
    bp, dem, K, err_by_k, rel = pe.fit_all_pwl(curves, grid, k_fixed=k_fixed)
    return plan, pred, grid, curves, bp, dem, K, err_by_k, rel


def key_of(*parts) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update((p.to_json() if isinstance(p, pd.DataFrame) else json.dumps(p, default=str, sort_keys=True)).encode())
    return h.hexdigest()


# =============================================================================== chart helpers
def calendar_chart(inst, cal, title=""):
    rows = []
    for p in range(inst.dims[0]):
        for w in range(inst.dims[1]):
            o = inst.opts.option.values[cal[p, w]]
            pep = bool(inst.ppg.pepsico_flag.values[p])
            rows.append({"PPG": inst.ppg.ppg.values[p] + ("" if pep else " (rival, locked)"),
                         "Week": inst.week_labels[w], "Option": o, "Label": "" if o == "NONE" else o,
                         "Depth": float(inst.opts.depth.values[cal[p, w]])})
    df = pd.DataFrame(rows)
    y_order = list(dict.fromkeys(df.PPG))
    present = [o for o in OPTION_COLORS if o in set(df.Option)]
    base = alt.Chart(df).encode(x=alt.X("Week:N", sort=inst.week_labels, title=None, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y("PPG:N", sort=y_order, title=None, axis=alt.Axis(labelLimit=260)))
    rect = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("Option:N", scale=alt.Scale(domain=present, range=[OPTION_COLORS[o] for o in present]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["PPG", "Week", "Option", alt.Tooltip("Depth:Q", format=".0%")])
    text = base.mark_text(fontSize=11, fontWeight=600).encode(
        text="Label:N", color=alt.condition(alt.datum.Option == "TPR30", alt.value("white"), alt.value("#1f1f1f")))
    return (rect + text).properties(height=34 * len(y_order) + 40, title=title)


def kpi_table(rows: dict) -> pd.DataFrame:
    cols = ["objective", "pepsico_sales", "pepsico_margin", "retailer_sales", "retailer_margin", "pepsico_margin_ratio",
            "retailer_margin_ratio", "pepsico_share", "pepsico_promotions", "trade_spend", "units", "feasible"]
    df = pd.DataFrame({n: {c: k[c] for c in cols} for n, k in rows.items()}).T
    df.columns = ["Objective", "PepsiCo sales", "PepsiCo margin", "Retailer sales", "Retailer margin", "PepsiCo margin %",
                  "Retailer margin %", "PepsiCo share", "PepsiCo promos", "Trade spend", "Units", "Feasible"]
    return df


def style_kpis(df):
    money = ["Objective", "PepsiCo sales", "PepsiCo margin", "Retailer sales", "Retailer margin", "Trade spend", "Units"]
    pct = ["PepsiCo margin %", "Retailer margin %", "PepsiCo share"]
    fmt = {c: "{:,.0f}" for c in money if c in df} | {c: "{:.1%}" for c in pct if c in df}
    return df.style.format(fmt)


# =============================================================================== sidebar
with st.sidebar:
    st.title("🥤 PromoAI simulator")
    st.caption("LightGBM forecast → piecewise-linear demand → MILP promotion calendar.")

    up = st.file_uploader("Load rules JSON (paper Appendix B format)", type="json")
    if up is not None and st.session_state.get("_last_upload") != up.file_id:
        try:
            state_from_cfg(json.load(up))
            st.session_state["_last_upload"] = up.file_id
            st.rerun()
        except Exception as e:
            st.error(f"Could not read rules: {e}")

    with st.expander("① Objective weights", expanded=True):
        st.slider("Weight on PepsiCo (rest goes to retailer)", 0.0, 1.0, step=0.05, key="w_pepsico",
                  help="ω1 in paper eq. (2). 1.0 = only PepsiCo matters.")
        st.slider("PepsiCo: weight on margin vs sales", 0.0, 1.0, step=0.05, key="w_pep_margin",
                  help="ω3. 0 = maximise PepsiCo sales, 1 = maximise PepsiCo margin.")
        st.slider("Retailer: weight on margin vs sales", 0.0, 1.0, step=0.05, key="w_ret_margin", help="ω4.")

    with st.expander("② Business rules", expanded=True):
        c1, c2 = st.columns([1, 3])
        c1.toggle("On", key="floor_pep_on", help="Paper eq. (10a)")
        c2.slider("PepsiCo margin floor (%)", 20, 50, key="floor_pep", disabled=not st.session_state.floor_pep_on)
        c1, c2 = st.columns([1, 3])
        c1.toggle("On", key="floor_ret_on", help="Paper eq. (10b)")
        c2.slider("Retailer margin floor (%)", 10, 40, key="floor_ret", disabled=not st.session_state.floor_ret_on)
        c1, c2 = st.columns([1, 3])
        c1.toggle("On", key="share_on", help="Paper eq. (12)")
        c2.slider("Min PepsiCo share of retailer sales (%)", 30, 80, key="share", disabled=not st.session_state.share_on)
        c1, c2 = st.columns(2)
        c1.toggle("Total cap", key="max_total_on")
        c1.number_input("Max total promotions", 0, 200, key="max_total", disabled=not st.session_state.max_total_on)
        c2.toggle("PepsiCo cap", key="max_pep_on")
        c2.number_input("Max PepsiCo promotions", 0, 200, key="max_pep", disabled=not st.session_state.max_pep_on)
        c1, c2 = st.columns([1, 3])
        c1.toggle("On", key="weekly_on", help="JSON WeeklyMaxPromo")
        c2.slider("Max price-cut promos per week", 0, 9, key="weekly_cap", disabled=not st.session_state.weekly_on)
        c1, c2 = st.columns([1, 3])
        c1.toggle("On", key="dur_on", help="JSON MinMaxPromoDuration")
        c2.slider("Promotion block length (weeks)", 1, 6, key="block", disabled=not st.session_state.dur_on)
        c1, c2 = st.columns([1, 3])
        c1.toggle("On", key="spacing_on", help="JSON MinSpacing: rest weeks to avoid pantry loading")
        c2.slider("Min weeks between promotions", 0, 6, key="spacing", disabled=not st.session_state.spacing_on)
        c1, c2 = st.columns([1, 3])
        c1.toggle("On", key="maxw_on", help="JSON MaxPromoPerPPG")
        c2.slider("Max promo weeks per PepsiCo PPG", 0, 13, key="max_weeks", disabled=not st.session_state.maxw_on)

    with st.expander("③ Market & forecast data"):
        st.number_input("Random seed", 0, 9999, 7, key="seed")
        st.slider("Weeks of sales history", 78, 208, 156, step=13, key="hist_weeks")
        st.slider("Demand noise (σ of log error)", 0.02, 0.25, 0.08, step=0.01, key="noise")
        st.slider("How often PPGs promoted in history", 0.05, 0.30, 0.13, step=0.01, key="promo_prob")
        st.slider("Promotion sensitivity multiplier", 0.3, 1.5, 1.0, step=0.1, key="elast_mult",
                  help="Scales every PPG's elasticity in the hidden 'true' market. Lower = less responsive "
                       "(closer to the 2-5% uplift benchmark the paper cites).")
        st.slider("LightGBM random-search trials", 3, 30, 10, key="n_trials")

    with st.expander("④ Horizon, options & money"):
        st.date_input("First planning week", pd.Timestamp("2027-01-04"), key="start")
        st.slider("Weeks to plan", 4, 13, 13, key="n_weeks")
        st.multiselect("Promotion options allowed", ALL_ACTIVE, ALL_ACTIVE, key="allowed", help=OPTION_HELP)
        st.slider("PepsiCo share of discount funding", 0.0, 1.0, 0.5, step=0.05, key="funding",
                  help="Trade spend: PepsiCo pays this share of the shelf discount, the retailer the rest.")
        st.selectbox("PWL breakpoints", ["Auto (knee point)", "2", "3", "4", "5", "6"], key="k_choice")

    with st.expander("⑤ Solver"):
        st.select_slider("Stop at optimality gap", ["0% (prove optimal)", "0.5%", "1%", "2%", "5%"], key="gap")
        st.slider("Time limit (s)", 5, 300, key="time_limit")
        st.radio("Formulation", ["Convex hull (tight)", "Big-M (paper eq. 7)"], key="formulation")
        st.checkbox("Also run greedy heuristic (≈20 s)", True, key="run_greedy")

    run_clicked = st.button("▶ Run model", type="primary", width="stretch")
    st.download_button("⬇ Download rules JSON", json.dumps(cfg_from_state(), indent=2), "promoai_rules.json",
                       "application/json", width="stretch")
    if st.button("↺ Reset rules to defaults", width="stretch"):
        state_from_cfg(pe.DEFAULT_RULES)
        st.rerun()

CFG = cfg_from_state()

# =============================================================================== main: inputs
st.title("PromoAI: promotion calendar simulator")
st.caption("Rebuild of the PromoAI system in Llenas et al. (2026), *PepsiCo Deploys AI-Driven Pricing and Promotion "
           "Optimization at Scale*. Change the inputs, then press **▶ Run model** in the sidebar.")

tabs = st.tabs(["📥 Inputs", "🗓️ Optimal calendar", "✏️ Edit & compare", "🧭 Scenarios", "🎲 Reality simulation",
                "📈 Forecast", "〽️ Demand curves", "✅ Tests", "ℹ️ How it works"])

with tabs[0]:
    st.subheader("Products (PPGs)")
    st.caption("Edit any cell, untick **include** to drop a product, or add a row at the bottom. "
               "Rival products (PepsiCo unticked) are not optimised: their calendar is fixed below.")
    if "ppg_df" not in st.session_state:
        st.session_state.ppg_df = pe.DEFAULT_PPG.assign(include=True)[["include"] + list(pe.DEFAULT_PPG.columns)]
    ppg_edit = st.data_editor(
        st.session_state.ppg_df, key="ppg_editor", num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "include": st.column_config.CheckboxColumn("include", default=True),
            "ppg": st.column_config.TextColumn("PPG", required=True),
            "category": st.column_config.SelectboxColumn("category", options=["CSD", "Snacks"], required=True),
            "segment": st.column_config.TextColumn("segment", help="Size/format segment: rivals in the same segment create discount pressure"),
            "pepsico_flag": st.column_config.CheckboxColumn("PepsiCo?", default=True),
            "base_units": st.column_config.NumberColumn("base units / wk", min_value=10, max_value=10000, step=10),
            "shelf_price": st.column_config.NumberColumn("shelf price", min_value=0.1, max_value=50.0, step=0.01, format="%.2f"),
            "sell_in": st.column_config.NumberColumn("sell-in price", min_value=0.1, max_value=50.0, step=0.01, format="%.2f"),
            "unit_cost": st.column_config.NumberColumn("unit cost", min_value=0.0, max_value=50.0, step=0.01, format="%.2f"),
            "elasticity": st.column_config.NumberColumn("promo elasticity", min_value=0.5, max_value=6.0, step=0.1,
                                                        help="Higher = bigger sales lift from a discount"),
        })
    ppg = ppg_edit[ppg_edit["include"].fillna(False)].drop(columns="include").dropna(subset=["ppg"]).copy()
    ppg["ppg"] = ppg.ppg.astype(str).str.strip()
    ppg = ppg[ppg.ppg != ""].drop_duplicates("ppg").reset_index(drop=True)
    for c, d in [("brand", "Brand"), ("category", "CSD"), ("segment", "seg")]:
        ppg[c] = ppg[c].fillna(d).astype(str)
    ppg["pepsico_flag"] = ppg.pepsico_flag.fillna(True).astype(bool)
    for c in ["base_units", "shelf_price", "sell_in", "unit_cost", "elasticity"]:
        ppg[c] = pd.to_numeric(ppg[c], errors="coerce").fillna(pe.DEFAULT_PPG[c].median())
    problems = []
    if ppg.empty: problems.append("Include at least one product.")
    if not ppg.pepsico_flag.any(): problems.append("At least one product must be a PepsiCo product.")
    bad_margin = ppg[ppg.pepsico_flag & (ppg.sell_in <= ppg.unit_cost)].ppg.tolist()
    if bad_margin: st.warning(f"Sell-in price ≤ unit cost for {', '.join(bad_margin)}: PepsiCo loses money on every unit.")
    bad_ret = ppg[ppg.shelf_price <= ppg.sell_in].ppg.tolist()
    if bad_ret: st.warning(f"Shelf price ≤ sell-in price for {', '.join(bad_ret)}: the retailer loses money on every unit.")

    plan_preview = pe.calendar_frame(st.session_state.start, st.session_state.n_weeks)
    week_labels = [f"wk{w}" + ("*" if h else "") for w, h in zip(plan_preview.week_of_year, plan_preview.holiday)]
    rivals = ppg[~ppg.pepsico_flag].ppg.tolist()
    st.subheader("Rival promotion calendar (locked, paper eq. 13)")
    st.caption(f"Pick each rival's promotion per week ( * = holiday week). {OPTION_HELP}")
    if rivals:
        rkey = key_of(rivals, week_labels)
        if st.session_state.get("_rival_key") != rkey:
            st.session_state._rival_df = pd.DataFrame(
                [[pe.DEFAULT_RIVAL_PLAN.get(r, {}).get(w, "NONE") for w in range(len(week_labels))] for r in rivals],
                index=rivals, columns=week_labels)
            st.session_state._rival_key = rkey
        rival_edit = st.data_editor(
            st.session_state._rival_df, key=f"rival_editor_{rkey}", width="stretch",
            column_config={c: st.column_config.SelectboxColumn(c, options=["NONE"] + ALL_ACTIVE, required=True, width="small")
                           for c in week_labels})
        rival_plan = {r: {w: rival_edit.loc[r, c] for w, c in enumerate(week_labels) if rival_edit.loc[r, c] != "NONE"}
                      for r in rivals}
    else:
        st.info("No rival products included: discount pressure comes only from PepsiCo's own products.")
        rival_plan = {}

    with st.expander("Rules JSON that will be sent to the optimiser"):
        st.json(CFG)

    if problems:
        for p_ in problems:
            st.error(p_)

# options used in the plan: NONE + allowed + anything the rivals use
used_by_rivals = {o for d in rival_plan.values() for o in d.values()}
opt_names = ["NONE"] + [o for o in ALL_ACTIVE if o in st.session_state.allowed or o in used_by_rivals]
opts = pe.DEFAULT_OPTIONS[pe.DEFAULT_OPTIONS.option.isin(opt_names)].reset_index(drop=True)
not_allowed = [o for o in used_by_rivals if o not in st.session_state.allowed]

data_params = dict(seed=int(st.session_state.seed), hist_weeks=int(st.session_state.hist_weeks),
                   noise=float(st.session_state.noise), promo_prob=float(st.session_state.promo_prob),
                   elast_mult=float(st.session_state.elast_mult), n_trials=int(st.session_state.n_trials))
input_sig = key_of(ppg, data_params, opt_names, str(st.session_state.start), st.session_state.n_weeks,
                   st.session_state.k_choice, float(st.session_state.funding), rival_plan, CFG)


# =============================================================================== run
def run_model():
    with st.status("Running PromoAI…", expanded=True) as status:
        st.write("1/5 Simulating sales history, cleaning data and training LightGBM (cached after first run)…")
        hist, dq, fc = get_forecast(ppg, **data_params)
        st.write(f"   ✓ forecast test MAPE {fc.test_mape:.1%} (legacy average {fc.legacy_mape:.1%})")
        st.write("2/5 Scoring every PPG × week × option on a grid of rival pressure and fitting PWL breakpoints…")
        fc_key = key_of(ppg, data_params)
        plan, pred, grid, curves, bp, dem, K, err_by_k, rel = get_pwl(
            fc, fc_key, ppg, opts, str(st.session_state.start), int(st.session_state.n_weeks), st.session_state.k_choice)
        st.write(f"   ✓ {K} breakpoints, mean approximation error {rel.mean():.1%}")
        inst = pe.build_instance(ppg, opts, plan, bp, dem, rival_plan, float(st.session_state.funding), pred)
        st.write("3/5 Building and solving the MILP with HiGHS…")
        sol = pe.solve_promoai(inst, CFG)
        slack = None
        if not sol.ok:
            st.write(f"   ✗ {sol.status}: re-solving with slack to find which rule cannot be met…")
            slack = pe.solve_promoai(inst, CFG, elastic=True)
        else:
            st.write(f"   ✓ {sol.status}, gap {sol.gap:.2%}, {sol.runtime:.1f}s")
        base = pe.baseline_calendar(inst)
        g = None
        if st.session_state.run_greedy:
            st.write("4/5 Running the greedy heuristic for comparison…")
            g = pe.greedy(inst, CFG)
        st.write("5/5 Re-checking every rule with the independent simulator…")
        status.update(label="Done", state="complete", expanded=False)
    st.session_state.res = dict(hist=hist, dq=dq, fc=fc, plan=plan, grid=grid, curves=curves, bp=bp, dem=dem, K=K,
                                err_by_k=err_by_k, rel=rel, inst=inst, sol=sol, slack=slack, base=base, greedy=g,
                                cfg=CFG, ppg=ppg.copy(), sig=input_sig, data_params=data_params)
    st.session_state.pop("scen", None); st.session_state.pop("mc", None); st.session_state.pop("tests", None)


if run_clicked:
    if problems:
        st.sidebar.error("Fix the input problems shown in the Inputs tab first.")
    else:
        run_model()

res = st.session_state.get("res")
if res and res["sig"] != input_sig:
    st.info("Inputs have changed since the last run. Press **▶ Run model** to update the results.", icon="🔄")
if not_allowed:
    st.caption(f"Note: rivals use {', '.join(not_allowed)}, which you did not allow for PepsiCo. It stays available only to rivals' locked slots.")


def need_run():
    st.info("Press **▶ Run model** in the sidebar to see results.", icon="👈")


# =============================================================================== tab: optimal calendar
with tabs[1]:
    if not res:
        need_run()
    else:
        inst, sol, cfg = res["inst"], res["sol"], res["cfg"]
        if not sol.ok:
            st.error(f"No calendar satisfies all the rules ({sol.status}).")
            sl = res["slack"]
            if sl is not None and sl.slack:
                st.markdown("**Rules that cannot be met, and by how much** (smallest total violation):")
                st.dataframe(pd.DataFrame({"Rule": list(sl.slack), "Shortfall": list(sl.slack.values())})
                             .style.format({"Shortfall": "{:,.1f}"}), hide_index=True, width="stretch")
                st.caption("Loosen these rules in the sidebar (or edit the rival calendar) and run again.")
            else:
                st.caption("The conflict is in a hard rule (duration, spacing, frequency or locks). Try loosening those.")
        else:
            kb = pe.evaluate(inst, cfg, res["base"])
            ko = pe.evaluate(inst, cfg, sol.calendar)
            c = st.columns(6)
            c[0].metric("Objective", f"{ko['objective']:,.0f}", f"{ko['objective'] / kb['objective'] - 1:+.1%} vs no promo")
            c[1].metric("PepsiCo sales", f"{ko['pepsico_sales']:,.0f}", f"{ko['pepsico_sales'] / kb['pepsico_sales'] - 1:+.1%}")
            c[2].metric("PepsiCo margin", f"{ko['pepsico_margin']:,.0f}", f"{ko['pepsico_margin'] / kb['pepsico_margin'] - 1:+.1%}")
            c[3].metric("Retailer margin", f"{ko['retailer_margin']:,.0f}", f"{ko['retailer_margin'] / kb['retailer_margin'] - 1:+.1%}")
            c[4].metric("Trade spend", f"{ko['trade_spend']:,.0f}", help="PepsiCo's share of the discounts it funds")
            c[5].metric("PepsiCo promotions", ko["pepsico_promotions"], help="PPG-weeks on promotion")
            st.altair_chart(calendar_chart(inst, sol.calendar, "Recommended promotion calendar (* = holiday week)"),
                            width="stretch")

            # binding rules
            notes = []
            mf = cfg.get("MarginFloor") or {}
            if mf.get("pepsico") is not None and ko["pepsico_margin_ratio"] - mf["pepsico"] < 0.002:
                notes.append(f"**PepsiCo margin floor is binding** ({ko['pepsico_margin_ratio']:.1%} vs {mf['pepsico']:.0%}): "
                             "it is what stops more promotions.")
            if mf.get("retailer") is not None and ko["retailer_margin_ratio"] - mf["retailer"] < 0.002:
                notes.append(f"**Retailer margin floor is binding** ({ko['retailer_margin_ratio']:.1%}).")
            if cfg.get("MinMarketShare") is not None and ko["pepsico_share"] - cfg["MinMarketShare"] < 0.002:
                notes.append(f"**Market-share target is binding** ({ko['pepsico_share']:.1%}).")
            if cfg.get("MaxPromoPepsiCo") is not None and ko["pepsico_promotions"] >= cfg["MaxPromoPepsiCo"]:
                notes.append("**PepsiCo promotion cap is binding.**")
            if notes:
                st.markdown("  \n".join("• " + n for n in notes))

            rows = {"No PepsiCo promotion": kb}
            if res["greedy"] is not None:
                rows["Greedy heuristic"] = pe.evaluate(inst, cfg, res["greedy"])
            rows["PromoAI MILP"] = ko
            st.markdown("##### KPI comparison (same rules, same demand)")
            st.dataframe(style_kpis(kpi_table(rows)), width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Solver report")
                lg = pe.evaluate(inst, cfg, sol.calendar, demand="lgbm")
                st.dataframe(pd.DataFrame({
                    "Item": ["Status", "Formulation", "Objective (best calendar)", "Best possible (bound)", "Gap", "Runtime",
                             "B&B nodes", "Binaries / columns / rows", "Rule violations (independent check)",
                             "Value re-scored with raw LightGBM"],
                    "Value": [sol.status, cfg["Solver"]["formulation"], f"{sol.objective:,.1f}", f"{sol.dual_bound:,.1f}",
                              f"{sol.gap:.2%}", f"{sol.runtime:.1f} s", f"{sol.nodes}",
                              f"{sol.size['binaries']:,} / {sol.size['columns']:,} / {sol.size['rows']:,}",
                              str(len(ko["violations"])), f"{lg['objective']:,.1f} ({lg['objective'] / sol.objective - 1:+.2%})"]}),
                    hide_index=True, width="stretch")
            with c2:
                st.markdown("##### Units sold per week")
                wk = pd.DataFrame({"Week": inst.week_labels * 2,
                                   "Calendar": ["No PepsiCo promotion"] * len(inst.week_labels) + ["PromoAI MILP"] * len(inst.week_labels),
                                   "Units": np.r_[kb["units_grid"].sum(0), ko["units_grid"].sum(0)]})
                ch = alt.Chart(wk).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2).encode(
                    x=alt.X("Week:N", sort=inst.week_labels, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Units:Q", title="Units", axis=alt.Axis(format=",.0f")),
                    color=alt.Color("Calendar:N", scale=alt.Scale(domain=list(CAL_COLORS)[:1] + ["PromoAI MILP"],
                                                                  range=[CAL_COLORS["No PepsiCo promotion"], CAL_COLORS["PromoAI MILP"]]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["Week", "Calendar", alt.Tooltip("Units:Q", format=",.0f")])
                st.altair_chart(ch.properties(height=300), width="stretch")

            out = pd.DataFrame(inst.opts.option.to_numpy(dtype=object)[sol.calendar], index=inst.ppg.ppg, columns=inst.week_labels)
            st.download_button("⬇ Download calendar (CSV)", out.to_csv(), "promoai_calendar.csv", "text/csv")

# =============================================================================== tab: edit & compare
with tabs[2]:
    if not res or not res["sol"].ok:
        need_run() if not res else st.warning("No optimal calendar to edit: the rules are infeasible.")
    else:
        inst, sol, cfg = res["inst"], res["sol"], res["cfg"]
        st.markdown("Override any promotion (paper requirement: *edit a promotion and see KPIs recomputed*). "
                    "Rival rows are locked and changes to them are ignored.")
        cal_df = pd.DataFrame(inst.opts.option.to_numpy(dtype=object)[sol.calendar], index=inst.ppg.ppg.tolist(),
                              columns=inst.week_labels)
        names = list(inst.opts.option)
        edited = st.data_editor(cal_df, key=f"cal_edit_{res['sig']}", width="stretch",
                                column_config={c: st.column_config.SelectboxColumn(c, options=names, required=True, width="small")
                                               for c in inst.week_labels})
        cal_new = np.array([[names.index(v) for v in row] for row in edited.values])
        pep = inst.ppg.pepsico_flag.values.astype(bool)
        cal_new[~pep] = sol.calendar[~pep]
        ko = pe.evaluate(inst, cfg, sol.calendar)
        ke = pe.evaluate(inst, cfg, cal_new)
        changed = int((cal_new != sol.calendar).sum())
        c = st.columns(4)
        c[0].metric("Cells changed", changed)
        c[1].metric("Objective", f"{ke['objective']:,.0f}", f"{ke['objective'] - ko['objective']:+,.0f} vs optimal")
        c[2].metric("PepsiCo margin %", f"{ke['pepsico_margin_ratio']:.1%}",
                    f"{100 * (ke['pepsico_margin_ratio'] - ko['pepsico_margin_ratio']):+.1f} pts")
        c[3].metric("Feasible?", "Yes ✅" if ke["feasible"] else "No ❌")
        if ke["violations"]:
            st.error("This calendar breaks the rules:\n\n" + "\n".join(f"- {v}" for v in ke["violations"]))
        elif changed:
            st.success("Your calendar respects every rule." + (" The optimiser's calendar is still better."
                                                               if ke["objective"] < ko["objective"] - 1e-6 else ""))
        st.dataframe(style_kpis(kpi_table({"PromoAI MILP": ko, "Your edited calendar": ke})), width="stretch")

# =============================================================================== tab: scenarios
with tabs[3]:
    if not res or not res["sol"].ok:
        need_run() if not res else st.warning("Fix the infeasible rules first.")
    else:
        inst, cfg = res["inst"], res["cfg"]
        st.markdown("Planners take a **menu** of calendars to the retailer (paper §2.1.4). Pick scenarios and run them; "
                    "KPIs are always reported on the same basis.")
        floor = (cfg.get("MarginFloor") or {}).get("pepsico")
        options = {
            "Revenue-max (PepsiCo sales only)": pe.with_changes(cfg, Objective={"w_pepsico": 1.0, "w_pep_margin": 0.0, "w_ret_margin": 0.0}),
            "Balanced (your current weights)": cfg,
            "Margin-preserve (margins only)": pe.with_changes(cfg, Objective={"w_pepsico": 0.5, "w_pep_margin": 1.0, "w_ret_margin": 1.0}),
            "Retailer-friendly (30% PepsiCo / 70% retailer)": pe.with_changes(cfg, Objective={**cfg["Objective"], "w_pepsico": 0.3}),
            "What-if: PepsiCo margin floor +2 pts": pe.with_changes(cfg, MarginFloor={**(cfg.get("MarginFloor") or {}), "pepsico": (floor or 0.38) + 0.02}),
            "What-if: no margin floors": pe.with_changes(cfg, MarginFloor={}),
            "What-if: max 1 price-cut promo per week": pe.with_changes(cfg, WeeklyMaxPromo=[{"data": {"mechanic_type": ["Price"]}, "week_max_promo": 1}]),
            "What-if: PepsiCo funds 30% of discounts": None,
        }
        pick = st.multiselect("Scenarios", list(options), list(options)[:3])
        tl = st.slider("Time limit per scenario (s)", 5, 120, 30)
        if st.button("Run scenarios", type="primary"):
            rows, cals = [], {}
            bar = st.progress(0.0)
            for i, name in enumerate(pick):
                c = options[name]
                if c is None:                       # funding change needs new unit economics
                    inst2 = pe.build_instance(res["ppg"], inst.opts, res["plan"], res["bp"], res["dem"],
                                              {p: {w: inst.opts.option[r] for (pi, w, r) in inst.locks if inst.ppg.ppg[pi] == p and r}
                                               for p in inst.ppg.ppg}, 0.3, inst.lgbm)
                    s = pe.solve_promoai(inst2, pe.with_changes(cfg, Solver={**cfg["Solver"], "time_limit": tl}))
                    ev = (lambda cal: pe.evaluate(inst2, cfg, cal))
                else:
                    s = pe.solve_promoai(inst, pe.with_changes(c, Solver={**c["Solver"], "time_limit": tl}))
                    ev = (lambda cal: pe.evaluate(inst, cfg, cal))
                if s.ok:
                    k = ev(s.calendar)
                    rows.append({"Scenario": name, "Status": s.status, "Gap": s.gap, "PepsiCo sales": k["pepsico_sales"],
                                 "PepsiCo margin": k["pepsico_margin"], "Retailer margin": k["retailer_margin"],
                                 "PepsiCo margin %": k["pepsico_margin_ratio"], "PepsiCo share": k["pepsico_share"],
                                 "PepsiCo promos": k["pepsico_promotions"], "Trade spend": k["trade_spend"]})
                    cals[name] = s.calendar
                else:
                    rows.append({"Scenario": name, "Status": s.status})
                bar.progress((i + 1) / len(pick))
            st.session_state.scen = (pd.DataFrame(rows), cals)
        if st.session_state.get("scen"):
            df, cals = st.session_state.scen
            st.dataframe(df.set_index("Scenario").style.format(
                {"Gap": "{:.2%}", "PepsiCo sales": "{:,.0f}", "PepsiCo margin": "{:,.0f}", "Retailer margin": "{:,.0f}",
                 "PepsiCo margin %": "{:.1%}", "PepsiCo share": "{:.1%}", "Trade spend": "{:,.0f}"}, na_rep="–"),
                width="stretch")
            ok = df.dropna(subset=["PepsiCo sales"]) if "PepsiCo sales" in df else pd.DataFrame()
            if not ok.empty:
                long = ok.melt("Scenario", ["PepsiCo sales", "PepsiCo margin", "Retailer margin"], "Measure", "Value")
                ch = alt.Chart(long).mark_bar(cornerRadiusEnd=4, height=14).encode(
                    y=alt.Y("Scenario:N", sort=list(ok.Scenario), title=None),
                    x=alt.X("Value:Q", title=None, axis=alt.Axis(format=",.0f")),
                    color=alt.value("#2a78d6"),
                    tooltip=["Scenario", "Measure", alt.Tooltip("Value:Q", format=",.0f")]
                ).properties(width=260, height=34 * len(ok)).facet(column=alt.Column("Measure:N", title=None)).resolve_scale(x="independent")
                st.altair_chart(ch)
                st.caption("Differences smaller than the solver gap are not meaningful.")
            show = st.selectbox("Show calendar for", list(cals)) if cals else None
            if show:
                st.altair_chart(calendar_chart(inst, cals[show]), width="stretch")

# =============================================================================== tab: reality simulation
with tabs[4]:
    if not res or not res["sol"].ok:
        need_run() if not res else st.warning("Fix the infeasible rules first.")
    else:
        inst, sol, cfg = res["inst"], res["sol"], res["cfg"]
        st.markdown("The optimiser plans with the **forecast**. Here each calendar is played against the **hidden true "
                    "demand** that generated the data, many times with random weekly noise (Monte Carlo). "
                    "This shows whether the plan's gain survives forecast error and bad luck.")
        c1, c2, c3 = st.columns(3)
        n_runs = c1.slider("Simulation runs", 50, 1000, 300, step=50)
        noise = c2.slider("Weekly noise (σ)", 0.0, 0.4, float(res["data_params"]["noise"]), step=0.02)
        em = c3.slider("True promotion sensitivity (× elasticity)", 0.3, 1.5, float(res["data_params"]["elast_mult"]), step=0.1,
                       help="Set below the training value to stress-test: what if shoppers respond less than the forecast thinks?")
        if st.button("Run simulation", type="primary"):
            cals = {"No PepsiCo promotion": res["base"]}
            if res["greedy"] is not None:
                cals["Greedy heuristic"] = res["greedy"]
            cals["PromoAI MILP"] = sol.calendar
            st.session_state.mc = pe.simulate_reality(inst, cfg, res["ppg"], res["plan"], cals, n_runs, noise, em)
        mc = st.session_state.get("mc")
        if mc is not None:
            wide = mc.pivot(index="run", columns="calendar", values="objective")
            base = wide["No PepsiCo promotion"]
            c = st.columns(4)
            c[0].metric("MILP mean objective (simulated)", f"{wide['PromoAI MILP'].mean():,.0f}",
                        f"{wide['PromoAI MILP'].mean() / base.mean() - 1:+.1%} vs no promo")
            c[1].metric("Runs where MILP beats no promo", f"{(wide['PromoAI MILP'] > base).mean():.0%}")
            if "Greedy heuristic" in wide:
                c[2].metric("Runs where MILP beats greedy", f"{(wide['PromoAI MILP'] > wide['Greedy heuristic']).mean():.0%}")
            planned = pe.evaluate(inst, cfg, sol.calendar)["objective"]
            c[3].metric("Plan value: forecast vs reality", f"{planned:,.0f}",
                        f"{wide['PromoAI MILP'].mean() / planned - 1:+.1%} realised vs planned", delta_color="off")
            order = [n for n in CAL_COLORS if n in set(mc.calendar)]
            ch = alt.Chart(mc).transform_density("objective", groupby=["calendar"], as_=["objective", "density"]).mark_area(
                opacity=0.55, line={"strokeWidth": 2}).encode(
                x=alt.X("objective:Q", title="Objective in simulated reality", axis=alt.Axis(format=",.0f")),
                y=alt.Y("density:Q", title=None, axis=None),
                color=alt.Color("calendar:N", scale=alt.Scale(domain=order, range=[CAL_COLORS[o] for o in order]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["calendar:N"])
            st.altair_chart(ch.properties(height=280), width="stretch")
            summ = mc.groupby("calendar")[["objective", "pepsico_sales", "pepsico_margin", "pepsico_margin_ratio"]].agg(["mean", "std"])
            summ.columns = [f"{a} ({b})" for a, b in summ.columns]
            st.dataframe(summ.loc[order].style.format("{:,.1f}").format(
                {c_: "{:.2%}" for c_ in summ.columns if "ratio" in c_}), width="stretch")
            low = (mc[mc.calendar == "PromoAI MILP"].pepsico_margin_ratio < (cfg.get("MarginFloor") or {}).get("pepsico", -1)).mean()
            if (cfg.get("MarginFloor") or {}).get("pepsico") is not None:
                st.caption(f"In {low:.0%} of runs the realised PepsiCo margin falls below the floor: the plan sits right on the "
                           "floor, so noise pushes it under about half the time. A planner could add a safety buffer.")

# =============================================================================== tab: forecast
with tabs[5]:
    if not res:
        need_run()
    else:
        fc, dq = res["fc"], res["dq"]
        c = st.columns(4)
        c[0].metric("Test MAPE (LightGBM)", f"{fc.test_mape:.1%}")
        c[1].metric("Test MAPE (legacy average)", f"{fc.legacy_mape:.1%}", f"{fc.test_mape - fc.legacy_mape:+.1%}", delta_color="inverse")
        c[2].metric("Validation MAPE", f"{fc.val_mape:.1%}")
        c[3].metric("Trees (early stopping)", fc.best_iter)
        st.caption(f"Train {fc.split['train']:,} rows · validation {fc.split['valid']:,} (tuning) · test {fc.split['test']:,} "
                   "(latest 13 weeks, never seen in training). Objective: MAPE (paper A.1).")
        c1, c2 = st.columns([3, 2])
        with c1:
            p_sel = st.selectbox("PPG", sorted(fc.test.ppg.unique()))
            t = fc.test[fc.test.ppg == p_sel].sort_values("date")
            long = t.melt(["date", "option"], ["units", "forecast", "legacy"], "Series", "Units").replace(
                {"units": "Actual", "forecast": "LightGBM", "legacy": "Legacy average"})
            dom = ["Actual", "LightGBM", "Legacy average"]
            ch = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=55, filled=True), strokeWidth=2).encode(
                x=alt.X("date:T", title=None), y=alt.Y("Units:Q", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Series:N", scale=alt.Scale(domain=dom, range=["#0b0b0b", "#2a78d6", "#eb6834"]),
                                legend=alt.Legend(orient="top", title=None)),
                strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=dom, range=[[1, 0], [1, 0], [4, 3]]), legend=None),
                tooltip=[alt.Tooltip("date:T"), "option", "Series", alt.Tooltip("Units:Q", format=",.0f")])
            st.altair_chart(ch.properties(height=300, title=f"{p_sel}: test weeks"), width="stretch")
        with c2:
            imp = fc.importance.reset_index(); imp.columns = ["Feature", "Share"]
            ch = alt.Chart(imp).mark_bar(cornerRadiusEnd=4, color="#2a78d6").encode(
                x=alt.X("Share:Q", axis=alt.Axis(format="%"), title="Share of total gain"),
                y=alt.Y("Feature:N", sort="-x", title=None), tooltip=["Feature", alt.Tooltip("Share:Q", format=".1%")])
            st.altair_chart(ch.properties(height=300, title="Feature importance"), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Data-quality report")
            st.dataframe(pd.Series(dq, name="count").to_frame(), width="stretch")
        with c2:
            st.markdown("##### Selected hyperparameters")
            st.dataframe(pd.Series({k: v for k, v in fc.params.items() if k not in ("objective", "metric", "verbosity", "seed")},
                                   name="value").to_frame().astype(str), width="stretch")
        with st.expander("All random-search trials"):
            st.dataframe(fc.trials.drop(columns=["objective", "metric", "verbosity", "seed"]), width="stretch")

# =============================================================================== tab: demand curves
with tabs[6]:
    if not res:
        need_run()
    else:
        inst, grid, curves = res["inst"], res["grid"], res["curves"]
        c = st.columns(3)
        c[0].metric("Breakpoints used (K)", res["K"])
        c[1].metric("Mean PWL error vs LightGBM", f"{res['rel'].mean():.2%}")
        c[2].metric("Max PWL error", f"{res['rel'].max():.1%}")
        c1, c2 = st.columns([3, 2])
        with c1:
            cc = st.columns(2)
            p = cc[0].selectbox("PPG ", list(inst.ppg.ppg), key="curve_ppg")
            w = cc[1].selectbox("Week", list(range(len(inst.week_labels))), format_func=lambda i: inst.week_labels[i], key="curve_wk")
            pi = list(inst.ppg.ppg).index(p)
            rows = []
            for r, o in enumerate(inst.opts.option):
                for g, v in zip(grid, curves[pi, w, r]):
                    rows.append({"Pressure": g, "Units": v, "Option": o, "Kind": "LightGBM"})
                for b, v in zip(res["bp"][pi, w], res["dem"][pi, w, r]):
                    rows.append({"Pressure": b, "Units": v, "Option": o, "Kind": "PWL"})
            df = pd.DataFrame(rows)
            dom = list(inst.opts.option)
            col = alt.Color("Option:N", scale=alt.Scale(domain=dom, range=[OPTION_COLORS[o] if o != "NONE" else "#8f8e8a" for o in dom]),
                            legend=alt.Legend(orient="top", title=None))
            lg = alt.Chart(df[df.Kind == "LightGBM"]).mark_line(strokeWidth=2).encode(
                x=alt.X("Pressure:Q", title="Rival discount pressure (avg rival discount)", axis=alt.Axis(format="%")),
                y=alt.Y("Units:Q", axis=alt.Axis(format=",.0f")), color=col,
                tooltip=["Option", alt.Tooltip("Pressure:Q", format=".0%"), alt.Tooltip("Units:Q", format=",.0f")])
            pw = alt.Chart(df[df.Kind == "PWL"]).mark_line(strokeDash=[5, 3], strokeWidth=1.5,
                                                          point=alt.OverlayMarkDef(size=70, filled=True)).encode(
                x="Pressure:Q", y="Units:Q", color=col,
                tooltip=["Option", alt.Tooltip("Pressure:Q", format=".1%"), alt.Tooltip("Units:Q", format=",.0f")])
            st.altair_chart((lg + pw).properties(height=340, title="Solid = LightGBM forecast · dashed with dots = PWL used by the MILP"),
                            width="stretch")
        with c2:
            ek = pd.DataFrame({"K": list(res["err_by_k"]), "Error": list(res["err_by_k"].values())})
            ch = alt.Chart(ek).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), color="#2a78d6", strokeWidth=2).encode(
                x=alt.X("K:O", title="Breakpoints"), y=alt.Y("Error:Q", title="Normalised squared error"),
                tooltip=["K", alt.Tooltip("Error:Q", format=".6f")])
            rule = alt.Chart(pd.DataFrame({"K": [res["K"]]})).mark_rule(color="#e34948", strokeDash=[4, 3]).encode(x="K:O")
            st.altair_chart((ch + rule).properties(height=340, title="Choosing K: knee point"), width="stretch")

# =============================================================================== tab: tests
with tabs[7]:
    st.markdown("Built-in test cases: hand-solved toy problems (model logic) and checks on your current run.")
    if st.button("Run all tests", type="primary"):
        with st.spinner("Running tests…"):
            toy = pe.run_toy_tests()
            checks = []
            if res and res["sol"].ok:
                inst, sol, cfg = res["inst"], res["sol"], res["cfg"]
                ko = pe.evaluate(inst, cfg, sol.calendar)
                lg = pe.evaluate(inst, cfg, sol.calendar, demand="lgbm")
                checks.append(("P01", "Forecast beats legacy average", f"{res['fc'].test_mape:.1%} vs {res['fc'].legacy_mape:.1%}",
                               res["fc"].test_mape < res["fc"].legacy_mape))
                checks.append(("P02", "PWL error small (< 3%)", f"{res['rel'].mean():.2%}", res["rel"].mean() < 0.03))
                checks.append(("P03", "Independent simulator finds no rule violations", f"{len(ko['violations'])} violations", ko["feasible"]))
                checks.append(("P04", "Solver objective = simulator objective", f"{sol.objective:,.2f} vs {ko['objective']:,.2f}",
                               abs(sol.objective - ko["objective"]) < 1e-5 * abs(sol.objective)))
                checks.append(("P05", "Gap within target", f"{sol.gap:.2%} (target {cfg['Solver']['mip_gap']:.1%})",
                               sol.gap <= cfg["Solver"]["mip_gap"] + 1e-9))
                checks.append(("P06", "PWL plan value within 3% of raw LightGBM", f"{lg['objective'] / sol.objective - 1:+.2%}",
                               abs(lg["objective"] / sol.objective - 1) < 0.03))
                kb = pe.evaluate(inst, cfg, res["base"])
                checks.append(("P07", "MILP beats no-promotion baseline", f"{ko['objective']:,.0f} vs {kb['objective']:,.0f}",
                               ko["objective"] >= kb["objective"] - 1e-6 or not kb["feasible"]))
                if res["greedy"] is not None:
                    kg = pe.evaluate(inst, cfg, res["greedy"])
                    checks.append(("P08", "Greedy cannot beat the MILP's proven bound", f"{kg['objective']:,.0f} ≤ {sol.dual_bound:,.0f}",
                                   kg["objective"] <= sol.dual_bound + 1e-6))
                checks.append(("P09", "Rival locks respected", f"{len(inst.locks)} slots",
                               all(sol.calendar[p, w] == r for p, w, r in inst.locks)))
                tiny = pe.tiny_slice(inst)
                if tiny is not None:
                    tcfg = pe.with_changes(cfg, MaxPromoTotal=None, MaxPromoPepsiCo=None, WeeklyMaxPromo=[], MinMarketShare=None,
                                           Solver={"mip_gap": 0.0, "time_limit": 60, "formulation": "hull"})
                    try:
                        bf, n, nf = pe.brute_force(tiny, tcfg)
                        ts = pe.solve_promoai(tiny, tcfg)
                        tb = pe.solve_promoai(tiny, pe.with_changes(tcfg, Solver={**tcfg["Solver"], "formulation": "bigm"}))
                        same = bf is not None and ts.ok and tb.ok and abs(bf[0] - ts.objective) < 1e-6 * abs(bf[0]) \
                            and abs(bf[0] - tb.objective) < 1e-6 * abs(bf[0])
                        checks.append(("P10", f"Both formulations = brute force ({n:,} calendars, {nf} feasible)",
                                       f"{bf[0]:,.4f}" if bf else "no feasible calendar", same))
                    except ValueError as e:
                        checks.append(("P10", "Brute-force cross-check", str(e), True))
            st.session_state.tests = (toy, pd.DataFrame(checks, columns=["ID", "Check", "Observed", "Pass"]))
    if st.session_state.get("tests"):
        toy, chk = st.session_state.tests
        n_pass = (toy.Result == "PASS").sum() + int(chk.Pass.sum() if len(chk) else 0)
        st.metric("Tests passed", f"{n_pass} / {len(toy) + len(chk)}")
        color = lambda v: "background-color:#d7f0e1" if v in ("PASS", True) else "background-color:#f8dcdc" if v in ("FAIL", False) else ""
        st.markdown("##### Toy problem (2 PPGs × 4 weeks, hand-solved)")
        st.dataframe(toy.style.map(color, subset=["Result"]), hide_index=True, width="stretch")
        if len(chk):
            st.markdown("##### Checks on your current run")
            st.dataframe(chk.style.map(color, subset=["Pass"]), hide_index=True, width="stretch")
        else:
            st.caption("Run the model first to add checks on your current run.")

# =============================================================================== tab: how it works
with tabs[8]:
    st.markdown("""
### What the simulator does

**1. Sales history.** Weekly sales are simulated for each product group (PPG), with holidays, seasons,
random past promotions and *rival discount pressure*. Some realistic data errors are planted and then cleaned
(paper §2.1.1). The **Market & forecast data** controls change this world.

**2. Forecast: LightGBM (paper §2.1.2, A.1).** One global model learns how units depend on discount depth,
mechanic, placement, holidays and rival pressure. It is tuned by random search on the most recent weeks with a
MAPE objective and early stopping, then scored on 13 weeks it never saw.

**3. Straight-line demand (paper §2.1.3, eq. 6).** For every PPG × week × option, the forecast is evaluated across
rival-pressure levels and replaced by a few straight segments (breakpoints chosen by SLSQP, number chosen by a knee
point). An optimiser needs straight lines.

**4. Optimiser: MILP (paper A.5).** It picks one option for every PPG-week to maximise the weighted mix of PepsiCo
and retailer sales and margin (eq. 2), subject to:
- margin floors for PepsiCo and the retailer (eq. 10)
- caps on promotions (eq. 11)
- PepsiCo's market share (eq. 12)
- locked rival promotions (eq. 13)
- the planner's templates: weekly price-cut cap, block length, spacing and frequency.

HiGHS solves it by branch and bound and stops at the chosen optimality gap. If the rules conflict, the app re-solves
with slack and tells you which rule fails and by how much.

**5. Checking.** Every plan is re-checked by an independent simulator. The **Edit & compare** tab lets you override
any cell. **Scenarios** builds the negotiation menu. **Reality simulation** plays plans against the hidden true
market with noise. **Tests** runs hand-solved cases and a brute-force optimality check.

### Formulations
- **Big-M (paper eq. 7):** demand is switched on or off with big-M inequalities.
- **Convex hull (default):** the same calendars, but a tighter relaxation, so the solver proves quality faster.

*Results show how the method behaves, not PepsiCo's numbers.*
""")
