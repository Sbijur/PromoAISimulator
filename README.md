# PromoAI simulator (Streamlit)

Interactive simulator of the PromoAI promotion-calendar model from Llenas et al. (2026),
*PepsiCo Deploys AI-Driven Pricing and Promotion Optimization at Scale* (arXiv:2606.17941).

Pipeline: synthetic retailer history → data cleaning → global LightGBM forecast (MAPE) →
piecewise-linear demand in rival discount pressure → promotion-calendar MILP (HiGHS) → KPIs.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. Keep `app.py` and `promo_engine.py` in the same folder.

## What users can change

| Where | Controls |
|---|---|
| Sidebar ① Objective weights | PepsiCo vs retailer weight, margin-vs-sales weight for each party (sliders) |
| Sidebar ② Business rules | Margin floors, market share, promotion caps, weekly price-cut cap, block length, spacing, promo weeks per PPG (sliders + on/off toggles) |
| Sidebar ③ Market & forecast data | Seed, weeks of history, noise, promotion frequency, promotion sensitivity, number of LightGBM search trials |
| Sidebar ④ Horizon, options & money | First week, weeks to plan, allowed promotion options, PepsiCo funding share, number of PWL breakpoints |
| Sidebar ⑤ Solver | Optimality gap, time limit, formulation (convex hull or paper Big-M), greedy comparison |
| Sidebar | Upload / download rules JSON (paper Appendix B format), reset to defaults |
| Inputs tab | Editable product table (prices, costs, base volume, elasticity, segment, owner; add/remove rows), rival calendar (dropdown per week) |
| Edit & compare tab | Override any cell of the optimal calendar and see KPIs and rule violations recomputed |

## Tabs

Optimal calendar · Edit & compare · Scenarios (negotiation menu and what-ifs) · Reality simulation
(Monte Carlo against the hidden true demand) · Forecast · Demand curves · Tests (hand-solved toy cases,
brute-force optimality check, checks on the current run) · How it works.

The first run trains the forecast (≈5–20 s); after that it is cached unless the market/data inputs change.
A solve takes ≈10–60 s depending on the rules and gap. All data are synthetic.
