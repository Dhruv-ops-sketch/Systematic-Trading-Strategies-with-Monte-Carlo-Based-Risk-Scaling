import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = "/Users/dk5230/Downloads/aapl_us_d.csv"
LEVERAGE = 1.5
ONE_WAY_COST_BPS = 5
SLIPPAGE_BPS = 2
FAST_MA = 20
SLOW_MA = 100
VOL_SHORT = 20
VOL_LONG = 120
VOL_SPIKE_MULT = 2.0
PEAK_LOOKBACK = 252
DROP_FROM_PEAK = 0.12
TREND_SIZING_SCALE = 0.06
MAX_EXPOSURE = 1.0
VOL_TARGET_ANNUAL = 0.20 
VOL_TARGET_WINDOW = 20
TRAIN_WINDOW_DAYS = 504
HORIZON_DAYS = 5
N_PATHS = 12000
SEED = 42
EWMA_LAMBDA_LOW = 0.97
EWMA_LAMBDA_HIGH = 0.94
JUMP_PROB_LOW = 0.01
JUMP_PROB_HIGH = 0.03
JUMP_VOL_MULT_LOW = 3.0
JUMP_VOL_MULT_HIGH = 4.0
MC_MULT_MIN = 0.5
MC_MULT_MAX = 1.5
MC_NEUTRAL = 1.0
MC_SENSITIVITY = 0.8
PLOT_EQUITY = True
LOG_SCALE = True

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def sharpe(daily_returns: pd.Series) -> float:
    s = daily_returns.std(ddof=1)
    if s == 0 or np.isnan(s):
        return 0.0
    return float(np.sqrt(252) * daily_returns.mean() / s)

def summarize(name: str, df_local: pd.DataFrame, ret_col: str, eq_col: str, pos_col: str):
    days = len(df_local)
    cagr = float(df_local[eq_col].iloc[-1] ** (252 / days) - 1)
    sh = sharpe(df_local[ret_col])
    mdd = max_drawdown(df_local[eq_col])
    hit = float((df_local[ret_col] > 0).mean()) * 100.0
    turnover = float(df_local[pos_col].diff().abs().fillna(0).mean())
    avg_abs_pos = float(df_local[pos_col].abs().mean())
    return name, cagr, sh, mdd, hit, turnover, avg_abs_pos

def ewma_sigma(returns: np.ndarray, lam: float) -> float:
    var = np.var(returns, ddof=1) if len(returns) > 2 else float(returns[-1] ** 2)
    for x in returns:
        var = lam * var + (1 - lam) * (x ** 2)
    return float(np.sqrt(max(var, 1e-12)))

def simulate_horizon(mu_d, sigma_d, horizon, n_paths, jump_prob, jump_mult, rng):
    z = rng.standard_normal((horizon, n_paths))
    jumps = rng.random((horizon, n_paths)) < jump_prob
    vol = np.where(jumps, sigma_d * jump_mult, sigma_d)
    r = mu_d + vol * z
    return r.sum(axis=0)

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower() for c in df.columns]

date_candidates = ["date", "data", "timestamp", "time"]
date_col = next((c for c in date_candidates if c in df.columns), df.columns[0])
df = df.rename(columns={date_col: "date"})
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

close_candidates = ["close", "zamkniecie", "adj close", "adjusted close", "price", "last"]
close_col = next((c for c in close_candidates if c in df.columns), None)
if close_col is None:
    raise ValueError(f"No close-like column found. Columns: {df.columns.tolist()}")

df["price"] = df[close_col].astype(float)
df = df.loc[df["price"].notna()].copy()

df["r"] = np.log(df["price"]).diff()
df = df.dropna().reset_index(drop=True)

out = df.copy()

cost_per_turn = (ONE_WAY_COST_BPS + SLIPPAGE_BPS) / 10000.0
out["pos_bh"] = 1.0
out["ret_bh"] = out["r"]
out["eq_bh"] = np.exp(out["ret_bh"].cumsum())
out["ma_fast"] = out["price"].rolling(FAST_MA).mean()
out["ma_slow"] = out["price"].rolling(SLOW_MA).mean()
out["trend_ok"] = (out["ma_fast"] > out["ma_slow"]).astype(float)
out["trend_strength"] = (out["ma_fast"] / out["ma_slow"]) - 1.0  # fraction
out["vol_short"] = out["r"].rolling(VOL_SHORT).std()
out["vol_long"] = out["r"].rolling(VOL_LONG).std()
out["panic_vol"] = (out["vol_short"] > (VOL_SPIKE_MULT * out["vol_long"])).astype(float)
out["recent_peak"] = out["price"].rolling(PEAK_LOOKBACK).max()
out["dd_from_peak"] = (out["price"] / out["recent_peak"]) - 1.0
out["panic_dd"] = (out["dd_from_peak"] <= -DROP_FROM_PEAK).astype(float)
out["panic"] = ((out["panic_vol"] == 1.0) | (out["panic_dd"] == 1.0)).astype(float)
trend_size = (out["trend_strength"] / TREND_SIZING_SCALE).clip(lower=0.0, upper=MAX_EXPOSURE)
trend_size = trend_size.fillna(0.0)
realized_daily = out["r"].rolling(VOL_TARGET_WINDOW).std()
realized_annual = (realized_daily * np.sqrt(252)).clip(lower=1e-6)
vol_scale = (VOL_TARGET_ANNUAL / realized_annual).clip(lower=0.0, upper=2.0)  # cap so it doesn't explode
vol_scale = vol_scale.fillna(0.0)
out["baseline_size"] = (trend_size * vol_scale).clip(0.0, MAX_EXPOSURE)
out["baseline_pos"] = out["baseline_size"] * out["trend_ok"] * (1.0 - out["panic"])
out["baseline_pos"] = out["baseline_pos"].fillna(0.0)

rng = np.random.default_rng(SEED)
n = len(out)
mc_mult = np.full(n, MC_NEUTRAL)

start_t = max(SLOW_MA + 5, VOL_LONG + 5, PEAK_LOOKBACK + 5, TRAIN_WINDOW_DAYS + 5)

for t in range(start_t, n - HORIZON_DAYS):
    if out["panic"].iloc[t] == 1.0 or out["trend_ok"].iloc[t] == 0.0:
        mc_mult[t] = MC_NEUTRAL
        continue

    window = out["r"].iloc[t - TRAIN_WINDOW_DAYS:t].values
    mu_d = float(np.mean(window))

    recent = out["r"].iloc[t - 63:t].values
    recent_vol = float(np.std(recent, ddof=1))
    vol_q = float(np.quantile(np.abs(window), 0.70))

    if recent_vol <= vol_q:
        lam = EWMA_LAMBDA_LOW
        jump_p = JUMP_PROB_LOW
        jump_mult = JUMP_VOL_MULT_LOW
    else:
        lam = EWMA_LAMBDA_HIGH
        jump_p = JUMP_PROB_HIGH
        jump_mult = JUMP_VOL_MULT_HIGH

    sigma_d = ewma_sigma(window, lam)

    sim_logR = simulate_horizon(mu_d, sigma_d, HORIZON_DAYS, N_PATHS, jump_p, jump_mult, rng)
    sim_simpleR = np.expm1(sim_logR)

    e = float(np.mean(sim_simpleR))
    sd = float(np.std(sim_simpleR, ddof=1))
    sd = max(sd, 1e-6)

    required = 2 * cost_per_turn
    edge = e - required

    score = edge / sd


    raw = 1.0 + MC_SENSITIVITY * np.tanh(score)
    mc_mult[t] = float(np.clip(raw, MC_MULT_MIN, MC_MULT_MAX))

out["mc_mult"] = pd.Series(mc_mult, index=out.index).ffill().fillna(MC_NEUTRAL)

out["pos_layered"] = (LEVERAGE * out["baseline_pos"] * out["mc_mult"]).clip(0.0, LEVERAGE)
out["pos_layered"] = out["pos_layered"].fillna(0.0)

def apply_costs_and_equity(df_, pos_col, name_prefix):
    df_[f"ret_{name_prefix}_gross"] = df_[pos_col].shift(1).fillna(0.0) * df_["r"]
    df_[f"turn_{name_prefix}"] = df_[pos_col].diff().abs().fillna(0.0)
    df_[f"cost_{name_prefix}"] = df_[f"turn_{name_prefix}"] * cost_per_turn
    df_[f"ret_{name_prefix}"] = df_[f"ret_{name_prefix}_gross"] - df_[f"cost_{name_prefix}"]
    df_[f"eq_{name_prefix}"] = np.exp(df_[f"ret_{name_prefix}"].cumsum())

apply_costs_and_equity(out, "baseline_pos", "base")
apply_costs_and_equity(out, "pos_layered", "layered")


print("\nCOMPARISON")
for (name, retcol, eqcol, poscol) in [
    ("Buy&Hold", "ret_bh", "eq_bh", "pos_bh"),
    ("Baseline (Trend + VolTarget + Panic)", "ret_base", "eq_base", "baseline_pos"),
    ("Layered (Baseline × MC multiplier)", "ret_layered", "eq_layered", "pos_layered"),
]:
    nm, cagr, sh, mdd, hit, turn, avgpos = summarize(name, out, retcol, eqcol, poscol)
    print(f"{nm}")
    print(f"  CAGR:        {100*cagr:.2f}%")
    print(f"  Sharpe:      {sh:.2f}")
    print(f"  Max Drawdown:{100*mdd:.2f}%")
    print(f"  Hit Rate:    {hit:.1f}% of days positive")
    print(f"  Avg turnover:{turn:.4f} (pos change/day)")
    print(f"  Avg |pos|:   {avgpos:.3f}")
    print("------------------------------------------------")
print("================================================\n")

if PLOT_EQUITY:
    plt.figure(figsize=(13, 5))
    plt.plot(out["date"], out["eq_bh"], label="Buy & Hold")
    plt.plot(out["date"], out["eq_base"], label="Baseline (Trend+VolTarget+Panic)")
    plt.plot(out["date"], out["eq_layered"], label=f"Layered × {LEVERAGE}x Leverage")

    plt.title("Equity Curves (net of costs)")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.grid(alpha=0.25)
    plt.legend()

    if LOG_SCALE:
        plt.yscale("log")
        plt.ylabel("Equity (log scale, normalized)")

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.show()

panic_rate = float(out["panic"].mean()) * 100.0
print(f"Panic triggered on ~{panic_rate:.2f}% of days.")
print(f"Average MC multiplier: {out['mc_mult'].mean():.3f}  (min={out['mc_mult'].min():.3f}, max={out['mc_mult'].max():.3f})")
print(f"Average baseline exposure: {out['baseline_pos'].abs().mean():.3f}")
print(f"Average layered exposure:  {out['pos_layered'].abs().mean():.3f}")
