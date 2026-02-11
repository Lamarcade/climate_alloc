# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 09:39:25 2026

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import OAS, LedoitWolf

#%% Utilities

def coerce_index_to_year_int(df) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.Index([int(str(x)[:4]) for x in df.index], name=df.index.name)
    return df.sort_index()

def levels_to_log_returns(df_levels) -> pd.DataFrame:
    df = coerce_index_to_year_int(df_levels).copy()
    if (df <= 0).any().any():
        raise ValueError("Levels must be strictly positive (log needed).")
    r = np.log(df).diff().dropna()
    r.index = pd.Index([int(x) for x in r.index])
    return r

def shrink_cov(X: np.ndarray, method: str = "oas") -> np.ndarray:
    """
    X: T x N matrix. Does not have to be centered
    Returns: N x N covariance matrix.
    """
    X = np.asarray(X)
    X = X - X.mean(axis=0, keepdims=True)
    method = method.lower()
    if method == "oas":
        return OAS(assume_centered=True).fit(X).covariance_
    if method in ("ledoit_wolf", "lw"):
        return LedoitWolf(assume_centered=True).fit(X).covariance_
    raise ValueError("method must be 'oas' or 'ledoit_wolf'")

def logpdf_mvn_chol(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    n = x.shape[0]
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
    z = np.linalg.solve(L, x - mean)
    quad = float(z.T @ z)
    logdet = 2.0 * float(np.log(np.diag(L)).sum())
    return -0.5 * (n * np.log(2.0*np.pi) + logdet + quad)

def cov_to_vol_corr(V: np.ndarray, names: list[str]):
    V = np.asarray(V)
    vol = np.sqrt(np.diag(V))
    vol_s = pd.Series(vol, index=names, name="vol")
    corr = V / np.outer(vol, vol)
    corr_df = pd.DataFrame(corr, index=names, columns=names)
    V_df = pd.DataFrame(V, index=names, columns=names)
    return V_df, vol_s, corr_df

#%% Scenarios

def prepare_scenarios_table(scenarios_df: pd.DataFrame) -> pd.DataFrame:
    scen = scenarios_df.copy()
    if "scenario" not in scen.columns:
        if "Scenario" in scen.columns:
            scen = scen.rename(columns={"Scenario": "scenario"})
        else:
            raise ValueError("Missing scenario column (expected 'Scenario' or 'scenario').")

    out = scen.set_index("scenario")

    # Rename year columns to int when possible, keep only years
    ren = {}
    year_cols = []
    for c in out.columns:
        try:
            y = int(str(c)[:4])
            ren[c] = y
            year_cols.append(y)
        except Exception:
            pass

    out = out.rename(columns=ren)
    year_cols = sorted(set(year_cols))
    if not year_cols:
        raise ValueError("No year columns found in scenarios (e.g. 2021..2050).")

    out = out[year_cols].astype(float)
    return out

#%% Scenario filtering 

def scenario_probabilities(
    r: pd.DataFrame,                # index=years(int), columns=sectors
    V: np.ndarray,                  # instantaneous covariance per year (N x N)
    scenarios_table: pd.DataFrame,  # index=scenario, columns=years(int), values=g_y^s
    dt: float = 1.0,
    prior: dict | None = None
) -> pd.DataFrame:
    r = r.copy()
    r.index = pd.Index([int(y) for y in r.index])
    years_obs = sorted(set(r.index))
    years_scen = sorted(int(y) for y in scenarios_table.columns)
    years_common = sorted(set(years_obs).intersection(years_scen))

    names = scenarios_table.index.astype(str).tolist()
    if prior is None:
        pi = {s: 1.0/len(names) for s in names}
    else:
        tot = sum(float(prior.get(s, 0.0)) for s in names)
        if tot <= 0:
            raise ValueError("Prior total mass is zero.")
        pi = {s: float(prior.get(s, 0.0))/tot for s in names}

    # No overlap => uniform
    if not years_common:
        out = pd.DataFrame({"scenario": names})
        out["log_score"] = np.log([pi[s] for s in names])
        w = np.exp(out["log_score"] - out["log_score"].max())
        out["probability"] = w / w.sum()
        out["years_used"] = 0
        out["year_min"] = np.nan
        out["year_max"] = np.nan
        return out.sort_values("probability", ascending=False).reset_index(drop=True)

    V = np.asarray(V)
    cov_dt = V * dt
    N = r.shape[1]
    ones = np.ones(N)

    rows = []
    for s in names:
        ll = 0.0
        for y in years_common:
            g = float(scenarios_table.loc[s, y])
            x = r.loc[y].to_numpy(dtype=float)
            ll += logpdf_mvn_chol(x, g * ones, cov_dt)
        ll += np.log(pi[s]) if pi[s] > 0 else -np.inf
        rows.append((s, ll))

    out = pd.DataFrame(rows, columns=["scenario", "log_score"])
    m = out["log_score"].max()
    w = np.exp(out["log_score"] - m)
    out["probability"] = w / w.sum()
    out["years_used"] = len(years_common)
    out["year_min"] = min(years_common)
    out["year_max"] = max(years_common)
    return out.sort_values("probability", ascending=False).reset_index(drop=True)

#%% GBM calibration 

def calibrate_gbm(
    df_levels: pd.DataFrame,
    dt: float = 1.0,
    cov_method: str = "oas",
    n_iter: int = 5,
) -> dict:
    """
    dE/E = mu_t * 1 dt + Sigma dW
    - mu_t scalar per time step (common across sectors)
    - V = Sigma Sigma^T constant, estimated with shrinkage
    """
    r = levels_to_log_returns(df_levels)  # T x N
    years = r.index
    cols = r.columns
    T, N = r.shape

    # init V
    S = shrink_cov(r.values, method=cov_method)  # cov of returns over dt
    V = S / dt

    mu_t = pd.Series(index=years, dtype=float, name="mu_t")
    mus_conv = pd.DataFrame(index=years)

    for i in range(n_iter):
        avg_diagV = float(np.mean(np.diag(V)))
        r_bar = r.mean(axis=1)
        mu_t = (r_bar / dt) + 0.5 * avg_diagV

        mean_matrix = (mu_t.values[:, None] - 0.5 * np.diag(V)[None, :]) * dt
        eps = r.values - mean_matrix

        S = shrink_cov(eps, method=cov_method)
        V = S / dt
        mus_conv[f"iter{i}"] = mu_t

    V_df, vol, corr = cov_to_vol_corr(V, list(cols))
    Sigma_chol = pd.DataFrame(
        np.linalg.cholesky(V + 1e-12 * np.eye(N)),
        index=cols, columns=cols
    )

    return {
        "mu_t": mu_t,
        "V": V_df,
        "corr": corr,
        "vol": vol,
        "Sigma_chol": Sigma_chol,
        "log_returns": r,
        "mus_conv": mus_conv
    }

#%% Sequential recalibration 

def sequential_recalibration(
    df_levels: pd.DataFrame,
    scenarios_table: pd.DataFrame,
    start_calib_year: int = 2020,
    end_year: int | None = None,
    cov_method: str = "oas",
    dt: float = 1.0,
    prior: dict | None = None
):
    df_levels = coerce_index_to_year_int(df_levels)
    all_years = sorted(df_levels.index.astype(int).tolist())
    if end_year is None:
        end_year = max(all_years)

    update_years = [y for y in all_years if start_calib_year <= y <= end_year]
    if not update_years:
        raise ValueError("No update years available with given start/end years.")

    covs, corrs = {}, {}
    vol_rows, probs_rows = [], []

    for T in update_years:
        df_T = df_levels.loc[df_levels.index <= T]
        r_T = levels_to_log_returns(df_T)

        if r_T.shape[0] < 2:
            out_probs = pd.DataFrame({"scenario": scenarios_table.index.astype(str)})
            out_probs["log_score"] = 0.0
            out_probs["probability"] = 1.0 / len(out_probs)
            out_probs["years_used"] = 0
            out_probs["year_min"] = np.nan
            out_probs["year_max"] = np.nan
        else:
            S = shrink_cov(r_T.values, method=cov_method)
            V = S / dt

            V_df, vol_s, corr_df = cov_to_vol_corr(V, list(r_T.columns))
            covs[T] = V_df
            corrs[T] = corr_df
            vol_rows.append(pd.Series(vol_s.values, index=vol_s.index, name=T))

            out_probs = scenario_probabilities(r_T, V, scenarios_table, dt=dt, prior=prior)

        out_probs = out_probs.copy()
        out_probs.insert(0, "update_year", T)
        probs_rows.append(out_probs)

    probs_long = pd.concat(probs_rows, ignore_index=True)
    vols_df = pd.DataFrame(vol_rows) if vol_rows else pd.DataFrame()
    return probs_long, covs, corrs, vols_df

#%% Simulation by drawing scenarios

def prepare_probabilities(probs_df: pd.DataFrame) -> pd.Series:
    if "scenario" not in probs_df.columns or "probability" not in probs_df.columns:
        raise ValueError("probs_df must contain 'scenario' and 'probability'.")
    p = probs_df.set_index("scenario")["probability"].astype(float)
    p = p / p.sum()
    return p

def simulate_paths(
    e_T: pd.Series,
    V: pd.DataFrame | np.ndarray,
    scenarios_table: pd.DataFrame,
    probs: pd.Series,
    years: list[int] | None = None,
    n_sims: int = 10000,
    seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)

    sectors = list(e_T.index)
    N = len(sectors)

    V_np = V.values if isinstance(V, pd.DataFrame) else np.asarray(V)
    if V_np.shape != (N, N):
        raise ValueError(f"V must be shape ({N},{N}) consistent with e_T sectors.")
    L = np.linalg.cholesky(V_np + 1e-12 * np.eye(N))

    # align probs to scenarios_table
    probs = probs.reindex(scenarios_table.index).fillna(0.0)
    probs = probs / probs.sum()

    all_years = [int(y) for y in scenarios_table.columns]
    if years is None:
        years = all_years
    else:
        years = [int(y) for y in years]
        missing = sorted(set(years) - set(all_years))
        if missing:
            raise ValueError(f"Scenarios missing years: {missing}")

    scen_names = probs.index.to_numpy()
    pvals = probs.to_numpy()
    scen_draws = rng.choice(scen_names, size=n_sims, p=pvals)

    paths = np.zeros((n_sims, len(years) + 1, N), dtype=float)
    paths[:, 0, :] = e_T.values

    for k, y in enumerate(years, start=1):
        g = scenarios_table.loc[scen_draws, y].to_numpy(dtype=float)  # (n_sims,)
        z = rng.standard_normal((n_sims, N))
        eps = z @ L.T
        r = g[:, None] + eps
        paths[:, k, :] = paths[:, k-1, :] * np.exp(r)

    return {"paths": paths, "scen_draws": scen_draws, "years": years, "sectors": sectors}

def summarize_paths(paths: np.ndarray, years: list[int], sectors: list[str], q=(0.05, 0.5, 0.95)):
    idx = ["T0"] + [str(y) for y in years]
    mean = paths.mean(axis=0)
    df_mean = pd.DataFrame(mean, index=idx, columns=sectors)
    quantiles = {qq: pd.DataFrame(np.quantile(paths, qq, axis=0), index=idx, columns=sectors) for qq in q}
    return df_mean, quantiles

def summarize_total(paths, years, q=(0.05, 0.95)):
    total = paths.sum(axis=2)
    idx = ["T0"] + [str(y) for y in years]
    mean = total.mean(axis=0)
    out_mean = pd.Series(mean, index=idx, name="Total")
    out_q = {qq: pd.Series(np.quantile(total, qq, axis=0), index=idx, name=f"q{int(100*qq)}") for qq in q}
    return out_mean, out_q

#%% Plot

def plot_sector_paths_with_bands(df_mean, qs, title=None, q_low=0.05, q_high=0.95,
                                 ncols=4, figsize=(18, 10), sharex=True):
    if q_low not in qs or q_high not in qs:
        raise ValueError(f"qs must contain keys {q_low} and {q_high}. Found: {list(qs.keys())}")

    df_qlo = qs[q_low]
    df_qhi = qs[q_high]

    sectors = list(df_mean.columns)
    n = len(sectors)
    nrows = int(np.ceil(n / ncols))

    idx = list(df_mean.index)
    years = np.array([int(y) for y in idx[1:]], dtype=int)
    year0 = years[0] - 1
    x = np.concatenate([[year0], years])

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex)
    axes = np.array(axes).reshape(-1)

    for k, sec in enumerate(sectors):
        ax = axes[k]
        y_mean = df_mean[sec].to_numpy(float)
        y_lo   = df_qlo[sec].to_numpy(float)
        y_hi   = df_qhi[sec].to_numpy(float)

        ax.fill_between(x, y_lo, y_hi, color="C0", alpha=0.18, linewidth=0)
        ax.plot(x, y_mean, color="C0", linewidth=2.0, label="moyenne")
        ax.plot(x, y_lo, color="C0", linewidth=1.0, alpha=0.6, linestyle="--", label=f"q{int(100*q_low)}")
        ax.plot(x, y_hi, color="C0", linewidth=1.0, alpha=0.6, linestyle="--", label=f"q{int(100*q_high)}")

        ax.set_title(sec, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.ticklabel_format(axis='y', style='sci', useOffset=False)

        if k % ncols == 0:
            ax.set_ylabel("Émissions")
        if k >= (nrows - 1) * ncols:
            ax.set_xlabel("Année")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="upper center", ncol=3, frameon=False)

    if title is None:
        title = f"Trajectoires simulées par secteur (bande {int(100*q_low)}%-{int(100*q_high)}%)"
    fig.suptitle(title, y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

#%% Application

df = pd.read_excel("Data/history_filtered.xlsx")
df_E = df.set_index("GICS Sector Name").T
df_E = coerce_index_to_year_int(df_E)

scenarios_df = pd.read_excel("Data/logscenarios.xlsx")
scen_table = prepare_scenarios_table(scenarios_df)

# 1) sequential recalibration
probs_long, covs, corrs, vols = sequential_recalibration(df_E, scen_table, start_calib_year=2020, cov_method="oas")

# 2) take last available year T and simulate mixture
T = int(df_E.index.max())
e_T = df_E.loc[T]
V_T = covs[T]
probs_T = probs_long.loc[probs_long["update_year"] == T, ["scenario", "probability"]]
probs = prepare_probabilities(probs_T)

years_future = [y for y in scen_table.columns if y >= 2024]
sim = simulate_paths(e_T, V_T, scen_table, probs, years=years_future, n_sims=10000, seed=42)

df_mean, qs = summarize_paths(sim["paths"], sim["years"], sim["sectors"])
plot_sector_paths_with_bands(df_mean, qs, title="Trajectoires par secteur – mélange de scénarios")
plt.show()

tot_mean, tot_q = summarize_total(sim["paths"], sim["years"], q=(0.05, 0.95))
years = np.array([int(y) for y in tot_mean.index[1:]])
x = np.concatenate([[years[0]-1], years])
plt.figure(figsize=(10,5))
plt.fill_between(x, tot_q[0.05].values, tot_q[0.95].values, alpha=0.2)
plt.plot(x, tot_mean.values, linewidth=2)
plt.title("Total émissions – moyenne + q5/q95")
plt.xlabel("Année"); plt.ylabel("Émissions"); plt.grid(alpha=0.3)
plt.show()