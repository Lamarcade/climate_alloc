# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:39:01 2026

@author: LoïcMARCADET
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import multivariate_normal
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm
from matplotlib.lines import Line2D

import Config

from sklearn.covariance import LedoitWolf

#%% Test mapping

HIST_TO_SCEN = {
    "Industrials": "Industry",
    "Energy": "Electricity",
    "Utilities": "Supply",
    "Materials": "Transportation",
    "Consumer Discretionary": "AFOLU",
    "Consumer Staples": "Steel",
    "Health Care": "Other Industry",
    "Communication Services": "Other Energy Supply",
    "Financials": "Other",
    "Information Technology": "Cement",
    "Real Estate": "Chemicals",
}

SCEN_TO_HIST = {v: k for k, v in HIST_TO_SCEN.items()}

#%%

folder = "Data/NGFS5/"
file = "IAM_data.xlsx"

ngfs = pd.read_excel(folder + file)

# Approximation EU-15, lacks Norway, Switzerland and Poland for CAC40
filtered = ngfs[ngfs["Region"] == "GCAM 6.0 NGFS|EU-15"]

indic = "Kyoto"
filtered = filtered[filtered["Variable"].str.contains(indic, case=False, na=False)]

def reshape(df):

    df["Sector"] = df["Variable"].apply(lambda x: x.split("|")[-1] if "|" in x else "Global")
    
    year_cols = [col for col in df.columns if col.isdigit()]
    cols_to_keep = ["Model", "Scenario", "Sector"] + year_cols
    
    df_filtered = df[cols_to_keep].copy()
    
    return df_filtered

sectors = reshape(filtered)

def interpolate_years(df, start=2020, end=2050):
    context_cols = ["Scenario", "Sector"]
    
    year_cols = sorted([col for col in df.columns if col.isdigit() and start <= int(col) <= end])
    target_years = [str(y) for y in range(start, end + 1)]
    
    def interpolate_row(row):
        series = row[year_cols].astype(float)
        interpolated = pd.Series(index=target_years, dtype=float)
        interpolated.loc[year_cols] = series.values
        interpolated = interpolated.interpolate(method='linear')
        return interpolated
    
    interpolated_df = df.apply(interpolate_row, axis=1)
    
    final_df = pd.concat([df[context_cols].reset_index(drop=True), interpolated_df.reset_index(drop=True)], axis=1)
    return final_df

kyoto = interpolate_years(sectors)

#%% Modify emissions

clipped = kyoto.copy()
year_cols = sorted([col for col in clipped.columns if col.isdigit()])
mask = clipped[year_cols] < 0

clipped[mask] = 0
clipped = clipped[clipped["Sector"]!= "Kyoto Gases"]

#%% History 

hp = pd.read_excel("Data/history_processed.xlsx", index_col = 0)

def filter_companies(df,start_year, end_year=2023,
    instrument_col="Instrument", year_col="Year",scope_col="Scope12"):

    df = df.copy()

    df[year_col] = df[year_col].astype(int)

    df_period = df[
        (df[year_col] >= start_year) &
        (df[year_col] <= end_year)
    ]

    required_years = set(range(start_year, end_year + 1))

    valid_instruments = []

    for inst, g in df_period.groupby(instrument_col):
        years_with_scope = set(
            g.loc[g[scope_col].notna(), year_col]
        )

        if required_years.issubset(years_with_scope):
            valid_instruments.append(inst)

    df_filtered = df_period[
        df_period[instrument_col].isin(valid_instruments)
    ]
    
    df_filtered[year_col] = df_filtered[year_col].astype(str)

    return df_filtered
    
df_fi = filter_companies(hp, 2018)

emission_cols = ['CO2 Equivalent Emissions Direct, Scope 1', 'CO2 Equivalent Emissions Indirect, Scope 2', 'Scope12']

keep_cols = ['GICS Sector Name'] + emission_cols

last = df_fi[df_fi["Year"] == '2023'][keep_cols]

last = last.groupby('GICS Sector Name').sum().sort_values('Scope12', ascending = False)

grouped = (
    df_fi.groupby(["GICS Sector Name", "Year"])["Scope12"]
    .sum()
    .reset_index()
)

pivot = grouped.pivot(index="GICS Sector Name", columns="Year", values="Scope12")

pivot = pivot.loc[:, [y for y in pivot.columns if int(y) <= 2023]]

pivot.columns = pivot.columns.astype(str)

pivot = pivot.reindex(last.index)

pivot.index = last.index

#%%
# Calibration

def calibrate_sigma_and_corr_from_history(df_hist_wide: pd.DataFrame, shrinkage: str | None = "ledoit_wolf"):
    df = df_hist_wide.copy()
    df.columns = pd.to_numeric(df.columns, errors="ignore")
    df = df.reindex(sorted(df.columns), axis=1).astype(float)

    log_returns = np.log(df).diff(axis=1).iloc[:, 1:]   # index=sectors, column = years
    sigma_hist = log_returns.std(axis=1, ddof=1)

    X = log_returns.T   # rows = year, cols = sectors
    if shrinkage == "ledoit_wolf":
        lw = LedoitWolf().fit(X.values)
        cov = pd.DataFrame(lw.covariance_, index=X.columns, columns=X.columns)
    else:
        cov = X.cov()

    d = np.sqrt(np.diag(cov.values))
    corr_hist = cov.div(d, axis=0).div(d, axis=1)
    corr_hist = corr_hist.fillna(0.0)
    corr_hist = (corr_hist + corr_hist.T) / 2 #stabilize
    np.fill_diagonal(corr_hist.values, 1.0)

    return sigma_hist, corr_hist, log_returns

def map_history_params_to_scenario_sectors(
    sigma_hist: pd.Series,
    corr_hist: pd.DataFrame,
    scenario_sectors: list[str],
    scen_to_hist: dict[str, str]
):

    hist_names = [scen_to_hist[s] for s in scenario_sectors]

    sigma_scen = pd.Series(
        [sigma_hist.loc[h] for h in hist_names],
        index=scenario_sectors,
        name="sigma"
    )

    corr_scen = pd.DataFrame(
        [[corr_hist.loc[h1, h2] for h2 in hist_names] for h1 in hist_names],
        index=scenario_sectors,
        columns=scenario_sectors
    )

    corr_scen = (corr_scen + corr_scen.T) / 2
    np.fill_diagonal(corr_scen.values, 1.0)

    return sigma_scen, corr_scen


# Choose a scenario
def prepare_scenario_wide(df_scen: pd.DataFrame, scenario: str) -> pd.DataFrame:

    df = df_scen[df_scen["Scenario"].eq(scenario)].copy()
    if df.empty:
        raise ValueError(f"No scenario '{scenario}'")

    year_cols = [c for c in df.columns if str(c).isdigit()]
    years = sorted(int(c) for c in year_cols)

    selected_cols = []
    for y in years:
        selected_cols.append(y if y in df.columns else str(y))

    wide = df.set_index("Sector")[selected_cols].copy()
    wide.columns = years
    wide = wide.astype(float)
    return wide

def simulate_sector_paths(
    wide_scen: pd.DataFrame,
    sigma_scen: pd.Series,
    corr_scen: pd.DataFrame,
    n_paths: int = 1000,
    seed: int | None = 123,
    zero_absorbing: bool = True,
    floor: float = 0.0,
    vol_amplification: float = 1.0,
):
    """
      E_{i,t+1} = E_{i,t} * g_{i,t} * exp(-0.5*sigma_i^2 + sigma_i * Z_{i,t})
    with correlated Z.  sigma is multiplied by vol_amplification (>1 = more variance).
    """
    sectors = list(wide_scen.index)
    years = list(wide_scen.columns)

    sigma = sigma_scen.reindex(sectors).astype(float)
    corr = corr_scen.reindex(index=sectors, columns=sectors).astype(float)

    if sigma.isna().any():
        raise ValueError(f"Missing sigma: {sigma[sigma.isna()].index.tolist()}")
    if corr.isna().any().any():
        raise ValueError("Incomplete correlation.")

    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr.values, 1.0)

    eps = 1e-12
    L = np.linalg.cholesky(corr.values + eps * np.eye(len(sectors)))

    scen = wide_scen.astype(float)

    # Ratios g_{i,t} = scen_{t+1} / scen_t
    g = scen.shift(-1, axis=1).iloc[:, :-1].div(scen.iloc[:, :-1])
    g.columns = years[1:]

    rng = np.random.default_rng(seed)

    n_sectors = len(sectors)
    n_years = len(years)

    paths = np.zeros((n_paths, n_sectors, n_years), dtype=float)
    paths[:, :, 0] = scen[years[0]].values[None, :]

    sig = sigma.values * vol_amplification

    for p in range(n_paths):
        for t in range(1, n_years):
            z = L @ rng.standard_normal(n_sectors)
            prev = paths[p, :, t - 1]
            gt = g[years[t]].reindex(sectors).values
            gt = np.where(np.isfinite(gt), gt, 0.0)

            step = prev * gt * np.exp(-0.5 * sig**2 + sig * z)

            if zero_absorbing:
                step = np.where(prev <= 0.0, 0.0, step)
                step = np.where(gt <= 0.0, 0.0, step)

            if floor is not None:
                step = np.maximum(step, floor)

            paths[p, :, t] = step

    idx = pd.MultiIndex.from_product([range(n_paths), sectors], names=["path", "Sector"])
    df_paths = pd.DataFrame(
        paths.reshape(n_paths * n_sectors, n_years),
        index=idx,
        columns=years
    )

    return {
        "paths": df_paths,
        "g": g,
        "sigma": sigma * vol_amplification,
        "corr": corr,
        "vol_amplification": vol_amplification,
    }

def run_simulation_from_history_and_scenario(
    df_hist_wide: pd.DataFrame,
    df_scen: pd.DataFrame,
    scenario_name: str,
    scen_to_hist: dict[str, str],
    n_paths: int = 1000,
    seed: int | None = 123,
    shrinkage: str | None = "ledoit_wolf",
    vol_amplification: float = 1.0,
):
    # calibration 
    sigma_hist, corr_hist, log_returns = calibrate_sigma_and_corr_from_history(
        df_hist_wide,
        shrinkage=shrinkage
    )

    # scenario selection
    wide_scen = prepare_scenario_wide(df_scen, scenario_name)

    # mapping 
    sigma_scen, corr_scen = map_history_params_to_scenario_sectors(
        sigma_hist=sigma_hist,
        corr_hist=corr_hist,
        scenario_sectors=list(wide_scen.index),
        scen_to_hist=scen_to_hist
    )

    # simulation
    out = simulate_sector_paths(
        wide_scen=wide_scen,
        sigma_scen=sigma_scen,
        corr_scen=corr_scen,
        n_paths=n_paths,
        seed=seed,
        vol_amplification=vol_amplification,
    )

    out["sigma_hist"] = sigma_hist
    out["corr_hist"] = corr_hist
    out["log_returns_hist"] = log_returns
    out["wide_scenario"] = wide_scen

    return out


#%%
scen_name = "Net Zero 2050"
VOL_AMPLIFICATION = 1.0

out = run_simulation_from_history_and_scenario(
    df_hist_wide=pivot,
    df_scen=clipped,
    scenario_name=scen_name,
    scen_to_hist=SCEN_TO_HIST,
    n_paths=500,
    seed=42,
    shrinkage="ledoit_wolf",
    vol_amplification=VOL_AMPLIFICATION,
)

paths = out["paths"]
sigma_used = out["sigma"]
corr_used = out["corr"]

mean_path = paths.groupby("Sector").mean()

q05 = paths.groupby("Sector").quantile(0.05)
q50 = paths.groupby("Sector").quantile(0.50)
q95 = paths.groupby("Sector").quantile(0.95)

print(mean_path.head())
print(sigma_used)
print(corr_used.iloc[:5, :5])

#%% 

def plot_sector_trajectories_with_bands(
    paths: pd.DataFrame,
    wide_scenario: pd.DataFrame,
    sectors: list[str] | None = None,
    q_low: float = 0.05,
    q_high: float = 0.95,
    ncols: int = 2,
    figsize_per_subplot=(7, 4),
    scenario_color: str = "red",
    median_color: str = "black",
    band_color: str = "steelblue",
    suptitle: str | None = None,
):

    available_sectors = list(wide_scenario.index)

    if sectors is None:
        sectors = available_sectors
    else:
        missing = [s for s in sectors if s not in available_sectors]
        if missing:
            raise ValueError(f"Missing sectors: {missing}")

    n = len(sectors)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
        squeeze=False
    )

    years = list(wide_scenario.columns)

    for k, sector in enumerate(sectors):
        ax = axes[k // ncols][k % ncols]

        sector_paths = paths.xs(sector, level="Sector")
        sector_paths = sector_paths[years]

        q05 = sector_paths.quantile(q_low, axis=0)
        q50 = sector_paths.quantile(0.50, axis=0)
        q95 = sector_paths.quantile(q_high, axis=0)

        scen = wide_scenario.loc[sector, years]

        ax.fill_between(
            years,
            q05.values.astype(float),
            q95.values.astype(float),
            color=band_color,
            alpha=0.25,
            label=f"IC {int(100*q_low)}% - {int(100*q_high)}%"
        )
        ax.plot(
            years,
            q50.values.astype(float),
            color=median_color,
            linewidth=2,
            label="Simulated median trajectory"
        )
        ax.plot(
            years,
            scen.values.astype(float),
            color=scenario_color,
            linewidth=2,
            linestyle="--",
            label="Scenario"
        )

        ax.set_title(sector)
        ax.set_xlabel("Year")
        ax.set_ylabel("Emissions")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

    fig.tight_layout()
    plt.savefig("Tempfigs/traj.png")
    plt.close()
    
#%
paths = out["paths"]
wide_scenario = out["wide_scenario"]

plot_sector_trajectories_with_bands(paths=paths, wide_scenario=wide_scenario,ncols=2, suptitle=f"{scen_name}, simulated median vs scenario, 5%-95% interval")

#%%

def compute_rolling_sector_covariances(
    paths: pd.DataFrame,
    window_years: int = 10,
    method: str = "log_return",
    active_only: bool = True,
    min_obs: int = 3,
):
    years = sorted([int(c) for c in paths.columns])
    sectors = paths.index.get_level_values("Sector").unique().tolist()
    all_paths = paths.index.get_level_values("path").unique().tolist()

    cov_rows = []
    cov_mats = {}

    for p in tqdm(all_paths, "Number of paths"):
        wide = paths.xs(p, level="path").copy()
        wide = wide.reindex(sectors)
        wide.columns = [int(c) for c in wide.columns]
        wide = wide.sort_index(axis=1)

        if method == "log_return":
            base = wide.copy()
            if active_only:
                base = base.where(base > 0)
            data = np.log(base).diff(axis=1).iloc[:, 1:]

        elif method == "pct_change":
            base = wide.copy()
            if active_only:
                base = base.where(base > 0)
            data = base.pct_change(axis=1).iloc[:, 1:]

        elif method == "level":
            data = wide.copy()
            if active_only:
                data = data.where(data > 0)

        else:
            raise ValueError("method must be one of: 'log_return', 'pct_change', 'level'")

        data_t = data.T

        valid_years = sorted(data_t.index.tolist())

        for start_idx in range(0, len(valid_years) - window_years + 1):
            window_cols = valid_years[start_idx:start_idx + window_years]
            window_start = window_cols[0]
            window_end = window_cols[-1]

            window_df = data_t.loc[window_cols, sectors]

            cov_mat = pd.DataFrame(index=sectors, columns=sectors, dtype=float)
            nobs_mat = pd.DataFrame(index=sectors, columns=sectors, dtype=float)

            for i in sectors:
                for j in sectors:
                    pair = window_df[[i, j]].dropna()

                    n_obs = len(pair)
                    nobs_mat.loc[i, j] = n_obs

                    if n_obs >= min_obs:
                        cov_ij = pair.cov().iloc[0, 1]
                    else:
                        cov_ij = np.nan

                    cov_mat.loc[i, j] = cov_ij

                    cov_rows.append({
                        "path": p,
                        "window_start": window_start,
                        "window_end": window_end,
                        "sector_i": i,
                        "sector_j": j,
                        "covariance": cov_ij,
                        "n_obs": n_obs
                    })

            cov_mats[(p, window_start, window_end)] = {
                "cov": cov_mat,
                "n_obs": nobs_mat
            }

    cov_long = pd.DataFrame(cov_rows)
    return cov_long, cov_mats

""" cov_long, cov_mats = compute_rolling_sector_covariances(
    paths=out["paths"],
    window_years=10,
    method="log_return",
    active_only=True,
    min_obs=3
)

cov_summary = (
    cov_long
    .groupby(["window_start", "window_end", "sector_i", "sector_j"], as_index=False)
    .agg(
        mean_cov=("covariance", "mean"),
        median_cov=("covariance", "median"),
        q05_cov=("covariance", lambda x: x.quantile(0.05)),
        q95_cov=("covariance", lambda x: x.quantile(0.95)),
        mean_n_obs=("n_obs", "mean")
    )
) """
#%%

def plot_mean_cov_vs_other_sectors(
    cov_summary: pd.DataFrame,
    sector_ref: str,
    ncols: int = 3,
    figsize_per_subplot=(6, 4),
    ci_low_col: str = "q05_cov",
    ci_high_col: str = "q95_cov",
    mean_col: str = "mean_cov",
    year_col: str = "window_start",
    n_obs_col: str = "mean_n_obs",
    as_pct: bool = True,
    suptitle: str | None = None,
    extinction_rule: str = "nan",   # "nan", "n_obs", "either"
    min_n_obs: int = 3,
    draw_all_extinction_lines: bool = True,
):

    df = cov_summary.copy()

    required_cols = {year_col, "sector_i", "sector_j", mean_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns : {missing}")

    if extinction_rule in ("n_obs", "either") and n_obs_col not in df.columns:
        raise ValueError(f"Column '{n_obs_col}' required for extinction_rule='{extinction_rule}'")

    if ci_low_col not in df.columns:
        df[ci_low_col] = np.nan
    if ci_high_col not in df.columns:
        df[ci_high_col] = np.nan
    if n_obs_col not in df.columns:
        df[n_obs_col] = np.nan

    df = df[(df["sector_i"] == sector_ref) | (df["sector_j"] == sector_ref)].copy()

    if df.empty:
        raise ValueError(f"No data for '{sector_ref}'")

    df["other_sector"] = np.where(df["sector_i"] == sector_ref, df["sector_j"], df["sector_i"])
    df = df[df["other_sector"] != sector_ref].copy()

    if df.empty:
        raise ValueError(f"No counterpart sectors found for '{sector_ref}'")

    df_plot = (
        df.groupby([year_col, "other_sector"], as_index=False)
          .agg({
              mean_col: "mean",
              ci_low_col: "mean",
              ci_high_col: "mean",
              n_obs_col: "mean"
          })
          .sort_values(["other_sector", year_col])
    )

    if as_pct:
        df_plot[mean_col] = 100 * df_plot[mean_col]
        df_plot[ci_low_col] = 100 * df_plot[ci_low_col]
        df_plot[ci_high_col] = 100 * df_plot[ci_high_col]

    other_sectors = sorted(df_plot["other_sector"].unique().tolist())

    n = len(other_sectors)
    nrows = math.ceil(n / ncols)

    all_years = sorted(df_plot[year_col].dropna().astype(int).unique().tolist())
    x_min, x_max = min(all_years), max(all_years)

    extinction_years = {}

    for other in other_sectors:
        sub = df_plot[df_plot["other_sector"] == other].sort_values(year_col).copy()

        cond_nan = sub[mean_col].isna()
        cond_nobs = sub[n_obs_col] < min_n_obs

        if extinction_rule == "nan":
            #mean cov is NaN rule 
            extinct_mask = cond_nan
        elif extinction_rule == "n_obs":
            # Not enough observations rule
            extinct_mask = cond_nobs
        elif extinction_rule == "either":
            # Either mean cov is NaN or not enough observations
            extinct_mask = cond_nan | cond_nobs
        else:
            raise ValueError("extinction_rule must be 'nan', 'n_obs' or 'either'")

        if extinct_mask.any():
            extinction_year = int(sub.loc[extinct_mask, year_col].iloc[0])
            extinction_years[other] = extinction_year
        else:
            extinction_years[other] = None

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
        squeeze=False,
        sharex=True
    )

    cmap = plt.get_cmap("tab10")
    extinction_colors = {
        other: cmap(i % 10) for i, other in enumerate(other_sectors)
    }

    for k, other in enumerate(other_sectors):
        ax = axes[k // ncols][k % ncols]

        sub = df_plot[df_plot["other_sector"] == other].sort_values(year_col)

        x = sub[year_col].astype(int).values
        y = sub[mean_col].astype(float).values
        y_low = sub[ci_low_col].astype(float).values
        y_high = sub[ci_high_col].astype(float).values

        valid_ci = ~(np.isnan(y_low) | np.isnan(y_high))
        if valid_ci.any():
            ax.fill_between(
                x[valid_ci],
                y_low[valid_ci],
                y_high[valid_ci],
                color="steelblue",
                alpha=0.25,
                label="IC"
            )

        ax.plot(
            x,
            y,
            color="navy",
            linewidth=2,
            label="Mean covariance"
        )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

        if draw_all_extinction_lines:
            for sec, ext_year in extinction_years.items():
                if ext_year is not None:
                    ax.axvline(
                        ext_year,
                        color=extinction_colors[sec],
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.8
                    )
        else:
            ext_year = extinction_years.get(other)
            if ext_year is not None:
                ax.axvline(
                    ext_year,
                    color="crimson",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.9
                )

        ax.set_title(f"{sector_ref} vs {other}")
        ax.set_ylabel("Mean covariance (x100)" if as_pct else "Mean covariance")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Window start year")

    if suptitle is None:
        suptitle = (
            f"10-year rolling simulation-averaged covariance trajectories, "
            f"{sector_ref} vs other sectors"
        )

    fig.suptitle(suptitle, fontsize=14)

    legend_handles = [
        Line2D([0], [0], color="navy", lw=2, label="Mean covariance"),
        Line2D([0], [0], color="steelblue", lw=6, alpha=0.25, label="IC")
    ]

    if draw_all_extinction_lines:
        for sec, ext_year in extinction_years.items():
            if ext_year is not None:
                legend_handles.append(
                    Line2D(
                        [0], [0],
                        color=extinction_colors[sec],
                        lw=1.5,
                        linestyle=":",
                        label=f"Extinction {sec}: {ext_year}"
                    )
                )
    else:
        legend_handles.append(
            Line2D([0], [0], color="crimson", lw=2, linestyle=":", label="Extinction")
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(4, len(legend_handles)),
        bbox_to_anchor=(0.5, 0.98)
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("Tempfigs/covtraj.png")
    plt.close()
    
    
# plot_mean_cov_vs_other_sectors( cov_summary=cov_summary, sector_ref="Other", ncols=3, as_pct=True, extinction_rule="either",
#    min_n_obs=3,draw_all_extinction_lines=True)

#%% Bayesian filter on a simulated path

def _logpdf_mvn(x, mean, cov):
    """Log-pdf of a multivariate normal distribution."""
    n = len(x)
    diff = x - mean
    try:
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
        z = np.linalg.solve(L, diff)
        quad = float(z.T @ z)
        logdet = 2.0 * float(np.log(np.diag(L)).sum())
        return -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)
    except np.linalg.LinAlgError:
        return -np.inf


def filter_path_probabilities(
    single_path: pd.DataFrame,
    all_scenarios_wide: dict[str, pd.DataFrame],
    sigma_scen: pd.Series,
    corr_scen: pd.DataFrame,
    prior: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Run a Bayesian filter on a single simulated path to track scenario
    probabilities over time.

    At each time step, only sectors with strictly positive emissions
    (non-zero / non-extinct) are used in the likelihood evaluation.

    Parameters
    ----------
    single_path : pd.DataFrame
        Emissions for one path. Index = sectors, columns = years (int).
    all_scenarios_wide : dict[str, pd.DataFrame]
        {scenario_name: DataFrame} with index=sectors, columns=years (int).
    sigma_scen : pd.Series
        Per-sector volatility, indexed by sector name.
    corr_scen : pd.DataFrame
        Sector correlation matrix, indexed/columned by sector name.
    prior : np.ndarray or None
        Initial scenario probabilities. Uniform if None.

    Returns
    -------
    pd.DataFrame
        Rows = years (including initial year), columns = scenario names,
        values = posterior probabilities.
    """
    scenario_names = list(all_scenarios_wide.keys())
    K = len(scenario_names)
    sectors = list(single_path.index)
    years = [int(c) for c in single_path.columns]

    if prior is None:
        probas = np.ones(K) / K
    else:
        probas = np.asarray(prior, dtype=float)
        probas = probas / probas.sum()

    # Precompute scenario growth ratios g_{i,t}^s = scen_s(t) / scen_s(t-1)
    scenario_g = {}
    for s_name, s_wide in all_scenarios_wide.items():
        sw = s_wide.reindex(index=sectors, columns=years).astype(float)
        g = sw.shift(-1, axis=1).iloc[:, :-1].div(sw.iloc[:, :-1])
        g.columns = years[1:]
        scenario_g[s_name] = g

    # Full covariance: V = diag(sigma) @ C @ diag(sigma)
    sigma = sigma_scen.reindex(sectors).values.astype(float)
    C = corr_scen.reindex(index=sectors, columns=sectors).values.astype(float)
    V_full = np.diag(sigma) @ C @ np.diag(sigma)

    # Observed log-returns (NaN where emissions <= 0)
    path_vals = single_path.reindex(sectors)[years].astype(float)
    log_path = np.log(path_vals.where(path_vals > 0))
    log_returns = log_path.diff(axis=1).iloc[:, 1:]  # sectors × years[1:]

    history = [probas.copy()]  # prior at initial year

    for year in years[1:]:
        r_obs = log_returns[year]

        # Only use sectors that are alive (non-NaN log-return)
        active_mask = r_obs.notna()
        active_sectors = active_mask[active_mask].index.tolist()

        if len(active_sectors) == 0:
            # No observable data this year: probabilities unchanged
            history.append(probas.copy())
            continue

        idx = [sectors.index(s) for s in active_sectors]
        V_sub = V_full[np.ix_(idx, idx)]
        sigma_sub = sigma[idx]
        r_obs_sub = r_obs[active_sectors].values

        # Log-likelihood under each scenario
        log_liks = np.zeros(K)
        for k, s_name in enumerate(scenario_names):
            g_vals = scenario_g[s_name].loc[active_sectors, year].values.astype(float)
            # Guard against zero / negative growth (extinct sector in scenario)
            g_vals = np.where(g_vals > 0, g_vals, 1e-30)
            # Expected log-return: log(g) - 0.5*sigma^2
            mean_sub = np.log(g_vals) - 0.5 * sigma_sub ** 2
            log_liks[k] = _logpdf_mvn(r_obs_sub, mean_sub, V_sub)

        # Bayesian update in log-space
        log_priors = np.log(np.maximum(probas, 1e-300))
        log_post = log_liks + log_priors
        log_post -= log_post.max()
        posteriors = np.exp(log_post)
        posteriors /= posteriors.sum()

        probas = posteriors
        history.append(probas.copy())

    history_df = pd.DataFrame(history, index=years, columns=scenario_names)
    history_df.index.name = "Year"
    return history_df


def filter_from_paths(
    paths: pd.DataFrame,
    path_id: int,
    df_scen: pd.DataFrame,
    sigma_scen: pd.Series,
    corr_scen: pd.DataFrame,
    prior: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: extract a single path from the full paths DataFrame
    and run the Bayesian scenario filter.

    Parameters
    ----------
    paths : pd.DataFrame
        MultiIndex (path, Sector) × years, as returned by simulate_sector_paths.
    path_id : int
        Which path index to extract.
    df_scen : pd.DataFrame
        Full scenario data (e.g. clipped), with columns Scenario, Sector, and
        year columns.
    sigma_scen : pd.Series
        Per-sector volatility.
    corr_scen : pd.DataFrame
        Sector correlation matrix.
    prior : np.ndarray or None
        Initial scenario probabilities. Uniform if None.

    Returns
    -------
    pd.DataFrame
        Annual probability history (rows=years, columns=scenarios).
    """
    single_path = paths.xs(path_id, level="path")

    # Build all scenario wide DataFrames
    scenario_names = df_scen["Scenario"].unique()
    all_scenarios_wide = {}
    for s_name in scenario_names:
        all_scenarios_wide[s_name] = prepare_scenario_wide(df_scen, s_name)

    return filter_path_probabilities(
        single_path=single_path,
        all_scenarios_wide=all_scenarios_wide,
        sigma_scen=sigma_scen,
        corr_scen=corr_scen,
        prior=prior,
    )


#%% Run the filter on multiple simulated paths and average

def average_filter_from_paths(
    paths: pd.DataFrame,
    df_scen: pd.DataFrame,
    sigma_scen: pd.Series,
    corr_scen: pd.DataFrame,
    n_paths: int | None = None,
    prior: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Run the Bayesian filter on multiple simulated paths and return
    the average probability history as well as all individual histories.

    Parameters
    ----------
    paths : pd.DataFrame
        MultiIndex (path, Sector) × years.
    df_scen : pd.DataFrame
        Full scenario data with Scenario, Sector and year columns.
    sigma_scen, corr_scen : volatility & correlation.
    n_paths : int or None
        Number of paths to use. None = all available paths.
    prior : np.ndarray or None
        Initial scenario probabilities.

    Returns
    -------
    mean_history : pd.DataFrame  (years × scenarios), averaged probabilities.
    all_histories : list[pd.DataFrame], one per path.
    """
    all_path_ids = paths.index.get_level_values("path").unique().tolist()
    if n_paths is not None:
        all_path_ids = all_path_ids[:n_paths]

    # Build scenario wide DataFrames once
    scenario_names = df_scen["Scenario"].unique()
    all_scenarios_wide = {}
    for s_name in scenario_names:
        all_scenarios_wide[s_name] = prepare_scenario_wide(df_scen, s_name)

    all_histories = []
    for pid in tqdm(all_path_ids, desc="Filtering paths"):
        single_path = paths.xs(pid, level="path")
        h = filter_path_probabilities(
            single_path=single_path,
            all_scenarios_wide=all_scenarios_wide,
            sigma_scen=sigma_scen,
            corr_scen=corr_scen,
            prior=prior,
        )
        all_histories.append(h)

    stacked = np.stack([h.values for h in all_histories], axis=0)
    mean_vals = stacked.mean(axis=0)
    mean_history = pd.DataFrame(
        mean_vals,
        index=all_histories[0].index,
        columns=all_histories[0].columns,
    )
    mean_history.index.name = "Year"
    return mean_history, all_histories


#%% Stacked area plot

SCENAR_ABBREV = {
    "Below 2°C": "B2°",
    "Current Policies": "CurPo",
    "Delayed transition": "Delay",
    "Fragmented World": "Frag",
    "Low demand": "LowD",
    "Nationally Determined Contributions (NDCs)": "NDCs",
    "Net Zero 2050": "NZ",
}

SCENAR_COLORS = {
    "B2°": "#1f77b4",
    "CurPo": "#ff7f0e",
    "Delay": "#2ca02c",
    "Frag": "#d62728",
    "LowD": "#9467bd",
    "NDCs": "#8c564b",
    "NZ": "#e377c2",
}


def detect_scenario_extinctions(wide_scen: pd.DataFrame) -> dict[str, int]:
    """
    Detect the first year each sector reaches zero (or negative) emissions
    in the scenario trajectory.

    Returns
    -------
    dict  {sector_name: extinction_year}  (only sectors that go extinct).
    """
    extinctions = {}
    years = list(wide_scen.columns)
    for sector in wide_scen.index:
        vals = wide_scen.loc[sector, years].astype(float)
        mask = vals <= 0
        if mask.any():
            extinctions[sector] = int(years[mask.values.argmax()])
    return extinctions


def plot_probas_area(
    proba_history: pd.DataFrame,
    title: str = "Scenario probabilities over time",
    sort_year: int | None = None,
    extinctions: dict[str, int] | None = None,
):
    """
    Stacked area plot of scenario probabilities.

    Parameters
    ----------
    proba_history : pd.DataFrame
        Rows = years, columns = scenario names, values = probabilities.
    title : str
        Plot title.
    sort_year : int or None
        Year used to sort scenarios by descending probability.
        Defaults to the last year.
    extinctions : dict or None
        {sector_name: year} of sector extinctions to annotate on the plot.
    """
    df = proba_history.copy()
    df.rename(columns=SCENAR_ABBREV, inplace=True)

    if sort_year is None:
        sort_year = df.index[-1]
    df = df[df.columns[df.loc[sort_year].argsort()[::-1]]]

    color_list = [SCENAR_COLORS.get(col, "#333333") for col in df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot.area(ax=ax, color=color_list, linewidth=0.5)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Year")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)

    if extinctions:
        # Group sectors by extinction year for cleaner labels
        year_to_sectors: dict[int, list[str]] = {}
        for sec, yr in extinctions.items():
            year_to_sectors.setdefault(yr, []).append(sec)

        cmap = plt.get_cmap("Set2")
        for i, (yr, secs) in enumerate(sorted(year_to_sectors.items())):
            color = cmap(i % 8)
            ax.axvline(yr, color=color, linestyle=":", linewidth=1.8, alpha=0.85)
            label = ", ".join(secs)
            ax.annotate(
                f"{label}\n extinct {yr}",
                xy=(yr, 0.98 - 0.06 * i),
                fontsize=7,
                color=color,
                fontweight="bold",
                ha="left",
                va="top",
                xytext=(4, 0),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig("Tempfigs/allprobs.png")
    plt.close()


#%% Expected total emissions E[sum_i E_{i,t} | F_s] with absorbing barrier
#
# Key result: by linearity of expectation,
#   E[sum_i E_{i,t} | F_s] = sum_i E[E_{i,t} * 1{survived} | E_{i,s}]
#
# Even though sectors are correlated, the barrier event for sector i depends
# ONLY on sector i's marginal path.  Since Z_{i,t} ~ N(0,1) regardless of
# the correlation structure (diagonal of corr matrix = 1), the per-sector
# computation is independent.  Correlations are IRRELEVANT here.
#
# Without barrier (epsilon=0):
#   E[E_{i,t} | E_{i,s}] = E_{i,s} * prod_{tau=s}^{t-1} g_{i,tau}
#   (the GBM martingale correction cancels: E[exp(-sig^2/2 + sig Z)] = 1)
#
# With barrier (epsilon>0):
#   Solved per sector via 1D forward density propagation on a grid
#   in Y = log(E/epsilon) space with absorbing barrier at Y=0.

def _sector_expected_curve_barrier(
    e_i_s: float,
    growth_ratios: np.ndarray,
    sigma_i: float,
    epsilon: float,
    n_grid: int = 300,
) -> np.ndarray:
    """
    For a single sector, compute
      E[E_{i,s+k} * 1{E > epsilon for all intermediate steps} | E_{i,s}]
    for k = 0, 1, ..., len(growth_ratios).

    Uses forward propagation of the density on a 1D grid in
    Y = log(E / epsilon) space with absorbing barrier at Y = 0.
    """
    T = len(growth_ratios)
    curve = np.zeros(T + 1)
    curve[0] = e_i_s

    if e_i_s <= epsilon:
        return curve

    log_eps = np.log(max(epsilon, 1e-300))
    y_s = np.log(e_i_s) - log_eps

    # Drifts in Y-space: mu_k = log(g_k) - sigma^2/2
    drifts = np.empty(T)
    for k in range(T):
        g = growth_ratios[k]
        drifts[k] = np.log(g) - 0.5 * sigma_i ** 2 if g > 0 else -np.inf

    # Grid range
    finite_drifts = np.where(np.isfinite(drifts), drifts, 0)
    cumsum_d = np.cumsum(finite_drifts)
    max_std = sigma_i * np.sqrt(max(T, 1))
    y_max = max(y_s, y_s + (cumsum_d.max() if len(cumsum_d) else 0)) + 6 * max_std
    y_max = max(y_max, 6 * max_std)

    dy = y_max / n_grid
    grid = np.linspace(dy / 2, y_max - dy / 2, n_grid)

    # Init: delta approximation at y_s
    p = np.zeros(n_grid)
    idx0 = int(np.argmin(np.abs(grid - y_s)))
    p[idx0] = 1.0 / dy

    for k in range(T):
        if not np.isfinite(drifts[k]):
            break
        mu = drifts[k]
        # Gaussian transition kernel: kernel[i,j] = P(grid[i] -> grid[j])
        diff = grid[np.newaxis, :] - grid[:, np.newaxis] - mu
        kernel = np.exp(-0.5 * (diff / sigma_i) ** 2) / (sigma_i * np.sqrt(2 * np.pi))
        # Forward step (mass below y=0 is automatically lost = absorbed)
        p = (p * dy) @ kernel
        curve[k + 1] = epsilon * float(np.sum(np.exp(grid) * p * dy))

    return curve


def expected_total_emissions_curve(
    e_s: pd.Series,
    wide_scen: pd.DataFrame,
    sigma_scen: pd.Series,
    year_s: int,
    epsilon: float = 1.0,
    n_grid: int = 300,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Compute E[sum_i E_{i,t} | F_s] for all future years t >= s,
    with and without absorbing barrier at epsilon.

    Parameters
    ----------
    e_s : per-sector emissions at conditioning year s.
    wide_scen : scenario data, index=sectors, columns=years.
    sigma_scen : per-sector volatility (possibly amplified).
    year_s : conditioning year.
    epsilon : absorbing barrier threshold.
    n_grid : grid size for numerical integration.

    Returns
    -------
    per_sector_barrier : DataFrame (sectors × years), with barrier.
    total_barrier : Series, total with barrier.
    total_no_barrier : Series, total without barrier
        (= E_{i,s} * cumulative growth, the scenario-scaled trajectory).
    """
    sectors = list(e_s.index)
    all_years = sorted(int(c) for c in wide_scen.columns)
    future_years = [y for y in all_years if y >= year_s]

    scen = wide_scen.reindex(index=sectors, columns=future_years).astype(float)
    g = scen.shift(-1, axis=1).iloc[:, :-1].div(scen.iloc[:, :-1])
    g.columns = future_years[1:]

    per_sector = pd.DataFrame(0.0, index=sectors, columns=future_years)
    per_sector_nb = pd.DataFrame(0.0, index=sectors, columns=future_years)

    for sec in sectors:
        ei = float(e_s[sec])
        growth = g.loc[sec].values.astype(float)
        sig = float(sigma_scen.reindex([sec]).iloc[0])

        curve = _sector_expected_curve_barrier(ei, growth, sig, epsilon, n_grid)
        per_sector.loc[sec] = curve

        # No barrier: E[E_{i,t} | E_{i,s}] = E_{i,s} * prod g
        nb = np.zeros(len(future_years))
        nb[0] = ei
        for k, gr in enumerate(growth):
            if gr <= 0 or nb[k] <= 0:
                break
            nb[k + 1] = nb[k] * gr
        per_sector_nb.loc[sec] = nb

    total_barrier = per_sector.sum(axis=0)
    total_nb = per_sector_nb.sum(axis=0)

    return per_sector, total_barrier, total_nb


def mc_mean_with_barrier(
    paths: pd.DataFrame,
    epsilon: float,
) -> pd.Series:
    """
    Compute the MC average of total emissions from simulated paths,
    applying the absorbing barrier at epsilon retroactively.
    """
    n_all = len(paths.index.get_level_values("path").unique())
    n_sec = len(paths.index.get_level_values("Sector").unique())
    years = [int(c) for c in paths.columns]
    n_years = len(years)

    arr = paths.values.copy().reshape(n_all, n_sec, n_years)
    for t in range(1, n_years):
        extinct = arr[:, :, t - 1] <= epsilon
        arr[:, :, t] = np.where(extinct, 0.0, arr[:, :, t])
        arr[:, :, t] = np.where(arr[:, :, t] <= epsilon, 0.0, arr[:, :, t])
    total_per_path = arr.sum(axis=1)
    mc_mean = total_per_path.mean(axis=0)
    return pd.Series(mc_mean, index=years, name="MC mean")


def plot_expected_emissions(
    total_barrier: pd.Series,
    total_no_barrier: pd.Series,
    mc_mean: pd.Series | None = None,
    epsilon: float = 1.0,
    extinctions: dict[str, int] | None = None,
    title: str | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    years = [int(y) for y in total_no_barrier.index]

    ax.plot(years, total_no_barrier.values, "k--", lw=2,
            label="E[Σ Ei | Fs] no barrier")
    ax.plot(years, total_barrier.values, "r-", lw=2,
            label=f"E[Σ Ei | Fs] barrier ε={epsilon}")
    if mc_mean is not None:
        ax.plot([int(y) for y in mc_mean.index], mc_mean.values, "b:", lw=2,
                label=f"MC mean (barrier ε={epsilon})")

    if extinctions:
        year_to_sectors: dict[int, list[str]] = {}
        for sec, yr in extinctions.items():
            year_to_sectors.setdefault(yr, []).append(sec)
        cmap = plt.get_cmap("Set2")
        for i, (yr, secs) in enumerate(sorted(year_to_sectors.items())):
            color = cmap(i % 8)
            ax.axvline(yr, color=color, linestyle=":", linewidth=1.5, alpha=0.8)
            ax.annotate(", ".join(secs), xy=(yr, ax.get_ylim()[1] * (0.95 - 0.05 * i)),
                        fontsize=7, color=color, fontweight="bold", ha="left")

    ax.set_xlabel("Year")
    ax.set_ylabel("Total emissions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("Tempfigs/expected.png")
    plt.close()


#%% Mixture forward simulation from filter probabilities at time t

def simulate_mixture_from_t(
    e_t: pd.Series,
    year_t: int,
    scenario_probas: pd.Series | dict[str, float],
    df_scen: pd.DataFrame,
    sigma_scen: pd.Series,
    corr_scen: pd.DataFrame,
    n_paths: int = 1000,
    seed: int | None = 42,
    zero_absorbing: bool = True,
    floor: float = 0.0,
    vol_amplification: float = 1.0,
) -> dict:
    """
    Simulate forward from observed emissions at year_t using a mixture model:
    each path draws ONE scenario according to scenario_probas, then follows
    the GBM dynamics with that scenario's drift.

    Parameters
    ----------
    e_t : pd.Series
        Per-sector emissions at year t (index = sector names).
    year_t : int
        Conditioning year.
    scenario_probas : Series or dict {scenario_name: probability}.
    df_scen : pd.DataFrame
        Full scenario data (e.g. clipped) with Scenario, Sector, year columns.
    sigma_scen : pd.Series
        Per-sector volatility (possibly amplified).
    corr_scen : pd.DataFrame
        Sector correlation matrix.
    n_paths : int
        Number of simulation paths.
    seed : int or None
    zero_absorbing : bool
    floor : float
    vol_amplification : float
        Additional amplification on top of sigma_scen.

    Returns
    -------
    dict with keys:
        paths : pd.DataFrame, MultiIndex (path, Sector) × future years
        scen_draws : np.ndarray of scenario names per path
        scenario_probas : the input probabilities used
        expected_total : pd.Series, E[Σ E_i | F_t] per year (analytical mixture)
    """
    if isinstance(scenario_probas, dict):
        scenario_probas = pd.Series(scenario_probas, dtype=float)
    scenario_probas = scenario_probas / scenario_probas.sum()

    # Build scenario growth ratios for each scenario
    sectors = list(e_t.index)
    all_scenarios_g = {}
    all_scenarios_wide = {}
    for s_name in scenario_probas.index:
        wide = prepare_scenario_wide(df_scen, s_name)
        all_scenarios_wide[s_name] = wide
        sw = wide.reindex(index=sectors).astype(float)
        all_years = sorted(int(c) for c in sw.columns)
        future_years = [y for y in all_years if y > year_t]
        # g_{i,t} = scen(t) / scen(t-1), but we need the ratio *starting from year_t*
        prev_years = [year_t] + future_years[:-1]
        g_df = pd.DataFrame(index=sectors, columns=future_years, dtype=float)
        for y, yp in zip(future_years, prev_years):
            if y in sw.columns and yp in sw.columns:
                g_df[y] = sw[y] / sw[yp]
            else:
                g_df[y] = np.nan
        all_scenarios_g[s_name] = g_df.astype(float)

    # Determine future years from any scenario
    any_g = next(iter(all_scenarios_g.values()))
    future_years = list(any_g.columns)
    all_sim_years = [year_t] + future_years

    # Correlation / volatility
    sigma = sigma_scen.reindex(sectors).astype(float)
    corr = corr_scen.reindex(index=sectors, columns=sectors).astype(float)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr.values, 1.0)
    n_sectors = len(sectors)
    L = np.linalg.cholesky(corr.values + 1e-12 * np.eye(n_sectors))
    sig = sigma.values * vol_amplification

    rng = np.random.default_rng(seed)

    # Draw scenario per path
    scen_names = scenario_probas.index.to_numpy()
    pvals = scenario_probas.values.astype(float)
    scen_draws = rng.choice(scen_names, size=n_paths, p=pvals)

    # Simulate
    n_years = len(all_sim_years)
    arr = np.zeros((n_paths, n_sectors, n_years), dtype=float)
    arr[:, :, 0] = e_t.values[None, :]

    for p in range(n_paths):
        s_name = scen_draws[p]
        g_df = all_scenarios_g[s_name]
        for t_idx in range(1, n_years):
            z = L @ rng.standard_normal(n_sectors)
            prev = arr[p, :, t_idx - 1]
            yr = all_sim_years[t_idx]
            gt = g_df[yr].reindex(sectors).values.astype(float)
            gt = np.where(np.isfinite(gt), gt, 0.0)

            step = prev * gt * np.exp(-0.5 * sig ** 2 + sig * z)

            if zero_absorbing:
                step = np.where(prev <= 0.0, 0.0, step)
                step = np.where(gt <= 0.0, 0.0, step)
            if floor is not None:
                step = np.maximum(step, floor)

            arr[p, :, t_idx] = step

    idx = pd.MultiIndex.from_product(
        [range(n_paths), sectors], names=["path", "Sector"]
    )
    df_paths = pd.DataFrame(
        arr.reshape(n_paths * n_sectors, n_years),
        index=idx,
        columns=all_sim_years,
    )

    # Analytical expected total = weighted sum of per-scenario deterministic paths
    # E[Σ E_i(t') | F_t] = Σ_s p_s * Σ_i E_{i,t} * prod_{tau=t}^{t'-1} g_{i,tau}^s
    expected = pd.Series(0.0, index=all_sim_years)
    for s_name in scenario_probas.index:
        p_s = float(scenario_probas[s_name])
        if p_s <= 0:
            continue
        g_df = all_scenarios_g[s_name]
        det = np.zeros((n_sectors, n_years))
        det[:, 0] = e_t.values
        for t_idx in range(1, n_years):
            yr = all_sim_years[t_idx]
            gt = g_df[yr].reindex(sectors).values.astype(float)
            gt = np.where(np.isfinite(gt), gt, 0.0)
            det[:, t_idx] = det[:, t_idx - 1] * gt
            if zero_absorbing:
                det[:, t_idx] = np.where(det[:, t_idx - 1] <= 0, 0, det[:, t_idx])
                det[:, t_idx] = np.where(gt <= 0, 0, det[:, t_idx])
        expected += p_s * det.sum(axis=0)

    expected_total = pd.Series(expected, index=all_sim_years, name="E[Σ Ei | Ft]")

    return {
        "paths": df_paths,
        "scen_draws": scen_draws,
        "scenario_probas": scenario_probas,
        "expected_total": expected_total,
        "years": all_sim_years,
        "sectors": sectors,
    }


def plot_mixture_simulation(
    mix_result: dict,
    extinctions: dict[str, int] | None = None,
    title: str | None = None,
    q_low: float = 0.05,
    q_high: float = 0.95,
):
    """Plot total emissions from a mixture simulation with MC bands + analytical expectation."""
    paths = mix_result["paths"]
    years = mix_result["years"]
    expected = mix_result["expected_total"]

    # Total emissions per path per year
    total_per_path = paths.groupby("path").sum()
    mc_mean = total_per_path.mean(axis=0)
    mc_qlo = total_per_path.quantile(q_low, axis=0)
    mc_qhi = total_per_path.quantile(q_high, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = [int(y) for y in years]

    ax.fill_between(x, mc_qlo.values.astype(float), mc_qhi.values.astype(float),
                    color="steelblue", alpha=0.2,
                    label=f"MC {int(100*q_low)}%-{int(100*q_high)}%")
    ax.plot(x, mc_mean.values, "b-", lw=2, label="MC mean")
    ax.plot(x, expected.values, "r--", lw=2, label="Analytical E[Σ Ei | Ft]")

    if extinctions:
        year_to_sectors: dict[int, list[str]] = {}
        for sec, yr in extinctions.items():
            year_to_sectors.setdefault(yr, []).append(sec)
        cmap = plt.get_cmap("Set2")
        for i, (yr, secs) in enumerate(sorted(year_to_sectors.items())):
            color = cmap(i % 8)
            ax.axvline(yr, color=color, linestyle=":", linewidth=1.5, alpha=0.8)
            ax.annotate(", ".join(secs), xy=(yr, ax.get_ylim()[1] * (0.95 - 0.05 * i)),
                        fontsize=7, color=color, fontweight="bold", ha="left")

    ax.set_xlabel("Year")
    ax.set_ylabel("Total emissions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("Tempfigs/mixsim.png")
    plt.close()


#%% Run

N_FILTER_PATHS = 50

mean_history, all_histories = average_filter_from_paths(
    paths=out["paths"],
    df_scen=clipped,
    sigma_scen=out["sigma"],
    corr_scen=out["corr"],
    n_paths=N_FILTER_PATHS,
)

extinctions = detect_scenario_extinctions(out["wide_scenario"])

plot_probas_area(
    mean_history,
    title=f"Average scenario probabilities ({N_FILTER_PATHS} paths, simulated under '{scen_name}')",
    extinctions=extinctions,
)

#%% Expected total emissions with barrier

EPSILON = 50.0  # extinction threshold

year_s = int(out["wide_scenario"].columns[0])
e_s_scenario = out["wide_scenario"][year_s]

per_sector, total_barrier, total_nb = expected_total_emissions_curve(
    e_s=e_s_scenario,
    wide_scen=out["wide_scenario"],
    sigma_scen=out["sigma"],
    year_s=year_s,
    epsilon=EPSILON,
    n_grid=300,
)

mc_mean_barr = mc_mean_with_barrier(out["paths"], epsilon=EPSILON)

plot_expected_emissions(
    total_barrier=total_barrier,
    total_no_barrier=total_nb,
    mc_mean=mc_mean_barr,
    epsilon=EPSILON,
    extinctions=extinctions,
    title=f"E[Σ E_i | F_s] under '{scen_name}' (vol_amp={VOL_AMPLIFICATION}, ε={EPSILON})",
)

print("\nPer-sector expected emissions (with barrier):")
print(per_sector)

#%% Mixture forward simulation from filter at a chosen time

FILTER_PATH_ID = 24
MIXTURE_YEAR = 2025

# Get filter probabilities at MIXTURE_YEAR for one simulated path
filter_hist = all_histories[FILTER_PATH_ID]
probas_at_t = filter_hist.loc[MIXTURE_YEAR]
print(f"\nFilter probabilities at {MIXTURE_YEAR} (path {FILTER_PATH_ID}):")
print(probas_at_t)

# Get the emission levels at MIXTURE_YEAR from that simulated path
single_path = out["paths"].xs(FILTER_PATH_ID, level="path")
e_t = single_path[MIXTURE_YEAR]

mix = simulate_mixture_from_t(
    e_t=e_t,
    year_t=MIXTURE_YEAR,
    scenario_probas=probas_at_t,
    df_scen=clipped,
    sigma_scen=out["sigma"],
    corr_scen=out["corr"],
    n_paths=2000,
    seed=123,
    vol_amplification=1.0,  # sigma already amplified in out["sigma"]
)

print(f"\nAnalytical E[Σ Ei | F_{MIXTURE_YEAR}]:")
print(mix["expected_total"])

print(f"\nScenario draw distribution ({len(mix['scen_draws'])} paths):")
unique, counts = np.unique(mix["scen_draws"], return_counts=True)
for s, c in zip(unique, counts):
    print(f"  {s}: {c}  ({100*c/len(mix['scen_draws']):.1f}%)")

plot_mixture_simulation(
    mix,
    extinctions=extinctions,
    title=f"Mixture simulation from {MIXTURE_YEAR} (path {FILTER_PATH_ID}, '{scen_name}')",
)