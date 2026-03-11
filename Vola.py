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
):
    """
      E_{i,t+1} = E_{i,t} * g_{i,t} * exp(-0.5*sigma_i^2 + sigma_i * Z_{i,t})
    with correlated Z.
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

    sig = sigma.values

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
        "sigma": sigma,
        "corr": corr
    }

def run_simulation_from_history_and_scenario(
    df_hist_wide: pd.DataFrame,
    df_scen: pd.DataFrame,
    scenario_name: str,
    scen_to_hist: dict[str, str],
    n_paths: int = 1000,
    seed: int | None = 123,
    shrinkage: str | None = "ledoit_wolf",
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
        seed=seed
    )

    out["sigma_hist"] = sigma_hist
    out["corr_hist"] = corr_hist
    out["log_returns_hist"] = log_returns
    out["wide_scenario"] = wide_scen

    return out


#%%
scen_name = "Delayed transition"

out = run_simulation_from_history_and_scenario(
    df_hist_wide=pivot,
    df_scen=clipped,
    scenario_name=scen_name,
    scen_to_hist=SCEN_TO_HIST,
    n_paths=500,
    seed=42,
    shrinkage="ledoit_wolf"
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
    plt.show()
    
#%
paths = out["paths"]
wide_scenario = out["wide_scenario"]

plot_sector_trajectories_with_bands(
    paths=paths,
    wide_scenario=wide_scenario,
    ncols=2,
    suptitle=f"{scen_name}, simulated median vs scenario, 5%-95% interval"
)

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

cov_long, cov_mats = compute_rolling_sector_covariances(
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
)
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
    plt.show()
    
    
plot_mean_cov_vs_other_sectors(
    cov_summary=cov_summary,
    sector_ref="Other",
    ncols=3,
    as_pct=True,
    extinction_rule="either",
    min_n_obs=3,
    draw_all_extinction_lines=True
)