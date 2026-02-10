# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 13:42:08 2026

@author: LoïcMARCADET
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import multivariate_normal
import sys
import matplotlib.pyplot as plt
import seaborn as sns

import Config

#%%
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

pivot.to_excel("Data/history_filtered.xlsx")

#%%

def estimate_mu_sigma_by_sector(pivot, dt = 1.0):

    years = pd.Index(pivot.columns).astype(int)
    x = pivot.copy()
    x.columns = years
    x = x.reindex(sorted(x.columns), axis=1).astype(float)

    r = np.log(x).diff(axis=1)

    m = r.mean(axis=1, skipna=True)                 # E[r]
    v = r.var(axis=1, ddof=1, skipna=True)          # Var(r) sample

    sigma = np.sqrt(v / dt)
    mu = (m / dt) + 0.5 * (sigma ** 2)

    out = pd.DataFrame({
        "mu": mu,
        "sigma": sigma,
        "n_obs": r.count(axis=1)
    })

    return out

params = estimate_mu_sigma_by_sector(pivot, dt=1.0)
params = params.sort_values("sigma", ascending=False)
params.head()

#%%

def simulate_gbm_paths(e0: float, mu: float, sigma: float, n_steps: int, n_paths: int, dt: float = 1.0, seed: int | None = 42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_paths, n_steps))

    drift = (mu - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt) * z

    log_paths = np.log(e0) + np.cumsum(drift + diff, axis=1)
    paths = np.exp(log_paths)

    paths = np.concatenate([np.full((n_paths, 1), e0), paths], axis=1)
    return paths

def plot_real_vs_paths_from_params(
    pivot: pd.DataFrame,
    params: pd.DataFrame,
    sector: str,
    start_year: int = 2018,
    end_year: int = 2023,
    n_paths: int = 200,
    seed: int = 42
):
    mu = float(params.loc[sector, "mu"])
    sigma = float(params.loc[sector, "sigma"])

    years = pd.Index(pivot.columns).astype(int)
    real = pivot.copy()
    real.columns = years
    real = real.reindex(sorted(real.columns), axis=1)

    real = real.loc[sector, (real.columns >= start_year) & (real.columns <= end_year)].astype(float).dropna()

    if real.empty or len(real) < 2:
        raise ValueError(f"Not enough values")

    t = real.index.to_numpy()          
    y = real.to_numpy()                
    e0 = float(y[0])
    n_steps = len(y) - 1

    paths = simulate_gbm_paths(e0=e0, mu=mu, sigma=sigma, n_steps=n_steps, n_paths=n_paths, dt=1.0, seed=seed)

    plt.figure(figsize=(10, 5))

    for i in range(n_paths):
        plt.plot(t, paths[i, :], color="steelblue", alpha=0.08, linewidth=1)

    plt.plot(t, paths.mean(axis=0), color="navy", linewidth=2, label="Moyenne simulée")
    plt.plot(t, y, color="black", linewidth=2.5, marker="o", label="Réel")

    plt.title(f"{sector} — GBM (μ={mu:.4f}, σ={sigma:.4f}) — {start_year}→{end_year}")
    plt.xlabel("Année")
    plt.ylabel("Scope12")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"mu": mu, "sigma": sigma, "real": real, "paths": paths}


#%%

def simulate_gbm_paths(e0: float, mu: float, sigma: float, n_steps: int, n_paths: int, dt: float = 1.0, seed: int | None = None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_paths, n_steps))

    drift = (mu - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt) * z

    log_paths = np.log(e0) + np.cumsum(drift + diff, axis=1)
    paths = np.exp(log_paths)

    # ajoute le point initial t=0
    paths = np.concatenate([np.full((n_paths, 1), e0), paths], axis=1)
    return paths

def plot_total_real_vs_simulations(
    pivot: pd.DataFrame,
    params: pd.DataFrame,
    start_year: int = 2018,
    end_year: int = 2023,
    n_paths: int = 200,
    seed: int = 42
):
    # --- préparer pivot (années triées)
    years = pd.Index(pivot.columns).astype(int)
    x = pivot.copy()
    x.columns = years
    x = x.reindex(sorted(x.columns), axis=1).astype(float)

    # --- garder la période demandée
    x = x.loc[:, (x.columns >= start_year) & (x.columns <= end_year)].dropna(axis=0, how="any")
    # dropna how="any": on garde seulement les secteurs complets sur toute la période
    # (cohérent avec ton filtre initial, mais safe)

    if x.shape[0] == 0:
        raise ValueError("Aucun secteur avec des valeurs complètes sur la période demandée.")

    # --- réel total
    total_real = x.sum(axis=0)  # index = années
    t = total_real.index.to_numpy()
    y_real = total_real.to_numpy()

    n_steps = len(t) - 1
    if n_steps < 1:
        raise ValueError("Période trop courte pour simuler (il faut au moins 2 années).")

    # --- simulation: total = somme des GBM secteur
    rng = np.random.default_rng(seed)
    total_paths = np.zeros((n_paths, len(t)), dtype=float)

    # s'assurer qu'on a les params pour les secteurs présents
    common_sectors = x.index.intersection(params.index)
    missing = x.index.difference(params.index)
    if len(missing) > 0:
        raise ValueError(f"Il manque mu/sigma dans params pour ces secteurs: {list(missing)[:10]} ...")

    for sector in common_sectors:
        mu = float(params.loc[sector, "mu"])
        sigma = float(params.loc[sector, "sigma"])
        e0 = float(x.loc[sector, t[0]])

        # seed différente par secteur (reproductible)
        sector_seed = int(rng.integers(0, 2**32 - 1))
        paths_sector = simulate_gbm_paths(e0, mu, sigma, n_steps=n_steps, n_paths=n_paths, dt=1.0, seed=sector_seed)

        total_paths += paths_sector

    # --- plot
    plt.figure(figsize=(10, 5))

    for i in range(n_paths):
        plt.plot(t, total_paths[i, :], color="indianred", alpha=0.08, linewidth=1)

    plt.plot(t, total_paths.mean(axis=0), color="maroon", linewidth=2, label="Total simulé (moyenne)")
    plt.plot(t, y_real, color="black", linewidth=2.5, marker="o", label="Total réel")

    plt.title(f"Total Scope12 (tous secteurs) — Réel vs {n_paths} simulations GBM par secteur — {start_year}→{end_year}")
    plt.xlabel("Année")
    plt.ylabel("Scope12 total")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"total_real": total_real, "total_paths": total_paths, "sectors_used": list(common_sectors)}


#%%
# Modèle multivarié avec shrinkage



def calibrate_mvgmb_shrinkage(
    pivot: pd.DataFrame,
    start_year: int = 2018,
    end_year: int = 2023,
    shrink_lambda: float = 0.3,
):
    """
    Calibre un GBM multivarié sur les secteurs:
      dE_i / E_i = mu_i dt + ... avec covariance instantanée Sigma (sur les innovations)
    via retours log: r = diff(log(E)).
    
    Shrinkage: Sigma = (1-lam)*S + lam*diag(S)
    """
    # --- préparer la matrice niveaux (secteurs x années)
    years = pd.Index(pivot.columns).astype(int)
    X = pivot.copy()
    X.columns = years
    X = X.reindex(sorted(X.columns), axis=1).astype(float)

    # --- période de calibration
    X = X.loc[:, (X.columns >= start_year) & (X.columns <= end_year)]

    # on garde uniquement les secteurs complets sur la période
    X = X.dropna(axis=0, how="any")

    if X.shape[0] < 2:
        raise ValueError("Il faut au moins 2 secteurs avec données complètes sur la période.")
    if X.shape[1] < 2:
        raise ValueError("Il faut au moins 2 années pour calculer des log-returns.")

    sectors = X.index
    years_used = X.columns

    # --- log-returns (secteurs x (T-1))
    R = np.log(X).diff(axis=1).iloc[:, 1:]  # retire la première colonne NaN
    # ici R a shape (n_sectors, n_steps)

    n_sectors, n_steps = R.shape
    if n_steps < 2:
        raise ValueError("Pas assez d'incréments temporels pour estimer une covariance (>=2 recommandé).")

    # --- stats des retours
    r_bar = R.mean(axis=1)  # moyenne par secteur
    S = R.T.cov(ddof=1)     # covariance entre secteurs (pandas: cov sur colonnes, donc on transpose)
    # S est (n_sectors x n_sectors), index/columns = secteurs

    # --- shrinkage vers diagonale
    lam = float(shrink_lambda)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("shrink_lambda doit être entre 0 et 1.")
    T = pd.DataFrame(np.diag(np.diag(S.to_numpy())), index=sectors, columns=sectors)
    Sigma = (1 - lam) * S + lam * T

    # --- mu : E[r] = mu - 0.5*diag(Sigma)  (dt=1)
    mu = r_bar + 0.5 * pd.Series(np.diag(Sigma.to_numpy()), index=sectors)

    return {
        "sectors": sectors,
        "years": years_used,
        "X": X,
        "R": R,
        "mu": mu,               # Series
        "Sigma": Sigma,         # DataFrame
        "shrink_lambda": lam,
    }

def simulate_mvgmb_paths(mu: pd.Series, Sigma: pd.DataFrame, E0: pd.Series, n_steps: int, n_paths: int, dt: float = 1.0, seed: int = 42):
    """
    Simule des trajectoires du GBM multivarié corrélé.
    Retourne un array paths de shape (n_paths, n_steps+1, n_sectors).
    """
    sectors = mu.index
    mu_vec = mu.loc[sectors].to_numpy()
    Sigma_mat = Sigma.loc[sectors, sectors].to_numpy()
    E0_vec = E0.loc[sectors].to_numpy()

    # Cholesky (petit jitter si nécessaire pour SPD)
    jitter = 1e-12
    for _ in range(5):
        try:
            L = np.linalg.cholesky(Sigma_mat + jitter * np.eye(len(sectors)))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        raise np.linalg.LinAlgError("Cholesky impossible: Sigma pas SPD (même avec jitter).")

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_paths, n_steps, len(sectors)))  # iid
    # chocs corrélés: eps = Z @ L.T  (car cov = L L^T)
    eps = Z @ L.T

    drift = (mu_vec - 0.5 * np.diag(Sigma_mat)) * dt  # (d,)
    # diffusion: sqrt(dt) * eps
    increments = drift[None, None, :] + np.sqrt(dt) * eps  # (n_paths, n_steps, d)

    logE = np.log(E0_vec)[None, None, :] + np.cumsum(increments, axis=1)  # (n_paths, n_steps, d)
    E = np.exp(logE)

    # ajouter E0 au début
    E0_block = E0_vec[None, None, :].repeat(n_paths, axis=0)  # (n_paths, 1, d)
    paths = np.concatenate([E0_block, E], axis=1)  # (n_paths, n_steps+1, d)
    return paths

def plot_total_real_vs_simulated_total(
    pivot: pd.DataFrame,
    start_year: int = 2018,
    end_year: int = 2023,
    n_paths: int = 300,
    shrink_lambda: float = 0.3,
    seed: int = 42,
    plot_spaghetti: bool = False,
):
    cal = calibrate_mvgmb_shrinkage(pivot, start_year=start_year, end_year=end_year, shrink_lambda=shrink_lambda)

    X = cal["X"]               # niveaux (secteurs x années)
    mu = cal["mu"]
    Sigma = cal["Sigma"]
    years = cal["years"]
    sectors = cal["sectors"]

    # point initial (année start_year)
    E0 = X.loc[:, years.min()]  # Series index=sectors

    n_steps = len(years) - 1
    paths = simulate_mvgmb_paths(mu, Sigma, E0, n_steps=n_steps, n_paths=n_paths, dt=1.0, seed=seed)
    # paths shape: (n_paths, len(years), n_sectors)

    # total simulé
    total_paths = paths.sum(axis=2)   # (n_paths, len(years))
    total_mean = total_paths.mean(axis=0)
    total_p05 = np.quantile(total_paths, 0.05, axis=0)
    total_p95 = np.quantile(total_paths, 0.95, axis=0)

    # total réel
    total_real = X.sum(axis=0).to_numpy()

    # plot
    t = years.to_numpy()

    plt.figure(figsize=(10, 5))

    if plot_spaghetti:
        for i in range(n_paths):
            plt.plot(t, total_paths[i, :], color="indianred", alpha=0.05, linewidth=1)

    plt.fill_between(t, total_p05, total_p95, color="indianred", alpha=0.20, label="Simulé (P5–P95)")
    plt.plot(t, total_mean, color="maroon", linewidth=2, label="Simulé (moyenne)")
    plt.plot(t, total_real, color="black", linewidth=2.5, marker="o", label="Réel (total)")

    plt.title(f"Total Scope12 — GBM multivarié corrélé + shrinkage (λ={shrink_lambda}) — {start_year}→{end_year}")
    plt.xlabel("Année")
    plt.ylabel("Scope12 total")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "calibration": cal,
        "paths": paths,
        "total_paths": total_paths,
        "total_real": pd.Series(total_real, index=years),
    }



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

#kyoto = kyoto[kyoto["Scenario"] != 'Low demand']

restricted = kyoto[kyoto["Sector"] == "Kyoto Gases"]

#%%


def calibrate_gbm_1d_on_total(pivot: pd.DataFrame, start_year: int = 2018, end_year: int = 2023, dt: float = 1.0):
    years = pd.Index(pivot.columns).map(lambda c: int(c))
    X = pivot.copy()
    X.columns = years
    X = X.reindex(sorted(X.columns), axis=1).astype(float)

    X = X.loc[:, (X.columns >= start_year) & (X.columns <= end_year)]
    total = X.sum(axis=0, skipna=True)

    r = np.log(total).diff().dropna()
    if len(r) < 2:
        raise ValueError("Pas assez d'incréments pour estimer une variance (>=2).")

    m = float(r.mean())
    v = float(r.var(ddof=1))
    sigma = np.sqrt(v / dt)
    mu = (m / dt) + 0.5 * sigma**2
    return mu, sigma, total, r

def softmax_logweights(logw: pd.Series) -> pd.Series:
    m = np.max(logw.values)
    w = np.exp(logw.values - m)
    return pd.Series(w / w.sum(), index=logw.index)

def simulate_gbm_1d(E0: float, mu: float, sigma: float, years: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_steps = len(years) - 1
    eps = rng.standard_normal(n_steps)
    r = (mu - 0.5 * sigma**2) + sigma * eps
    logE = np.log(E0) + np.concatenate([[0.0], np.cumsum(r)])
    return pd.Series(np.exp(logE), index=years, name="Simulated")

def extract_ngfs_decarb_rates(ngfs_df: pd.DataFrame, sector: str, scenario: str) -> pd.Series:
    row = ngfs_df[(ngfs_df["Sector"] == sector) & (ngfs_df["Scenario"] == scenario)]
    if len(row) != 1:
        raise ValueError(f"NGFS: attendu 1 ligne pour sector={sector}, scenario={scenario}, trouvé {len(row)}")
    row = row.iloc[0]

    # récupère toutes les colonnes "année"
    vals = {}
    for c in ngfs_df.columns:
        try:
            y = int(c)
        except Exception:
            continue
        v = row[c]
        if pd.notna(v):
            vals[y] = float(v)

    levels = pd.Series(vals).sort_index()
    if levels.empty:
        raise ValueError(f"NGFS: aucune colonne année exploitable pour {scenario}/{sector}")
    if (levels <= 0).any():
        raise ValueError("NGFS contient des valeurs <=0, impossible de calculer des log-changes.")

    rates = np.log(levels).diff().dropna()   # index = année courante
    rates.name = scenario
    return rates

def rebuild_rebased_path_from_rates(E0: float, rates: pd.Series, years: np.ndarray) -> pd.Series:
    years = pd.Index(years).astype(int)
    out = pd.Series(index=years, dtype=float)
    out.iloc[0] = float(E0)

    for i in range(1, len(years)):
        y = int(years[i])
        prev = out.iloc[i - 1]
        if pd.isna(prev):
            out.iloc[i] = np.nan
            continue
        if y in rates.index:
            out.iloc[i] = prev * float(np.exp(rates.loc[y]))
        else:
            # si pas de taux pour cette année, on laisse NaN (ou tu peux forward-fill le dernier taux)
            out.iloc[i] = np.nan
    return out

def scenario_loglikelihood_from_rates(sim_E: pd.Series, ngfs_rates: pd.Series, sigma: float):
    sim_r = np.log(sim_E).diff().dropna()
    common_years = sim_r.index.intersection(ngfs_rates.index)
    if len(common_years) == 0:
        raise ValueError("Aucune année commune entre simulation et taux NGFS.")
    e = (sim_r.loc[common_years] - ngfs_rates.loc[common_years]).to_numpy()
    T = len(e)
    ll = -0.5 * (T * np.log(2 * np.pi * sigma**2) + (e @ e) / (sigma**2))
    return ll

def posterior_over_scenarios_from_rates(ngfs_df: pd.DataFrame, sim_E: pd.Series, sector: str, sigma: float):
    scenarios = sorted(ngfs_df.loc[ngfs_df["Sector"] == sector, "Scenario"].unique().tolist())
    if not scenarios:
        raise ValueError(f"Aucun scénario trouvé pour Sector={sector}")

    logw = {}
    for sc in scenarios:
        rates = extract_ngfs_decarb_rates(ngfs_df, sector=sector, scenario=sc)
        logw[sc] = scenario_loglikelihood_from_rates(sim_E, rates, sigma=sigma)

    logw = pd.Series(logw).sort_values(ascending=False)
    post = softmax_logweights(logw)
    return post, logw

def run_sim_and_filter(ngfs_df: pd.DataFrame, mu_kg: float, sigma_kg: float, E0_2020_real: float,
                       sector: str = "Kyoto Gases", start_year: int = 2020, end_year: int = 2050, seed: int = 42):
    years = np.arange(start_year, end_year + 1, dtype=int)
    sim_E = simulate_gbm_1d(E0=E0_2020_real, mu=mu_kg, sigma=sigma_kg, years=years, seed=seed)

    post, logw = posterior_over_scenarios_from_rates(ngfs_df, sim_E, sector=sector, sigma=sigma_kg)

    scenarios = post.sort_values(ascending=False).index.tolist()  # tous, triés
    
    plt.figure(figsize=(10, 5))
    plt.plot(sim_E.index, sim_E.values, color="black", linewidth=2.5, label="Trajectoire simulée (GBM)")
    
    for sc in scenarios:
        rates = extract_ngfs_decarb_rates(ngfs_df, sector=sector, scenario=sc)
        rebased = rebuild_rebased_path_from_rates(E0_2020_real, rates, years)
        plt.plot(rebased.index, rebased.values, linewidth=1.2, alpha=0.35)  # pas de label pour éviter une légende géante
    
    plt.title(f"{sector} — GBM {start_year}→{end_year} + NGFS rebasés (tous scénarios)")
    plt.xlabel("Année")
    plt.ylabel("Émissions")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    out = pd.DataFrame({"posterior_prob": post, "log_likelihood": logw}).sort_values("posterior_prob", ascending=False)
    return sim_E, out

mu_kg, sigma_kg, total_series, total_log_returns = calibrate_gbm_1d_on_total(pivot, 2018, 2023)
E0_2020_real = float(pivot["2020"].sum())  # ou pivot[2020].sum() selon tes colonnes

sim_E, scenario_ranking = run_sim_and_filter(
    ngfs_df=restricted,
    mu_kg=mu_kg,
    sigma_kg=sigma_kg,
    E0_2020_real=E0_2020_real,
    sector="Kyoto Gases",
    start_year=2020,
    end_year=2050,
    seed=1
)
print(scenario_ranking.head(10))


#%%%%%%% New model

def calibrate_gbm_by_sector(
    df: pd.DataFrame,
    value_col: str = "Scope12",
    instrument_col: str = "Instrument",
    sector_col: str = "GICS Sector Name",
    time_col: str = "Year",
    dt: float = 1.0,
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Calibre un GBM commun (mu, sigma) par secteur en poolant les log-rendements
    des entreprises du secteur.

    Retourne un DataFrame avec mu_hat, sigma_hat, n_returns, n_firms.
    """

    d = df[[instrument_col, sector_col, time_col, value_col]].copy()

    # S'assure que Year est numérique et triable
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")

    # Filtre valeurs utilisables (positives et non-nulles)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[instrument_col, sector_col, time_col, value_col])
    d = d[d[value_col] > 0]

    # Tri par entreprise/temps
    d = d.sort_values([sector_col, instrument_col, time_col])

    # Crée la valeur précédente par entreprise (dans le même secteur, mais instrument suffit ici)
    d["prev_value"] = d.groupby([sector_col, instrument_col])[value_col].shift(1)
    d["prev_time"] = d.groupby([sector_col, instrument_col])[time_col].shift(1)

    # Ne garde que les incréments consécutifs en temps (ici: années consécutives)
    # Si tes années ne sont pas forcément consécutives, tu peux plutôt accepter tout écart
    # et mettre dt_effectif = (Year - prev_year) * dt.
    d["delta_year"] = d[time_col] - d["prev_time"]
    d = d[(d["prev_value"] > 0) & (d["delta_year"] == 1)]

    # Log-rendements
    d["log_return"] = np.log(d[value_col] / d["prev_value"])

    # Agrégation / MLE par secteur
    out = []
    for sector, g in d.groupby(sector_col):
        r = g["log_return"].to_numpy()
        n = r.size
        n_firms = g[instrument_col].nunique()

        if n < min_obs:
            out.append({
                sector_col: sector,
                "mu_hat": np.nan,
                "sigma_hat": np.nan,
                "n_returns": int(n),
                "n_firms": int(n_firms),
            })
            continue

        m = r.mean()              # estimate of (mu - 0.5 sigma^2) * dt
        v_mle = ((r - m) ** 2).mean()  # MLE variance (divide by n, not n-1)

        sigma_hat = np.sqrt(v_mle / dt)
        mu_hat = (m / dt) + 0.5 * sigma_hat**2

        out.append({
            sector_col: sector,
            "mu_hat": float(mu_hat),
            "sigma_hat": float(sigma_hat),
            "n_returns": int(n),
            "n_firms": int(n_firms),
        })

    return pd.DataFrame(out).sort_values("n_returns", ascending=False)


params_scope12 = calibrate_gbm_by_sector(df_fi, value_col="Scope12")

#%%

def gbm_params_from_series(year: np.ndarray, x: np.ndarray):
    """
    Calibre (mu, sigma) d'un GBM à partir d'une série (year, x) positive.
    Accepte des gaps: dt = year[t]-year[t-1].
    Retourne (mu_hat, sigma_hat, n_returns).
    """
    # tri
    order = np.argsort(year)
    year = year[order].astype(float)
    x = x[order].astype(float)

    # filtre x>0
    mask = np.isfinite(year) & np.isfinite(x) & (x > 0)
    year, x = year[mask], x[mask]
    if len(x) < 2:
        return np.nan, np.nan, 0

    # increments
    dt = np.diff(year)
    valid = (dt > 0) & np.isfinite(dt) & np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (x[1:] > 0) & (x[:-1] > 0)
    if valid.sum() < 2:
        # avec 1 seul rendement, sigma non identifiable
        r = np.log(x[1:][valid] / x[:-1][valid])
        if r.size == 1:
            return np.nan, np.nan, 1
        return np.nan, np.nan, 0

    dt = dt[valid]
    r = np.log(x[1:][valid] / x[:-1][valid])  # log-return sur dt variable

    # MLE avec dt variable:
    # r_k ~ N( (mu - 0.5 sigma^2) dt_k, sigma^2 dt_k )
    # Estimation fermée en 2 étapes:
    # 1) estimate a = (mu - 0.5 sigma^2) via WLS: r_k = a dt_k + eps, Var(eps)=sigma^2 dt_k
    #    => WLS poids 1/dt_k, a_hat = sum(r_k)/sum(dt_k)
    a_hat = r.sum() / dt.sum()

    # 2) MLE sigma^2: sigma^2_hat = (1/n) * sum( (r_k - a_hat dt_k)^2 / dt_k )
    n = r.size
    sigma2_hat = (( (r - a_hat * dt) ** 2 ) / dt).mean()
    sigma_hat = np.sqrt(sigma2_hat)

    mu_hat = a_hat + 0.5 * sigma2_hat
    return float(mu_hat), float(sigma_hat), int(n)


def calibrate_by_firm(df: pd.DataFrame,
                      value_col="Scope12",
                      instrument_col="Instrument",
                      sector_col="GICS Sector Name",
                      time_col="Year") -> pd.DataFrame:
    d = df[[instrument_col, sector_col, time_col, value_col]].copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[instrument_col, sector_col, time_col, value_col])
    d = d[d[value_col] > 0]

    rows = []
    for (sector, inst), g in d.groupby([sector_col, instrument_col]):
        mu, sigma, nret = gbm_params_from_series(g[time_col].to_numpy(), g[value_col].to_numpy())
        rows.append({
            sector_col: sector,
            instrument_col: inst,
            "mu_hat": mu,
            "sigma_hat": sigma,
            "sigma2_hat": (sigma**2 if np.isfinite(sigma) else np.nan),
            "n_returns": nret,
            "n_points": int(g.shape[0]),
            "year_min": float(g[time_col].min()),
            "year_max": float(g[time_col].max()),
        })

    firm_params = pd.DataFrame(rows)
    # Garde des calibrations fiables: au moins 2 rendements pour estimer sigma
    firm_params = firm_params[firm_params["n_returns"] >= 2].reset_index(drop=True)
    return firm_params


# --- Usage
firm_params = calibrate_by_firm(df_fi, value_col="Scope12")
sector_summary = firm_params.groupby("GICS Sector Name").agg(
     n_firms=("Instrument", "nunique"),
     mu_median=("mu_hat", "median"),
     mu_p10=("mu_hat", lambda s: s.quantile(0.10)),
     mu_p90=("mu_hat", lambda s: s.quantile(0.90)),
     sigma_median=("sigma_hat", "median"),
     sigma_p10=("sigma_hat", lambda s: s.quantile(0.10)),
     sigma_p90=("sigma_hat", lambda s: s.quantile(0.90)),
 ).reset_index()

#%


def plot_distributions(firm_params: pd.DataFrame, sector_col="GICS Sector Name"):
    fp = firm_params.dropna(subset=["mu_hat", "sigma_hat"]).copy()

    # Optionnel: retirer outliers extrêmes pour la lisibilité (winsorisation simple)
    for c in ["mu_hat", "sigma_hat"]:
        lo, hi = fp[c].quantile([0.01, 0.99])
        fp = fp[(fp[c] >= lo) & (fp[c] <= hi)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    sns.violinplot(data=fp, x=sector_col, y="mu_hat", inner="box", cut=0, ax=axes[0])
    axes[0].set_title("Distribution intra-secteur des μ calibrés (GBM) — Scope12")
    axes[0].set_ylabel("μ̂ (par an)")

    sns.violinplot(data=fp, x=sector_col, y="sigma_hat", inner="box", cut=0, ax=axes[1])
    axes[1].set_title("Distribution intra-secteur des σ calibrés (GBM) — Scope12")
    axes[1].set_ylabel("σ̂ (par an)")

    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()

#%%

def plot_global_mu_sigma_distributions(
    firm_params: pd.DataFrame,
    clip_quantiles=(0.01, 0.99),
    min_returns=2,
    bins=40,
):
    """
    Plot la forme de la distribution de mu_hat et sigma_hat (tous secteurs confondus).
    - clip_quantiles: winsorisation pour la lisibilité (None pour désactiver)
    - min_returns: filtre pour éviter des estimations trop instables
    """

    fp = firm_params.copy()
    fp = fp.dropna(subset=["mu_hat", "sigma_hat"])
    fp = fp[fp["n_returns"] >= min_returns]

    # Optionnel mais recommandé: enlever infinis/valeurs absurdes
    fp = fp[np.isfinite(fp["mu_hat"]) & np.isfinite(fp["sigma_hat"])]
    fp = fp[fp["sigma_hat"] >= 0]

    # Winsorisation légère pour voir la "forme" sans être écrasé par quelques outliers
    if clip_quantiles is not None:
        qlo, qhi = clip_quantiles
        mu_lo, mu_hi = fp["mu_hat"].quantile([qlo, qhi])
        sg_lo, sg_hi = fp["sigma_hat"].quantile([qlo, qhi])
        fp = fp[(fp["mu_hat"].between(mu_lo, mu_hi)) & (fp["sigma_hat"].between(sg_lo, sg_hi))]

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Distribution de mu
    sns.histplot(fp["mu_hat"], bins=bins, stat="density", ax=axes[0], color="steelblue", alpha=0.35)
    sns.kdeplot(fp["mu_hat"], ax=axes[0], color="steelblue", linewidth=2)
    axes[0].axvline(fp["mu_hat"].median(), color="black", linestyle="--", linewidth=1, label=f"médiane={fp['mu_hat'].median():.3g}")
    axes[0].set_title("Distribution de μ̂ (tous secteurs confondus)")
    axes[0].set_xlabel("μ̂ (par an)")
    axes[0].set_ylabel("densité")
    axes[0].legend()

    # --- Distribution de sigma
    sns.histplot(fp["sigma_hat"], bins=bins, stat="density", ax=axes[1], color="darkorange", alpha=0.35)
    sns.kdeplot(fp["sigma_hat"], ax=axes[1], color="darkorange", linewidth=2)
    axes[1].axvline(fp["sigma_hat"].median(), color="black", linestyle="--", linewidth=1, label=f"médiane={fp['sigma_hat'].median():.3g}")
    axes[1].set_title("Distribution de σ̂ (tous secteurs confondus)")
    axes[1].set_xlabel("σ̂ (par an)")
    axes[1].set_ylabel("densité")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Petit résumé numérique (utile pour interpréter la forme)
    summary = fp[["mu_hat", "sigma_hat"]].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).T
    return fp, summary

def compute_log_returns_all_firms(
    df: pd.DataFrame,
    value_col="Scope12",
    instrument_col="Instrument",
    time_col="Year",
    keep_gaps=True,
):
    """
    Retourne un DataFrame de log-rendements poolés sur toutes les entreprises.
    - keep_gaps=True: accepte dt variable (FY2018 -> FY2020 donne dt=2)
    - keep_gaps=False: ne garde que les années consécutives (dt=1)
    """
    d = df[[instrument_col, time_col, value_col]].copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[instrument_col, time_col, value_col])
    d = d[d[value_col] > 0]
    d = d.sort_values([instrument_col, time_col])

    d["prev_value"] = d.groupby(instrument_col)[value_col].shift(1)
    d["prev_time"] = d.groupby(instrument_col)[time_col].shift(1)
    d["dt"] = d[time_col] - d["prev_time"]

    # valid increments
    valid = (
        (d["prev_value"] > 0) &
        np.isfinite(d["prev_value"]) &
        np.isfinite(d[value_col]) &
        np.isfinite(d["dt"]) &
        (d["dt"] > 0)
    )
    d = d[valid].copy()

    if not keep_gaps:
        d = d[d["dt"] == 1].copy()

    d["log_return"] = np.log(d[value_col] / d["prev_value"])
    d["log_return_annualized"] = d["log_return"] / d["dt"]

    return d[[instrument_col, time_col, "dt", "log_return", "log_return_annualized"]]


def plot_log_return_distributions(
    returns_df: pd.DataFrame,
    clip_quantiles=(0.01, 0.99),
    bins=60
):
    """
    Plot la distribution des log-rendements poolés:
    - log_return (sur dt)
    - log_return_annualized (par an)
    """
    r = returns_df.copy()
    r = r.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_return", "log_return_annualized"])

    # Winsorisation légère pour voir la forme (désactive en passant clip_quantiles=None)
    if clip_quantiles is not None:
        qlo, qhi = clip_quantiles
        lo1, hi1 = r["log_return"].quantile([qlo, qhi])
        lo2, hi2 = r["log_return_annualized"].quantile([qlo, qhi])
        r = r[r["log_return"].between(lo1, hi1) & r["log_return_annualized"].between(lo2, hi2)]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(r["log_return"], bins=bins, stat="density", ax=axes[0], color="slateblue", alpha=0.35)
    sns.kdeplot(r["log_return"], ax=axes[0], color="slateblue", linewidth=2)
    axes[0].axvline(r["log_return"].median(), color="black", linestyle="--", linewidth=1,
                    label=f"médiane={r['log_return'].median():.3g}")
    axes[0].set_title("Distribution des log-rendements r = log(Xt+dt / Xt)")
    axes[0].set_xlabel("log-rendement (sur dt)")
    axes[0].set_ylabel("densité")
    axes[0].legend()

    sns.histplot(r["log_return_annualized"], bins=bins, stat="density", ax=axes[1], color="seagreen", alpha=0.35)
    sns.kdeplot(r["log_return_annualized"], ax=axes[1], color="seagreen", linewidth=2)
    axes[1].axvline(r["log_return_annualized"].median(), color="black", linestyle="--", linewidth=1,
                    label=f"médiane={r['log_return_annualized'].median():.3g}")
    axes[1].set_title("Distribution des log-rendements annualisés r/dt")
    axes[1].set_xlabel("log-rendement annualisé")
    axes[1].set_ylabel("densité")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return r

#%%

def compute_log_returns_dt1(df, value_col="Scope12", instrument_col="Instrument", time_col="Year"):
    d = df[[instrument_col, time_col, value_col]].copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[instrument_col, time_col, value_col])
    d = d[d[value_col] > 0]
    d = d.sort_values([instrument_col, time_col])

    d["prev_value"] = d.groupby(instrument_col)[value_col].shift(1)
    d["prev_year"] = d.groupby(instrument_col)[time_col].shift(1)
    d["dt"] = d[time_col] - d["prev_year"]

    d = d[(d["prev_value"] > 0) & np.isfinite(d["prev_value"]) & np.isfinite(d["dt"]) & (d["dt"] == 1)]
    d["log_return"] = np.log(d[value_col] / d["prev_value"])
    return d["log_return"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def normal_pdf(x, mean, std):
    var = std**2
    return (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)


def plot_logreturn_empirical_vs_fits(
    df,
    firm_params,
    value_col="Scope12",
    clip_quantiles=(0.01, 0.99),
    bins=60,
):
    # --- Log-returns dt=1
    r = compute_log_returns_dt1(df, value_col=value_col)
    if r.size < 5:
        raise ValueError("Pas assez de log-returns (dt=1) pour tracer une distribution.")

    # Option: winsorisation légère pour la lisibilité (désactive avec clip_quantiles=None)
    if clip_quantiles is not None:
        qlo, qhi = clip_quantiles
        lo, hi = np.quantile(r, [qlo, qhi])
        r_plot = r[(r >= lo) & (r <= hi)]
    else:
        r_plot = r

    # --- Fit Normal MLE sur les log-returns: mean = m_hat, std = s_hat (MLE -> ddof=0)
    m_mle = float(r.mean())
    s_mle = float(np.sqrt(((r - m_mle) ** 2).mean()))
    if s_mle <= 0:
        raise ValueError("Variance nulle sur les log-returns, impossible de fitter une densité normale.")

    # --- Fit "médianes entreprise": median(mu_hat), median(sigma_hat) -> m = mu - 0.5 sigma^2 (dt=1)
    fp = firm_params.dropna(subset=["mu_hat", "sigma_hat"]).copy()
    fp = fp[np.isfinite(fp["mu_hat"]) & np.isfinite(fp["sigma_hat"]) & (fp["sigma_hat"] >= 0)]
    fp = fp[fp["n_returns"] >= 2] if "n_returns" in fp.columns else fp

    mu_med = float(fp["mu_hat"].median())
    sigma_med = float(fp["sigma_hat"].median())
    m_med = mu_med - 0.5 * sigma_med**2
    s_med = sigma_med

    # --- Grid pour les courbes
    x_min, x_max = np.quantile(r_plot, [0.001, 0.999])
    pad = 0.1 * (x_max - x_min)
    x = np.linspace(x_min - pad, x_max + pad, 600)

    pdf_mle = normal_pdf(x, m_mle, s_mle)
    pdf_med = normal_pdf(x, m_med, s_med)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 5))

    # Empirique: hist + KDE
    sns.histplot(r_plot, bins=bins, stat="density", color="slateblue", alpha=0.25, label="Empirique (hist)")
    sns.kdeplot(r_plot, color="slateblue", linewidth=2, label="Empirique (KDE)")

    # Fits
    plt.plot(x, pdf_mle, color="black", linewidth=2,
             label=f"Normal fit MLE sur r : m={m_mle:.3g}, s={s_mle:.3g}")
    plt.plot(x, pdf_med, color="darkorange", linewidth=2, linestyle="--",
             label=f"Normal via médianes (μ̃,σ̃) : m={m_med:.3g}, s={s_med:.3g}")

    plt.title("Log-returns (dt=1) — empirique vs ajustements (GBM)")
    plt.xlabel("r = log(Xt+1 / Xt)")
    plt.ylabel("densité")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "n_returns": int(r.size),
        "m_mle": m_mle,
        "s_mle": s_mle,
        "mu_med": mu_med,
        "sigma_med": sigma_med,
        "m_from_medians": m_med,
        "s_from_medians": s_med,
    }

#%%

def firm_log_returns_dt1(df,
                         value_col="Scope12",
                         instrument_col="Instrument",
                         sector_col="GICS Sector Name",
                         time_col="Year"):
    d = df[[instrument_col, sector_col, time_col, value_col]].copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[instrument_col, sector_col, time_col, value_col])
    d = d[d[value_col] > 0]
    d = d.sort_values([instrument_col, time_col])

    d["prev_value"] = d.groupby(instrument_col)[value_col].shift(1)
    d["prev_year"] = d.groupby(instrument_col)[time_col].shift(1)
    d["dt"] = d[time_col] - d["prev_year"]

    d = d[(d["prev_value"] > 0) & (d["dt"] == 1)]
    d["log_return"] = np.log(d[value_col] / d["prev_value"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_return"])
    return d[[instrument_col, sector_col, time_col, "log_return"]]


def sector_returns_panel(df_returns,
                         sector_col="GICS Sector Name",
                         time_col="Year",
                         agg="mean",
                         min_firms_per_sector_year=3):
    """
    Construit r_{t,sector} en agrégeant les log-returns d'entreprises d'un secteur pour une année donnée.
    Retourne un panel (index=Year, columns=Sector).
    """
    g = df_returns.groupby([time_col, sector_col])["log_return"]

    if agg == "mean":
        sector_rt = g.mean()
    elif agg == "median":
        sector_rt = g.median()
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    counts = g.size().rename("n_firms")
    sector_rt = sector_rt.to_frame("r").join(counts).reset_index()

    # filtre les (Year, Sector) avec trop peu d'entreprises
    sector_rt = sector_rt[sector_rt["n_firms"] >= min_firms_per_sector_year]

    panel = sector_rt.pivot(index=time_col, columns=sector_col, values="r").sort_index()
    return panel


def calibrate_common_mu_and_Sigma(panel: pd.DataFrame):
    """
    panel: DataFrame (T x S) de log-returns sectoriels (dt=1).
    Estime mu_* (commun) et Sigma (SxS) par MLE en supposant:
      r_t ~ N(m, Sigma) avec m_s = (mu_* - 0.5 Sigma_ss)
    On fait une estimation simple:
      - Sigma: covariance MLE sur les données centrées (par secteur)
      - mu_*: minimise l'erreur quadratique entre m_s observé et (mu_* - 0.5 Sigma_ss)
    """
    X = panel.dropna(how="all")
    # Option: imputation pairewise (sinon on perd beaucoup de dates). Ici: pairwise cov via pandas.cov
    # mean par secteur (sur les dates où dispo)
    m_hat = X.mean(axis=0)  # (S,)
    # covariance (pairwise complete obs)
    Sigma_hat = X.cov(ddof=0)  # MLE: ddof=0

    diag = np.diag(Sigma_hat.values)
    # m_s ≈ mu_* - 0.5 * Sigma_ss  => mu_* ≈ average_s (m_s + 0.5 Sigma_ss)
    mu_star_hat = float(np.mean(m_hat.values + 0.5 * diag))

    return mu_star_hat, Sigma_hat, m_hat


#%%

# =============================================================================
# res = plot_real_vs_paths_from_params(pivot, params, sector="Energy", start_year=2018, end_year=2023, n_paths=300, seed=1)
# 
# # Exemple:
# res_total = plot_total_real_vs_simulations(pivot, params, start_year=2018, end_year=2023, n_paths=500, seed=1)
# 
# # Exemple:
# res_tot = plot_total_real_vs_simulated_total(pivot, start_year=2018, end_year=2023, n_paths=1000, shrink_lambda=0.4, seed=1)
# 
# plot_distributions(firm_params)
# 
# # Usage:
# fp_used, summary = plot_global_mu_sigma_distributions(firm_params, clip_quantiles=(0.01, 0.99), min_returns=3)
# print(summary)
# 
# # --- Usage
# returns_df = compute_log_returns_all_firms(df_fi, value_col="Scope12", keep_gaps=True)
# returns_used = plot_log_return_distributions(returns_df, clip_quantiles=(0.01, 0.99))
# 
# # Usage:
# stats = plot_logreturn_empirical_vs_fits(df_fi, firm_params, value_col="Scope12", clip_quantiles=(0.01, 0.99))
# print(stats)
# =============================================================================


# ---- Usage
df_r = firm_log_returns_dt1(df_fi, value_col="Scope12")
panel = sector_returns_panel(df_r, agg="mean", min_firms_per_sector_year=3)
mu_star, Sigma, m_by_sector = calibrate_common_mu_and_Sigma(panel)
print(mu_star)
print(Sigma.shape)