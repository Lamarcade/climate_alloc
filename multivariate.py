# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:24:44 2026

@author: LoïcMARCADET
"""

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf, OAS

df = pd.read_excel("Data/history_filtered.xlsx")
years = ["2018","2019","2020","2021","2022","2023"]

df_E = df.set_index("GICS Sector Name")

df_E = df_E.T

def calibrate_gbm(E, shrink="lw", mu_shared=True):
    """
    Calibrates a multidimensional lognormal GBM with annual time steps (dt = 1).

    Model:
        d e_t = diag(e_t) (mu * 1 dt + Sigma dW_t)

    Parameters
    ----------
    E : array-like, shape (n_years, d)
        Annual emission levels, strictly positive.
    shrink : {"lw", "oas"}, default "lw"
        Shrinkage estimator for the covariance (Ledoit-Wolf or OAS).
    mu_shared : bool, default True
        If True, estimate a single scalar mu.
        If False, estimate a vector mu of dimension d.

    Returns
    -------
    mu_out : float or ndarray, shape (d,)
        Estimated drift parameter(s).
    Omega_hat : ndarray, shape (d, d)
        Shrinkage estimate of Sigma Sigma^T.
    Sigma_hat : ndarray, shape (d, d)
        A square root of Omega_hat (Cholesky).
    """
    E = np.asarray(E, dtype=float)

    # Log-increments (dt = 1)
    Y = np.log(E)
    R = np.diff(Y, axis=0)              # (T, d), with T = n_years - 1

    # Shrinkage covariance estimator: Cov(R) ≈ Omega
    if shrink == "lw":
        cov_est = LedoitWolf().fit(R)
    elif shrink == "oas":
        cov_est = OAS().fit(R)
    else:
        raise ValueError("shrink must be 'lw' or 'oas'")

    Omega_hat = cov_est.covariance_     # (d, d)

    # Drift estimation:
    # E[R] = mu * 1 - 0.5 * diag(Omega)
    m = R.mean(axis=0)                  # estimates mu_i - 0.5 Omega_ii
    mu_vec = m + 0.5 * np.diag(Omega_hat)

    if mu_shared:
        mu_out = float(mu_vec.mean())
    else:
        mu_out = mu_vec

    # One possible Sigma for simulation
    eps = 1e-12
    Sigma_hat = np.linalg.cholesky(
        Omega_hat + eps * np.eye(Omega_hat.shape[0])
    )

    return mu_out, Omega_hat, Sigma_hat

mu_LW, Omega_LW, Sigma_LW = calibrate_gbm(df_E, shrink="lw", mu_shared=True)
mu_OAS, Omega_OAS, Sigma_OAS = calibrate_gbm(df_E, shrink="oas", mu_shared=True)
mus, Omega, Sigma = calibrate_gbm(df_E, shrink="oas", mu_shared=False)

#%% Comparaison modèles

def loglik_gaussian(R, mean, Omega):
    """
    R    : (T, d) log-increments
    mean : (d,)
    Omega: (d, d)
    """
    T, d = R.shape
    L = np.linalg.cholesky(Omega)
    Linv = np.linalg.solve(L, np.eye(d))
    Omega_inv = Linv.T @ Linv

    diff = R - mean
    quad = np.einsum("ti,ij,tj->", diff, Omega_inv, diff)
    logdet = 2 * np.sum(np.log(np.diag(L)))

    return -0.5 * (T * (d*np.log(2*np.pi) + logdet) + quad)

# Log-increments annuels
R = np.diff(np.log(df_E.values), axis=0)

T, d = R.shape

mean_LW  = mu_LW * np.ones(R.shape[1]) - 0.5 * np.diag(Omega_LW)
mean_OAS = mu_OAS * np.ones(R.shape[1]) - 0.5 * np.diag(Omega_OAS)
mean_vec = mus - 0.5 * np.diag(Omega)

ll_LW  = loglik_gaussian(R, mean_LW,  Omega_LW)
ll_OAS = loglik_gaussian(R, mean_OAS, Omega_OAS)
ll_vec = loglik_gaussian(R, mean_vec, Omega)

# %% Tableau récapitulatif : métriques vs modèles

def summarize_mu(mu, max_items=6):
    """Affichage lisible de mu (scalaire ou vecteur)."""
    if np.isscalar(mu):
        return float(mu)
    mu = np.asarray(mu).ravel()
    if mu.size <= max_items:
        return "[" + ", ".join(f"{x:.4g}" for x in mu) + "]"
    return f"vecteur (taille={mu.size}) : mean={mu.mean():.4g}, std={mu.std(ddof=0):.4g}, min={mu.min():.4g}, max={mu.max():.4g}"

def build_model_comparison_table(ll_LW, ll_OAS, ll_vec, n_obs, d,
                                 mu_LW, mu_OAS, mus,
                                 Omega_LW, Omega_OAS, Omega):
    """
    Construit un tableau comparant les modèles estimés.
    Les k suivent exactement ta définition :
      - shared: 1 + d(d+1)/2
      - vector: d + d(d+1)/2
    """
    k_shared = 1 + d * (d + 1) // 2
    k_vec    = d + d * (d + 1) // 2

    def aic_bic(loglik, n_obs, k):
        aic = 2*k - 2*loglik
        bic = k*np.log(n_obs) - 2*loglik
        return float(aic), float(bic)

    rows = []

    for name, ll, k, mu, Omega_hat in [
        ("GBM lognormal | Cov: Ledoit-Wolf | μ partagé", ll_LW,  k_shared, mu_LW,  Omega_LW),
        ("GBM lognormal | Cov: OAS        | μ partagé", ll_OAS, k_shared, mu_OAS, Omega_OAS),
        ("GBM lognormal | Cov: OAS        | μ vecteur",  ll_vec, k_vec,    mus,   Omega),
    ]:
        aic, bic = aic_bic(ll, n_obs, k)
        rows.append({
            "Modèle": name,
            "LL": float(ll),
            "n_obs (T)": int(n_obs),
            "d (secteurs)": int(d),
            "k (nb paramètres)": int(k),
            "AIC": aic,
            "BIC": bic,
            "μ estimé (résumé)": summarize_mu(mu),
            "diag(Ω) mean": float(np.mean(np.diag(Omega_hat))),
        })

    table = pd.DataFrame(rows)

    # Mise en forme : on trie par BIC (plus petit = meilleur)
    table = table.sort_values("BIC", ascending=True).reset_index(drop=True)

    return table

n_obs = T  

comparison_table = build_model_comparison_table(
    ll_LW=ll_LW, ll_OAS=ll_OAS, ll_vec=ll_vec,
    n_obs=n_obs, d=d,
    mu_LW=mu_LW, mu_OAS=mu_OAS, mus=mus,
    Omega_LW=Omega_LW, Omega_OAS=Omega_OAS, Omega=Omega
)

print("\n=== Comparaison des modèles (trié par BIC croissant) ===")
print(comparison_table.to_string(index=False))
