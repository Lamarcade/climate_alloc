# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:42:10 2025

@author: LoïcMARCADET
"""

from Modelnu import Modelnu
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
from scipy.stats import multivariate_normal, norm
from matplotlib.backends.backend_pdf import PdfPages
from math import inf
from scipy.stats import entropy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from tqdm import tqdm
from scipy.optimize import root_scalar, minimize

import Config

#%%
class Alloc():
        
    def __init__(self, rf, budget, short_sales = False, rf_params = False, sheet_name = 0, order_returns = False):
        """
        Initializes a Portfolio instance.

        Parameters:
            mu (array-like): Expected returns of the assets.
            sigma (array-like): Covariance matrix of the asset returns.
            rf (float): Risk-free rate.
            short_sales (bool, optional): If True, allows short selling. Defaults to True.
            sectors (DataFrame, optional): DataFrame containing sector information. Defaults to None.
            tickers (list, optional): List of asset tickers. Defaults to None.
            rf_params (bool, optional): If True, includes a risk-free asset. Defaults to False.
        """
        self.rf = rf
        
        self.budget = budget

        self.short_sales = short_sales
        
        # Is the risk-free included in the mean and covariance?
        self.rf_params = rf_params
        
        self.existing_plot = False
        
        self.current_time = 0 
        
        self.emissions_matrix()
        self.emissions_matrix(sim = True, sheet_name = sheet_name)
        K, n, T_em = self.scenario_emissions.shape
        self.time = T_em
        
        self.GICS = Config.GICS.copy()
        self.GICS.remove("Real Estate")
        
        
        if order_returns:
            self.specific_returns(scenario = Config.INDEX2SCENAR[sheet_name])
        else:
            self.return_stats()
        
    def update_year(self, probas):
        self.current_time +=1 
        self.probas = probas
        
    def set_budget(self, budget):
        self.budget = budget
        
    def emissions_matrix(self, sim = False, path = "Data/Simul/Test3/Test3_1.xlsx", sheet_name = 0):
        '''Computes the matrix of sector emissions over time for each scenario'''
        # dim Scenar x Company x Time
        
        #emissions = Config.EM_LAST.copy()
        #emissions.columns = [Config.FUTURE_START_YEAR-1]
        
        emissions = Config.LAST_EM.copy()
        emissions = emissions.drop("Real Estate (n=29)")
        
        nus = Config.NUS_ORDER.copy()
        if sim:
            # Simulated rates
            mus = pd.read_excel(path, index_col = 0, sheet_name = sheet_name)
            em = emissions.copy()
            ind = em.index.str.replace(r"\s*\(n=\d+\)", "", regex=True)
            em.index = ind
            mus = mus.reindex(ind)
            while em.columns[-1] < 2050:       
                max_year = em.columns[-1] +1
                em[max_year] = mus[max_year] * em[max_year-1] / 100 + em[max_year-1]
                em = em.copy()

        else:
            mus = pd.read_excel("Data/scenarios.xlsx", index_col = "Scenario")
            sces = []
            for scenar in mus.index:
                sce = emissions.copy()
                ind = sce.index.str.replace(r"\s*\(n=\d+\)", "", regex=True)
                sce.index = ind
                nus = nus.reindex(ind)
                while sce.columns[-1] < 2050:       
                    max_year = sce.columns[-1] +1
                    sce[max_year] = (nus + mus.loc[scenar, max_year]).squeeze() * sce[max_year-1] / 100 + sce[max_year-1]
                    sce = sce.copy()
                sces.append((scenar, sce))
                
            scenario_emissions = np.stack([df.to_numpy() for (_,df) in sces], axis=0)

        if sim:
            self.sim_emissions = em
            
        #K x n x T [scenar x company x time]
        # Time 2022 to 2050
        else:
            self.scenario_emissions = scenario_emissions
        
        
    def return_stats(self, specific = None):
        '''Computes the historical returns mean and covariance'''
        returns = pd.read_excel("Data/Stoxx_returns.xlsx", index_col = 0)
        returns['GICS Sector Name'] = returns.groupby('Instrument')['GICS Sector Name'].ffill()
        returns = returns[['Date', 'GICS Sector Name', 'Instrument', 'Total Return']]
        returns.columns = ['Date', 'Sector', 'Ticker', 'Return']
        
        returns['Date'] = pd.to_datetime(returns['Date'])
        
        returns = returns.dropna(subset=['Return', 'Sector'])
        
        if specific is not None:
            returns = returns[returns["Ticker"].isin(specific)]
        
        sector_returns = returns.groupby(['Date', 'Sector'])['Return'].mean().unstack()
        sector_returns.drop(columns = ["Real Estate"], inplace = True)
        
        self.mu = sector_returns.mean()

        self.sigma = sector_returns.cov()
        
        self.n = len(self.mu)
        
    def specific_returns(self, scenario):
        returns = pd.read_excel("Data/Stoxx_returns.xlsx", index_col = 0)
        returns['GICS Sector Name'] = returns.groupby('Instrument')['GICS Sector Name'].ffill()
        returns = returns[['Date', 'GICS Sector Name', 'Instrument', 'Total Return']]
        returns.columns = ['Date', 'Sector', 'Ticker', 'Return']
        
        returns['Date'] = pd.to_datetime(returns['Date'])
        
        returns = returns.dropna(subset=['Return', 'Sector'])
        
        self.cumulative_emissions(scenario = scenario)
        
        ce = self.ce.copy()
        
        select = pd.DataFrame()
        if self.rf_params:
            n = len(self.GICS[:-1])
        else:
            n = len(self.GICS)
        
        # Index of emitter ranks
        dico = {a:(i/n) for (i,a) in enumerate(ce.index.str.replace(r"\s*\(n=\d+\)", "", regex=True))}
        
        for i in range(n):
            sector = returns[returns["Sector"] == self.GICS[i]].groupby('Ticker').mean("Return")
            sector.sort_values(by = "Return", ascending = False, inplace = True)
            
            rank = dico[self.GICS[i]]
            
            # First emitter : take the best return and so on with quantiles
            ticker = sector.iloc[int(rank*(len(sector.index)-1))].name
            
            values = returns[returns["Ticker"] == ticker]
            
            select = pd.concat([select, values], axis = 0)
        
        select_returns = select.groupby(['Date', 'Sector'])['Return'].mean().unstack()
        #select_returns.drop(columns = ["Real Estate"], inplace = True)
        
        self.mu = select_returns.mean()

        self.sigma = select_returns.cov()
        
        self.n = len(self.mu)
    
    def simulate_returns(self, n_sim = 1000, horizon = 2050 - Config.FUTURE_START_YEAR):
        return np.random.multivariate_normal(self.mu, self.sigmas, size=(n_sim, horizon))
      
    def get_probas_simulation(self, path ="Data/Simul/Test3/Calib_1.xlsx", sheet_name = 0):
        self.all_probas = pd.read_excel(path, index_col = 0, sheet_name = sheet_name)
       
    def get_rates(self, path = "Data/Simul/Test3/Test3_1.xlsx", sheet_name= 0):
        self.rates = pd.read_excel(path, index_col = 0, sheet_name = sheet_name)
       
    def cut_budget(self, start, end, linear=False, alpha = 1.5, x0 = 14):
        '''Splits the budget into parts in a linear or sigmoidal way'''
        
        def reverse_sigmoid(alpha, T):
            t = np.arange(1, T + 1)
            return 1 / (1 + np.exp(alpha * (t - T / 2)))
        
        def reverse_sigmoid(start, end, i, alpha, x0 = 13):
            # Decreasing sigmoid
            sigma = 1 / (1 + np.exp(-alpha * (i - x0)))
            return start + (end - start) * (sigma)
    
        if linear:
            self.parts = np.full(self.time, self.budget / self.time)
        else:
            
            i_vals = np.arange(self.time)
            s_vals = reverse_sigmoid(start, end, i_vals, alpha, x0 = 14)

            scale_factor = self.budget / np.sum(s_vals)

            self.parts = s_vals * scale_factor
        
    def risk_free_stats(self):
        """
        Adjusts the portfolio statistics to include a risk-free asset.

        Returns:
            Portfolio: Updated Portfolio instance.
        """
        if not(self.rf_params):
            rf_name = "Risk free"
            self.mu.loc[rf_name] = self.rf
            
            self.n = self.n+1

            self.sigma[rf_name] = 0.0            
            self.sigma.loc[rf_name] = 0.0
            self.sigma.loc[rf_name, rf_name] = 0.0
            
            self.rf_params = True
            
            self.GICS += ["Risk-free"]
                
            return self
        
        
#%% Stats

    def get_variance(self, weights):
        """
        Computes the variance of the portfolio.

        Parameters:
            weights (array-like): Asset weights in the portfolio.

        Returns:
            float: Variance of the portfolio.
        """
        return weights.T.dot(self.sigma).dot(weights)
    
    def get_risk(self, weights):
        """
        Computes the risk (standard deviation) of the portfolio.

        Parameters:
            weights (array-like): Asset weights in the portfolio.

        Returns:
            float: Risk of the portfolio.
        """
        
        return np.sqrt(weights.T.dot(self.sigma).dot(weights))
    
    def get_return(self, weights):
        """
        Computes the return of the portfolio.

        Parameters:
            weights (array-like): Asset weights in the portfolio.

        Returns:
            float: Return of the portfolio.
        """
        return weights.T.dot(self.mu)
    
    def get_sharpe(self, weights):
        """
        Computes the Sharpe ratio of the portfolio.

        Parameters:
            weights (array-like): Asset weights in the portfolio.

        Returns:
            float: Sharpe ratio of the portfolio.
        """
        return ((self.get_return(weights) - self.rf)/ self.get_risk(weights))     
    
    def neg_sharpe(self, weights):
        return -self.get_sharpe(weights)
    
    def get_carbon(self, weights):
        # Emissions shape (K, n, T) 
        # Current emissions (K,n)
        # Probas shape (K,)
        # Weights shape (n,)
        # Means shape (n,)
        if self.rf_params:
            w = weights[:-1]
        else:
            w = weights
        means = self.probas.T.dot(self.scenario_emissions[:,:,self.current_time])
        return w.T.dot(means)
    
    def true_carbon(self, weights):
        if self.rf_params:
            w = weights[:-1]
        else:
            w = weights
        return w.T.dot(self.sim_emissions[self.current_time + Config.FUTURE_START_YEAR])
    
#%% 

    def bounds(self, input_bounds = None):
        """
        Creates bounds for the asset weights.

        Parameters:
            input_bounds (list, optional): List of tuples specifying the bounds. Defaults to None.

        Returns:
            list or None: List of bounds or None if short selling is allowed.
        """
        if self.short_sales:
            bounds = None
        elif input_bounds is None:
            bounds = [(0.0, 1.0) for _ in range(self.n)]
        else:
            bounds = input_bounds
        return bounds
    
    def init_weights(self):
        """
        Initializes the weights for the portfolio using equal weights.

        Returns:
            array: Initial weights.
        """
        return(np.ones(self.n)/self.n)
    
    def carbon_constraint(self, max_budget):
        """
        Create a carbon constraint for the optimization.

        Parameters:
            max_budget (float): Maximum allowed carbon emissions for the portfolio.

        Returns:
            dict: Constraint dictionary for optimization.
        """        
        
        # budget = self.parts[self.current_time]
        return({'type': 'ineq', 'fun': lambda w: max_budget - self.get_carbon(w)})
    
    def optimal_portfolio_carbon(self, max_budget, input_bounds = None):
        """
        Calculate the optimal portfolio (Sharpe maximizer) for the given ESG constraints.

        Parameters:
            min_ESG (float): Minimum portfolio ESG score required.
            input_bounds (list, optional): Bounds for the weights. Defaults to None.

        Returns:
            array: Optimal weights.
        """        
        initial_weights = self.init_weights()
        boundaries = self.bounds(input_bounds)
        constraints = [
            self.carbon_constraint(max_budget),
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
        
        result = minimize(self.neg_sharpe, x0 = initial_weights, method='SLSQP', 
                          constraints=constraints, 
                          bounds=boundaries)
        return result.x
    
    def allocate_over_time(self, log=True, update_budget = True):

        K, n, T = self.scenario_emissions.shape
        
        all_weights = []
        
        for t in range(T-1):
            self.current_time = t
            self.probas = self.all_probas.loc[:, t + Config.FUTURE_START_YEAR].values  # 1D array (K,)
    
            #print(self.parts[t], " after updating")
            weights = self.optimal_portfolio_carbon(max_budget=self.parts[t])
            
            if log:
                sharpe = self.get_sharpe(weights)
                ret = self.get_return(weights)
                risk = self.get_risk(weights)
                carbon = self.get_carbon(weights)
                realized = self.true_carbon(weights)
                all_weights.append({
                    'year': t + Config.FUTURE_START_YEAR,
                    'weights': weights,
                    'sharpe': sharpe,
                    'return': ret,
                    'risk': risk,
                    'carbon': carbon,
                    'realized': realized,
                    'allocated_budget': self.parts[t]
                })
                
            if update_budget: 
                #print(self.parts[t+1], " before updating")

                self.redistribute_budget(t, realized)
    
        return all_weights if log else weights
    
    def run_multiple_allocations(self, sim_files, proba_files, update_budget = True):

        assert len(sim_files) == len(proba_files), "Incompatible file numbers"
    
        total_budgets = []
        sharpes = []
    
        for (sim_path, sim_sheet), (proba_path, proba_sheet) in tqdm(zip(sim_files, proba_files), total = 1000, desc = "Simulation number"):
            alloc.cut_budget(start = 5e+07, end = 5e+06, linear=False, alpha = 0.3)
            alloc.emissions_matrix(sim=True, path=sim_path, sheet_name=sim_sheet)
    
            alloc.get_probas_simulation(path=proba_path, sheet_name=proba_sheet)
    
            run = alloc.allocate_over_time(log=True, update_budget = update_budget)
            df_run = pd.DataFrame(run)
    
            total_realized = df_run["realized"].sum()
            total_budgets.append(total_realized)
            
            sharpe_mean = df_run["sharpe"].mean()
            
            sharpes.append(sharpe_mean)
    
        return sharpes, total_budgets
    
    def redistribute_budget(self, t, realized):
        
        delta = realized - self.parts[t]

        if t+1 < len(self.parts) and delta != 0:
            future_sum = self.parts[t+1:].sum()
            if future_sum > 0:
                self.parts[t+1:] -= delta * (self.parts[t+1:] / future_sum)
            else:
                self.parts[t+1:] = 0.0

    def compare_strat(self, sim_files, proba_files, update_budget = True):

        assert len(sim_files) == len(proba_files), "Incompatible file numbers"
    
        tb, tbn = [], []
        sh, shn = [], []
        tr, trn = [], []
    
        for (sim_path, sim_sheet), (proba_path, proba_sheet) in tqdm(zip(sim_files, proba_files), total = 1000, desc = "Simulation number"):
            alloc.cut_budget(start = 5e+07, end = 5e+06, linear=False, alpha = 0.3)
            alloc.emissions_matrix(sim=True, path=sim_path, sheet_name=sim_sheet)
    
            alloc.get_probas_simulation(path=proba_path, sheet_name=proba_sheet)
    
            run = alloc.allocate_over_time(log=True, update_budget = update_budget)
            df_run = pd.DataFrame(run)
    
            total_realized = df_run["realized"].sum()
            tb.append(total_realized)
            
            sharpe_mean = df_run["sharpe"].mean()
            
            returns = df_run["return"].sum()
            
            sh.append(sharpe_mean)
            tr.append(returns)
            
            history_naive = alloc.allocate_naive_investor(log=True, update_budget = update_budget)
            df_naive = pd.DataFrame(history_naive)
            
            total_realized = df_run["realized"].sum()
            tbn.append(total_realized)
            
            sharpe_mean = df_naive["sharpe"].mean()
            
            returns = df_naive["return"].sum()
            
            shn.append(sharpe_mean)
            trn.append(returns)
    
        return tb, tbn, sh, shn, tr, trn

#%%

    def cumulative_emissions(self, scenario = "Net Zero 2050"):
        mus = pd.read_excel("Data/scenarios.xlsx", index_col = "Scenario")

        emissions = Config.LAST_EM.copy()
        emissions = emissions.drop("Real Estate (n=29)")
        sce = emissions.copy()
        while sce.columns[-1] < 2050:       
            max_year = sce.columns[-1] +1
            sce[max_year] = mus.loc[scenario, max_year] * sce[max_year-1] / 100 + sce[max_year-1]
            sce = sce.copy()
        cumulative_emissions = sce.sum(axis = 1)
        cumulative_emissions.sort_values(ascending = False, inplace = True)
        
        self.ce = cumulative_emissions
        
        plt.figure()
        ax = cumulative_emissions.plot.barh()
        ax.bar_label(ax.containers[0], fmt = "%.3g")
        plt.title(f"Cumulative emissions for {scenario}")
        
    def dist_returns(self):
        returns = pd.read_excel("Data/Stoxx_returns.xlsx", index_col = 0)
        returns['GICS Sector Name'] = returns.groupby('Instrument')['GICS Sector Name'].ffill()
        returns = returns[['Date', 'GICS Sector Name', 'Instrument', 'Total Return']]
        returns.columns = ['Date', 'Sector', 'Ticker', 'Return']

        returns['Date'] = pd.to_datetime(returns['Date'])

        returns = returns.dropna(subset=['Return', 'Sector'])
        n = len(GICS[:-1])
        fig, axes = plt.subplots(2, n//2 + n%2)
        for i in range(n):
            test = returns[returns["Sector"] == GICS[i]].groupby('Ticker').mean("Return")
            ax = axes[i%2, i//2]
            sns.histplot(test, ax = ax)
            ax.set_title(GICS[i])
        fig.suptitle("Distribution of returns inside sectors")
   
#%%

    def allocate_naive_investor(self, log=True, update_budget=True):
    
        K, n, T = self.scenario_emissions.shape
        all_weights = []
    
        # Linear budget
        self.parts = np.full(self.time, self.budget / self.time)
    
        prev_emissions = self.sim_emissions[Config.FUTURE_START_YEAR]
    
        for t in range(T-1):
            self.current_time = t
    
            current_emissions = self.sim_emissions[t + Config.FUTURE_START_YEAR]
    
            if t > 0:

                decarb_rate = (current_emissions - prev_emissions) / prev_emissions

                expected_emissions = current_emissions * (1 + decarb_rate)
            else:
                # First year
                expected_emissions = current_emissions.copy()
    
            prev_emissions = current_emissions.copy()
    
            # Expected constraint
            def carbon_estimate(weights):
                if self.rf_params:
                    w = weights[:-1]
                else:
                    w = weights
                return w.T.dot(expected_emissions.values)
    
            constraints = [
                {'type': 'ineq', 'fun': lambda w, mb=self.parts[t]: mb - carbon_estimate(w)},
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            initial_weights = self.init_weights()
            boundaries = self.bounds()
    
            result = minimize(self.neg_sharpe, x0=initial_weights,
                              method='SLSQP',
                              constraints=constraints,
                              bounds=boundaries)
            weights = result.x
    
            if log:
                sharpe = self.get_sharpe(weights)
                ret = self.get_return(weights)
                risk = self.get_risk(weights)

                carbon_naive = carbon_estimate(weights)
                realized = self.true_carbon(weights)
    
                all_weights.append({
                    'year': t + Config.FUTURE_START_YEAR,
                    'weights': weights,
                    'sharpe': sharpe,
                    'return': ret,
                    'risk': risk,
                    'carbon_naive': carbon_naive,
                    'realized': realized,
                    'allocated_budget': self.parts[t]
                })
    
            if update_budget:
                    
                self.redistribute_budget(t, realized)
                
        return all_weights if log else weights
                        
   
    
#%%

# Below : about 4 times as many emissions in the start as in the end

# NZ : 32 times as many emissions in the start as in the end

scenar_used = 5
budget = 1e+09

#alloc = Alloc(rf = 0.003, budget=budget, sheet_name = scenar_used, order_returns = True)
alloc = Alloc(rf = 0, budget=budget, sheet_name = scenar_used, order_returns = True)
alloc.risk_free_stats()
alloc.cut_budget(start = 5e+07, end = 5e+06, linear=False, alpha = 0.3)

abb = Config.INDEX2ABB[scenar_used]

# Simulations with the lowest KL for each scenario for 10 runs
kl_scenar_best_sims = {0:3, 1:6, 2:10, 3:1, 4:4, 5:6, 6:9}

GICS = Config.GICS.copy()
GICS.remove("Real Estate")
GICS += ["Risk-free"]
palette = sns.color_palette("tab10", n_colors=len(GICS))
color_mapping = dict(zip(GICS, palette))  # {sector_name: color}

#%%
def test_plot():
    alloc.get_probas_simulation(f"Data/Simul/Test3/Calib_{kl_scenar_best_sims[scenar_used]}.xlsx", sheet_name = scenar_used)
    history = alloc.allocate_over_time()
    
    carbs = [i["carbon"] for i in history]
    real = [i["realized"] for i in history]
    
    plt.plot(alloc.parts, marker='o')
    plt.plot(carbs, label = "Expected")
    plt.plot(real, label = 'Realized')
    plt.title("Budget split")
    plt.ylabel("Budget for year (tCO2eq)")
    plt.xlabel("Year")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"Figs/Alloc/{abb}/carbon_traj_{abb}_{int(budget/1e+06)}Mt.png")
    plt.show()

    # Plot 1
    sector_emissions = pd.DataFrame(
        alloc.scenario_emissions[scenar_used, :, :],
        index=GICS,
        columns=range(Config.FUTURE_START_YEAR - 1, 2051)
    )
    
    emissions_df = sector_emissions.T.reset_index().melt(id_vars="index", var_name="Sector", value_name="Emissions")
    emissions_df.rename(columns={"index": "Year"}, inplace=True)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=emissions_df, x="Year", y="Emissions", hue="Sector", palette=color_mapping)
    plt.title(f"Emissions by sector {Config.INDEX2SCENAR[scenar_used]}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Figs/Alloc/{abb}/emissions_{abb}.png")
    plt.show()


    # Plot 2
    weights_df = pd.DataFrame({
        "Year": [i["year"] for i in history],
        "Weights": [i["weights"] for i in history]
    })
    
    weights_expanded = pd.DataFrame(weights_df['Weights'].tolist(), columns=GICS)
    weights_expanded['Year'] = weights_df['Year']
    
    weights_melted = weights_expanded.melt(id_vars='Year', var_name='Sector', value_name='Weight')
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=weights_melted, x="Year", y="Weight", hue="Sector", palette=color_mapping)
    plt.title("Weight evolution by sector")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Figs/Alloc/{abb}/weights_{abb}_{int(budget/1e+06)}Mt.png")
    plt.show()

#%%


def distribution_tests():
    sim_files= [(f"Data/Simul/Test3/All/Test3_{i}.xlsx", scenar_used) for i in range(1,1001)]
    nocalib_files= [(f"Data/Simul/Test3/All/Nocalib_{i}.xlsx", scenar_used) for i in range(1,1001)]
    
    sharpes, totals = alloc.run_multiple_allocations(sim_files, nocalib_files)
    
    plt.figure(figsize=(12,6))
    sns.histplot(totals, bins=20, kde=True, color="skyblue")
    
    plt.axvline(budget, color="red", linestyle="--", linewidth=2, label=f"Max budget ({int(budget/1e+06)}Mt)")
    plt.axvline(np.mean(totals), color="green", linestyle="--", linewidth=2, label="Mean of realized budgets")
    
    plt.xlabel("Realized budget")
    plt.ylabel("Frequency")
    plt.title("Distribution of realized budgets across runs")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12,6))
    sns.histplot(sharpes, bins=20, kde=False, color="skyblue")
    
    plt.xlabel("Sharpe ratio")
    plt.ylabel("Frequency")
    plt.title("Distribution of time-averaged Sharpe ratios across runs")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return sharpes, totals

def return_tests():
    sim_files= [(f"Data/Simul/Test3/All/Test3_{i}.xlsx", scenar_used) for i in range(1,1001)]
    nocalib_files= [(f"Data/Simul/Test3/All/Nocalib_{i}.xlsx", scenar_used) for i in range(1,1001)]
    
    tb, tbn, sh, shn, tr, trn = alloc.compare_strat(sim_files, nocalib_files, update_budget = True)
    
    
    return tb, tbn, sh, shn, tr, trn

#sharpes, totals = distribution_tests()

#%%
num_test = 1
sim_path = f"Data/Simul/Test3/All/Test3_{num_test}.xlsx"
alloc.emissions_matrix(sim=True, path=sim_path, sheet_name=scenar_used)
alloc.cut_budget(start = 5e+07, end = 5e+06, linear=False, alpha = 0.3)
plt.figure()
plt.plot(alloc.parts, label ="Initial budgets", marker='o')

alloc.get_probas_simulation(f"Data/Simul/Test3/Calib_{num_test}.xlsx", sheet_name = scenar_used)
history = alloc.allocate_over_time()

carbs = [i["carbon"] for i in history]
real = [i["realized"] for i in history]

plt.plot(alloc.parts, label = "Updated budgets", marker='o')
plt.plot(carbs, label = "Expected", marker = "+")
plt.plot(real, label = 'Realized', marker = "*")
plt.title("Budget split")
plt.ylabel("Budget for year (tCO2eq)")
plt.xlabel("Year")
plt.grid(True)
plt.legend()

plt.show()

history_naive = alloc.allocate_naive_investor()
    
