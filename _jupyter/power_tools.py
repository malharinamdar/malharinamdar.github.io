import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
from multiprocess.pool import ThreadPool

import multiprocess as mp
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest as prop_test
from statsmodels.stats.weightstats import ttest_ind as ttest
import pandas as pd


class PowerSim:
    '''
    PowerSim class for simulation of power analysis.
    '''
    
    def __init__(self, metric='proportion', relative_effect=False, nsim=100, 
                 variants=None, comparisons=None, alternative='two-tailed', alpha=0.05, 
                 correction='bonferroni', fdr_method='indep'):
        """
        PowerSim class for simulation of power analysis.

        Parameters
        ----------
        metric : str
            Count, proportion, or average
        relative effect : bool
            True when change is percentual (not absolute).
        variants : int
            Number of cohorts or variants to use (remember, total number of groups = control + number of variants)
        comparisons : list
            List of tuple with the tests to run
        nsim : int
            Number of replicates to simulate power
        alternative : str
            Alternative hypothesis, 'two-tailed', 'greater', 'smaller'
        alpha : float
            One minus statistical confidence
        correction : str
            Type of correction: 'bonferroni', 'holm', 'fdr' or None
        fdr_method : 'indep' | 'negcorr'
            If 'indep' it implements Benjamini/Hochberg for independent or if
            'negcorr' it corresponds to Benjamini/Yekutieli.         
        """

        self.metric = metric
        self.relative_effect = relative_effect
        self.variants = variants
        self.comparisons = list(itertools.combinations(range(self.variants+1), 2)) if comparisons is None else comparisons
        self.nsim = nsim
        self.alternative = alternative
        self.alpha = alpha
        self.correction = correction
        self.fdr_method = fdr_method
    
    # simulate data
    def __run_experiment(self, baseline=[1.0], sample_size=[100], effect=[0.10], 
                         compliance=[1.0], standard_deviation=[1]):
        """
        Simulate data to run power analysis.

        Parameters
        ----------
        baseline : list
            List baseline rates for counts or proportions, or base average for mean comparisons
        sample_size : list
            List with sample for control and arm groups
        effect : list
            List with effect sizes
        standard_deviation : list
            List of standard deviations by groups
        compliance : list
            List with compliance values

        Returns
        -------
        Two vectors with simulated data
        """

        # initial checks
        if len(effect) != self.variants:
            if len(effect)>1:
                raise ValueError('Effects should be same length as the number of self.variants or length 1!')
            effect = list(itertools.repeat(effect[0], self.variants))

        if len(compliance) != self.variants:
            if len(compliance)>1:
                raise ValueError('Compliance rates should be same length as the number of self.variants or length 1!')
            compliance = list(itertools.repeat(compliance[0], self.variants))
        
        if len(standard_deviation) != self.variants+1:
            if len(standard_deviation)>1:
                raise ValueError('Standard deviations should be same length as the number of self.variants+1 or length 1!')
            standard_deviation = list(itertools.repeat(standard_deviation[0], self.variants+1))
        
        if len(sample_size) != self.variants+1:
            if len(sample_size)>1:
                raise ValueError('N should be same length as the number of self.variants+1 or length 1!')
            sample_size = list(itertools.repeat(sample_size[0], self.variants+1))

        if len(baseline) != self.variants+1:
            if len(baseline)>1:
                raise ValueError('Baseline values should be same length as the number of self.variants+1 or length 1!')
            baseline = list(itertools.repeat(baseline[0], self.variants+1))

        re = list(range(self.variants))

        # outputs
        dd = np.array([])
        
        # index of condition
        vv = np.array([])

        if self.metric == 'count':
            c_data = np.random.poisson(baseline[0], sample_size[0])
            dd = c_data
            vv = list(itertools.repeat(0, len(c_data)))

            for i in range(self.variants):
                if self.relative_effect:
                    re[i] = baseline[i+1] * (1.00 + effect[i])
                else:
                    re[i] = baseline[i+1] + effect[i]
                t_data_c = np.random.poisson(re[i], int(np.round(sample_size[i+1] * compliance[i])))
                t_data_nc = np.random.poisson(baseline[i+1], int(np.round(sample_size[i+1] * (1 - compliance[i]))))
                t_data = np.append(t_data_c, t_data_nc)
                dd = np.append(dd, t_data)
                vv = np.append(vv, list(itertools.repeat(i + 1, len(t_data))))

        if self.metric == 'proportion':
            c_data = np.random.binomial(n=1, size=int(sample_size[0]), p=baseline[0])
            dd = c_data
            vv = list(itertools.repeat(0, len(c_data)))

            for i in range(self.variants):
                if self.relative_effect:
                    re[i] = baseline[i+1] * (1.00 + effect[i])
                else:
                    re[i] = baseline[i+1] + effect[i]

                t_data_c = np.random.binomial(n=1, size=int(np.round(sample_size[i+1] * compliance[i])), p=re[i])
                t_data_nc = np.random.binomial(n=1, size=int(np.round(sample_size[i+1] * (1 - compliance[i]))), p=baseline[i+1])
                t_data = np.append(t_data_c, t_data_nc)
                dd = np.append(dd, t_data)
                vv = np.append(vv, list(itertools.repeat(i + 1, len(t_data))))

        if self.metric == 'average':
            c_data = np.random.normal(baseline[0], standard_deviation[0], sample_size[0])
            dd = c_data
            vv = list(itertools.repeat(0, len(c_data)))

            for i in range(self.variants):
                if self.relative_effect:
                    re[i] = baseline[i+1] * (1.00 + effect[i])
                else:
                    re[i] = baseline[i+1] + effect[i]

                t_data_c = np.random.normal(re[i], standard_deviation[i+1], int(np.round(sample_size[i+1] * compliance[i])))
                t_data_nc = np.random.normal(baseline[i+1], standard_deviation[i+1], int(np.round(sample_size[i+1] * (1 - compliance[i]))))

                t_data = np.append(t_data_c, t_data_nc)
                dd = np.append(dd, t_data)
                vv = np.append(vv, list(itertools.repeat(i + 1, len(t_data))))

        return dd, vv


    def get_power(self, baseline=[1.0], effect=[0.10], sample_size=[1000], compliance=[1.0], standard_deviation=[1]):
        '''
        Estimate power using simulation. 

        Parameters
        ----------
        baseline : list
            List baseline rates for counts or proportions, or base average for mean comparisons.
        effect : list
            List with effect sizes.
        sample_size : list
            List with sample for control and arm groups.
        compliance : list
            List with compliance values.
        standard_deviation : list
            List of standard deviations of control and variants.

        Returns
        -------
        power : float
        '''

        # create empty values for results
        results = []
        ncomparisons = len(self.comparisons)
        
        pvalues = {}
        for c in range(len(self.comparisons)):
            pvalues[c] = []

        # iterate over simulations
        for i in range(self.nsim):
            # y = output, x = index of condition
            y, x = self.__run_experiment(baseline=baseline, effect=effect, 
                                       sample_size=sample_size, compliance=compliance, 
                                       standard_deviation=standard_deviation)

            # iterate over variants
            l_pvalues = []
            for j, h in self.comparisons:

                # getting pvalues
                if self.metric == 'count':
                    
                    ty = np.append(y[np.isin(x, j)], y[np.isin(x, h)])
                    tx = np.append(x[np.isin(x, j)], x[np.isin(x, h)])
                    tx[np.isin(tx, j)] = 0
                    tx[np.isin(tx, h)] = 1
                    
                    model = sm.Poisson(ty, sm.add_constant(tx))
                    pm = model.fit(disp=False)
                    pvalue = pm.pvalues[1]
                    z = pm.params[1]

                elif self.metric == 'proportion':
                    z, pvalue = sm.stats.proportions_ztest(
                        [np.sum(y[np.isin(x, h)]), np.sum(y[np.isin(x, j)])], 
                        [len(y[np.isin(x, h)]), len(y[np.isin(x, j)])])
                    
                elif self.metric == 'average':
                    z, pvalue = stats.ttest_ind(y[np.isin(x, h)], y[np.isin(x, j)], equal_var=False)

                l_pvalues.append(pvalue)

            pvalue_adjustment = {
                'two-tailed': 1,
                'greater': 2,
                'smaller': 2
            }

            correction_methods = {
                'bonferroni': self.bonferroni,
                'holm': self.holm_bonferroni,
                'hochberg': self.hochberg,
                'sidak': self.sidak,
                'fdr': self.lsu
            }

            if self.correction in correction_methods:
                significant = correction_methods[self.correction](np.array(l_pvalues), self.alpha/pvalue_adjustment[self.alternative])

            for v, p in enumerate(significant):
                pvalues[v].append(p)
   
        # results.append(int(np.sum(pvalues)) >= len(self.comparisons))
        power = pd.DataFrame(pd.DataFrame(pvalues).mean()).reset_index()
        power.columns = ['comparisons', 'power']
        power['comparisons'] = power['comparisons'].map(dict(enumerate(self.comparisons)))
        
        return power


    def grid_sim_power(self, baseline_rates=None, effects=None, sample_sizes=None,
                    compliances=[[1]], standard_deviations=[[1]], threads=3, plot=False):
        """
        Return Pandas DataFrame with parameter combinations and statistical power

        Parameters
        ----------
        baseline_rates : list
            List of baseline rates for counts or proportions, or base average for mean comparisons.
        effects : list
            List with effect sizes.
        sample_sizes : list
            List with sample for control and variants.
        compliances : list
            List with compliance values.
        standard_deviations : list
            List of standard deviations of control and variants.
        threads : int
            Number of threads for parallelization.
        plot : bool
            Whether to plot the results.       
        """

        pdict = {'baseline': baseline_rates, 'effect': effects, 'sample_size': sample_sizes,
                'compliance': compliances, 'standard_deviation': standard_deviations}
        grid = self.__expand_grid(pdict)
        
        parameters = list(grid.itertuples(index=False, name=None))
        
        grid['nsim'] = self.nsim
        grid['alpha'] = self.alpha
        grid['alternative'] = self.alternative
        grid['metric'] = self.metric
        grid['variants'] = self.variants
        grid['comparisons'] = str(self.comparisons)
        grid['relative_effect'] = self.relative_effect
        grid = grid.loc[:, ['baseline', 'effect', 'sample_size', 'compliance', 'standard_deviation',
                            'variants', 'comparisons', 'nsim', 'alpha', 'alternative', 'metric', 'relative_effect']]
        pool = ThreadPool(processes=threads)
        results = pool.starmap(self.get_power, parameters)
        pool.close()
        pool.join()

        results = pd.concat(results)

        # create index 
        index = []
        repeating_number = grid.shape[0]    
        repeating_count = len(self.comparisons)
        for i in range(0, repeating_number):
            index.extend([i] * repeating_count)
    
        results['index'] = index
        results = results.pivot(index=['index'], columns=['comparisons'], values=['power'])
        results.columns = [str((i,j)) for i,j in self.comparisons]
        
        grid = pd.concat([grid, results], axis=1)
        grid.sample_size = grid.sample_size.map(str)
        grid.effect = grid.effect.map(str)
        if plot:
            self.plot_power(grid)
        return grid


    # plot power simulation
    def plot_power(self, data):
        '''
        Plot statistical power by scenario
        '''

        value_vars = [str((i,j)) for i,j in self.comparisons]

        cols = ['baseline', 'effect', 'sample_size', 'compliance', 'standard_deviation',
            'variants', 'comparisons', 'nsim', 'alpha', 'alternative', 'metric', 'relative_effect']
        
        temp = pd.melt(data, id_vars=cols, var_name='comparison', value_name='power', value_vars=value_vars)

        d_relative_effect = {True: 'relative', False: 'absolute'}
        effects = list(temp.effect.unique())
        for i in effects:
            plot = sns.lineplot(x='sample_size', y='power', hue='comparison', errorbar=None, data=temp[temp['effect'] == i], legend='full')
            plt.hlines(y=0.8, linestyles='dashed', xmin=0, xmax=len(temp.sample_size.unique()) - 1, colors='gray')
            plt.title(f'Simulated power estimation for {self.metric}s, {d_relative_effect[self.relative_effect]} effects {str(i)}\n (sims per scenario:{self.nsim})')
            plt.legend(bbox_to_anchor=(1.05, 1), title='comparison', loc='upper left')
            plt.xlabel('\n sample size')
            plt.ylabel('power\n')
            plt.setp(plot.get_xticklabels(), rotation=45)
            plt.show()


    # create grid dataframe
    def __expand_grid(self, dictionary):
        '''
        Auxiliary function to expand a dictionary
        '''
        return pd.DataFrame([row for row in itertools.product(*dictionary.values())], 
            columns=dictionary.keys())
    

    # pvalue adjustment functions 
    def bonferroni(self, pvals, alpha=0.05):
        """A function for controlling the FWER at some level alpha using the
        classical Bonferroni procedure.

        Parameters
        ----------
        pvals : array_like
            Set of p-values of the individual tests.
        alpha: float
            The desired family-wise error rate.

        Output: 
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """
        m, pvals = len(pvals), np.asarray(pvals)
        return pvals < alpha/float(m)


    def hochberg(self, pvals, alpha=0.05):
        """A function for controlling the FWER using Hochberg's procedure.

        Parameters
        ----------
        pvals : array_like
            Set of p-values of the individual tests.
        alpha: float
            The desired family-wise error rate.

        Output
        -------
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """
        m, pvals = len(pvals), np.asarray(pvals)
        # sort the p-values into ascending order
        ind = np.argsort(pvals)

        """Here we have k+1 (and not just k) since Python uses zero-based
        indexing."""
        test = [p <= alpha/(m+1-(k+1)) for k, p in enumerate(pvals[ind])]
        significant = np.zeros(np.shape(pvals), dtype='bool')
        significant[ind[0:np.sum(test)]] = True
        return significant


    def holm_bonferroni(self, pvals, alpha=0.05):
        """A function for controlling the FWER using the Holm-Bonferroni
        procedure.

        Parameters
        ----------
        pvals : array_like
            Set of p-values of the individual tests.
        alpha: float
            The desired family-wise error rate.
        
        Output
        -------
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """

        m, pvals = len(pvals), np.asarray(pvals)
        ind = np.argsort(pvals)
        test = [p > alpha/(m+1-k) for k, p in enumerate(pvals[ind])]

        """The minimal index k is m-np.sum(test)+1 and the hypotheses 1, ..., k-1
        are rejected. Hence m-np.sum(test) gives the correct number."""
        significant = np.zeros(np.shape(pvals), dtype='bool')
        significant[ind[0:m-np.sum(test)]] = True
        return significant


    def sidak(self, pvals, alpha=0.05):
        """A function for controlling the FWER at some level alpha using the
        procedure by Sidak.

        Parameters
        ----------
        pvals : array_like
            Set of p-values of the individual tests.
        alpha: float
            The desired family-wise error rate.

        Output
        ------
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """
        n, pvals = len(pvals), np.asarray(pvals)
        return pvals < 1. - (1.-alpha) ** (1./n)


    def lsu(self, pvals, q=0.05):
        """The (non-adaptive) one-stage linear step-up procedure (LSU) for
        controlling the false discovery rate, i.e. the classic FDR method
        proposed by Benjamini & Hochberg (1995).

        Parameters
        ----------
        pvals: array_like  
            Set of p-values of the individual tests.
        q: float
            The desired false discovery rate.

        Output:
        --------
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """

        m = len(pvals)
        sort_ind = np.argsort(pvals)
        k = [i for i, p in enumerate(pvals[sort_ind]) if p < (i+1.)*q/m]
        significant = np.zeros(m, dtype='bool')
        if k:
            significant[sort_ind[0:k[-1]+1]] = True
        return significant