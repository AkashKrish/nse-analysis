# import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats

import concurrent.futures
from symbol import Symbol
from index import Index

TEMP_DATA_PATH = 'temp_data.h5'

# Schema for required dataframes
BETA_SCHEMA = ['symbol', 'benchmark', 'interval',
               'alpha', 'beta', 'std_dev', 'r_square', 'p_value', 'std_error']

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())
RISK_FREE_RATE = np.log(1 + 0.075) / 250


class Portifolio(Symbol, Index):

    def __init__(self, symbol_list=None, start=None, end=None,
                 index=None, benchmark=None,
                 min_rows=None, volume=None, mcap=None,
                 force_load=False):

        self.symbol_meta = self.get_symbol_meta()
        self.symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index,
            start=start, min_rows=min_rows,
            volume=volume, mcap=mcap
        )
        self.index_list = self.get_index_list(
            index_list=benchmark
        )
        self.start = self.get_date(start, start=True)
        self.end = self.get_date(end, start=False)

    def calcluate_capm_interval(self, returns, benchmark_returns,
                                periods=[1, 3, 5, 10, 15], period_type='months'):

        common_index = np.intersect1d(returns.index, benchmark_returns.index)
        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        end = returns.index[-1]
        beta = pd.DataFrame(
            columns=['symbol', 'benchmark', 'interval',
                     'alpha', 'beta', 'std_dev', 'r_square']
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in periods:
                to_date = end
                if period_type == 'years':
                    from_date = end - relativedelta(years=i)
                elif period_type == 'months':
                    from_date = end - relativedelta(months=i)
                elif period_type == 'weeks':
                    from_date = end - relativedelta(days=7 * i)
                else:
                    return self.calculate_capm(
                        symbol_returns=returns, benchmark_returns=benchmark_returns,
                    )
                from_date = self.get_date(from_date, 'str')
                to_date = self.get_date(to_date, 'str')
                returns_temp = returns[from_date:to_date]
                benchmark_returns_temp = benchmark_returns[from_date:to_date]
                beta_temp = executor.submit(
                    self.calculate_capm, returns=returns_temp,
                    benchmark_returns=benchmark_returns_temp
                )
                capm_measures = beta_temp.result()
                capm_measures['interval'] = [
                    str(i).zfill(2) + period_type for n in range(0, len(capm_measures))
                ]
                beta = beta.append(capm_measures)
        beta = beta.sort_values(['symbol', 'benchmark', 'interval'])
        beta = beta[[
            'symbol', 'benchmark', 'interval',
            'alpha', 'beta', 'std_dev', 'r_square'
        ]]
        beta = beta.reset_index(drop=True)
        return beta

    def create_portifolio(self, symbol_list=None, index=None, returns=None,
                          initial_value=10000, weights=None, null_count=None,
                          start=None, end=None):
        if returns is None:
            symbol_list = self.get_symbol_list(
                symbol_list=symbol_list, index=index,
                start=start
            )
            returns = self.get_symbol_data(
                data='returns', symbol_list=symbol_list,
                start=start, end=end, null_count=null_count
            )

        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)

        returns = returns.fillna(returns.mean())
        portifolio = pd.DataFrame(0, index=returns.index.copy(),
                                  columns=returns.columns.copy())

        portifolio = [initial_value * weight for weight in weights] * np.exp(returns.cumsum())
        portifolio['value'] = portifolio.sum(axis=1)
        return portifolio

    def get_portifolio_returns(self, portifolio=None, start=None, end=None,
                               symbol_list=None, index=None,
                               returns=None, weights=None,
                               null_count=None):
        if portifolio is None:
            portifolio = self.create_portifolio(
                symbol_list=symbol_list, index=index, returns=returns,
                weights=weights, start=start, end=end, null_count=null_count
            )
        portifolio_returns = np.log(portifolio / portifolio.shift(1))
        portifolio_returns = portifolio_returns.ix[1:]
        return portifolio_returns

    def beta(self, returns, benchmark_returns):
        # Create a matrix of [returns, market]
        returns = pd.DataFrame(returns)
        if isinstance(benchmark_returns, pd.Series):
            names = [benchmark_returns.name]
        else:
            names = benchmark_returns.columns
        returns_matrix = returns.join(benchmark_returns)
        co_var = returns_matrix.cov()
        co_var = co_var.loc[:, names].drop(names)
        # Return the covariance of m divided by the standard deviation of the market returns
        return co_var / benchmark_returns.var()

    def alpha(self, returns, benchmark_returns):
        benchmark_returns = pd.DataFrame(benchmark_returns)
        beta = self.beta(returns, benchmark_returns)
        rf_benchmark, _ = self.set_risk_free_rate(benchmark_returns)
        market_mean = rf_benchmark.mean() + ((benchmark_returns - rf_benchmark).mean() * beta)
        for benchmark in market_mean.columns:
            market_mean[benchmark] = returns.mean() - market_mean[benchmark]
        return market_mean

    def basic_regression(self, returns, benchmark_returns):
        returns = pd.DataFrame(returns.dropna())
        benchmark_returns = pd.DataFrame(benchmark_returns.dropna())
        common_index = np.intersect1d(returns.index, benchmark_returns.index)
        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        rf_returns, _ = self.set_risk_free_rate(returns)
        rf_benchmark, _ = self.set_risk_free_rate(benchmark_returns)
        returns = returns - rf_returns
        benchmark_returns = benchmark_returns - rf_benchmark
        slope, intercept, r_value, _, _ = stats.linregress(
            y=returns.ix[:, 0], x=benchmark_returns.ix[:, 0]
        )
        std_dev = returns.std().values[0]
        regression_measures = pd.DataFrame(
            [[returns.columns[0], benchmark_returns.columns[0], intercept, slope, std_dev, r_value**2]],
            columns=['symbol', 'benchmark', 'alpha', 'beta', 'std_dev', 'r_square']
        )
        return regression_measures

    def calculate_capm(self, returns, benchmark_returns):
        returns = pd.DataFrame(returns)
        benchmark_returns = pd.DataFrame(benchmark_returns)
        regr_measures = pd.DataFrame(
            columns=['symbol', 'benchmark', 'alpha', 'beta', 'std_dev', 'r_square']
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for benchmark in benchmark_returns.columns:
                for symbol in returns.columns:
                    beta_executor = executor.submit(
                        self.basic_regression, returns=returns[symbol],
                        benchmark_returns=benchmark_returns[benchmark]
                    )
                    capm = beta_executor.result()
                    regr_measures = regr_measures.append(capm)
                    regr_measures = regr_measures.reset_index(drop=True)
        return regr_measures

    def calculate_capm_frequency(self, returns, benchmark_returns,
                                 frequency='M'):
        returns = pd.DataFrame(returns)
        returns = returns.resample(frequency).sum()
        benchmark_returns = benchmark_returns.resample(frequency).sum()
        interval_beta = self.calculate_capm(returns, benchmark_returns)
        return interval_beta

    def lower_partial_moment(self, returns, threshold=0, order=2):
        # Calculate the difference between the threshold and the returns
        returns = pd.DataFrame(returns)
        diff = threshold - returns
        # Set the minimum of each to 0
        diff = diff.clip(lower=0)
        diff = np.power(diff.pow(order).mean(), 1 / order)

        # Return the mean of the different to the power of order
        return pd.DataFrame(diff, columns=['lpm'])

    def higher_partial_moments(self, returns, threshold=0, order=2):
        # Calculate the difference between the threshold and the returns
        returns = pd.DataFrame(returns)
        diff = returns - threshold
        # Set the minimum of each to 0
        diff = diff.clip(lower=0)
        diff = np.power(diff.pow(order).mean(), 1 / order)

        # Return the mean of the different to the power of order
        return pd.DataFrame(diff, columns=['hpm'])

    def prices(self, returns, base=1000):
        returns = returns.loc[returns.first_valid_index():].fillna(0)
        prices = pd.Series(index=returns.index.copy())
        prices = base * np.exp((returns.cumsum()))
        return prices

    def draw_down(self, returns, tau=5):
        # Returns the draw-down given time period tau
        values = self.prices(returns)
        pos = len(values) - 1
        pre = pos - tau
        drawdown = float('+inf')
        # Find the maximum drawdown given tau
        while pre >= 0:
            dd_i = (values[pos] / values[pre]) - 1
            if dd_i < drawdown:
                drawdown = dd_i
            pos, pre = pos - 1, pre - 1
        # Drawdown should be positive
        return abs(drawdown)

    def max_draw_down(self, returns):
        # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
        max_drawdown = float('-inf')
        for i in range(0, len(returns)):
            drawdown_i = self.draw_down(returns, i)
            if drawdown_i > max_drawdown:
                max_drawdown = drawdown_i
        # Max draw-down should be positive
        return abs(max_drawdown)

    def average_draw_down(self, returns, periods=2):
        # Returns the average maximum drawdown over n periods
        drawdowns = []
        for i in range(0, len(returns)):
            drawdown_i = self.draw_down(returns, i)
            drawdowns.append(drawdown_i)
        drawdowns = sorted(drawdowns)
        total_dd = abs(drawdowns[0])
        for i in range(1, periods):
            total_dd += abs(drawdowns[i])
        return total_dd / periods

    def average_dd_squared(self, returns, periods=2):
        # Returns the average maximum drawdown squared over n periods
        drawdowns = []
        for i in range(0, len(returns)):
            drawdown_i = np.power(self.draw_down(returns, i), 2.0)
            drawdowns.append(drawdown_i)
        drawdowns = sorted(drawdowns)
        total_dd = abs(drawdowns[0])
        for i in range(1, periods):
            total_dd += abs(drawdowns[i])
        return total_dd / periods

    def treynor_ratio(self, returns, benchmark_returns):
        returns = pd.DataFrame(returns)
        benchmark_returns = pd.DataFrame(benchmark_returns)
        risk_free_rate, interval = self.set_risk_free_rate(returns)
        excess_returns = returns - risk_free_rate
        treynor_ratio = pd.DataFrame(
            index=returns.columns, columns=benchmark_returns.columns
        )
        for benchmark in benchmark_returns.columns:
            treynor_ratio[benchmark] = (
                np.sqrt(interval) * np.sqrt(np.floor_divide(len(returns), interval) + 1) *
                excess_returns.mean() /
                self.beta(returns, benchmark_returns)[benchmark]
            )
        return treynor_ratio

    def sharpe_ratio(self, returns):
        returns = pd.DataFrame(returns)
        risk_free_rate, interval = self.set_risk_free_rate(returns)
        excess_returns = returns - risk_free_rate
        sharpe_ratio = pd.DataFrame(
            np.sqrt(interval) * np.sqrt(np.floor_divide(len(returns), interval) + 1) *
            (excess_returns.mean() / excess_returns.std()).rename('sharpe_ratio')
        )
        return sharpe_ratio

    def information_ratio(self, returns, benchmark_returns):
        returns = pd.DataFrame(returns)
        benchmark_returns = pd.DataFrame(benchmark_returns)
        information_ratio = pd.DataFrame(index=returns.columns, columns=benchmark_returns.columns)
        for benchmark in benchmark_returns.columns:
            for symbol in returns.columns:
                excess_returns = returns[symbol] - benchmark_returns[benchmark]
                information_ratio.loc[symbol, benchmark] = (
                    excess_returns.mean() / excess_returns.std()
                )
        return information_ratio

    def modigliani_ratio(self, returns, benchmark_returns):
        returns = pd.DataFrame(returns)
        benchmark_returns = pd.DataFrame(benchmark_returns)
        rf_returns, interval = self.set_risk_free_rate(returns)
        rf_benchmark, interval = self.set_risk_free_rate(benchmark_returns)

        excess_returns = returns - rf_returns
        excess_benchmark_returns = benchmark_returns - rf_benchmark
        modigliani_ratio = pd.DataFrame(index=returns.columns, columns=benchmark_returns.columns)
        for benchmark in benchmark_returns.columns:
            for symbol in returns.columns:
                modigliani_ratio.loc[symbol, benchmark] = (
                    np.sqrt(interval) * np.sqrt(np.floor_divide(len(returns), interval) + 1) *
                    excess_returns.mean()[symbol] *
                    ((excess_returns).std()[symbol] / (excess_benchmark_returns).std()[benchmark]) +
                    np.exp(rf_returns.mean()[symbol])**interval
                )
        return modigliani_ratio

    def omega_ratio(self, returns, target=0):
        returns = pd.DataFrame(returns)
        risk_free_rate, interval = self.set_risk_free_rate(returns)
        excess_returns = (returns - risk_free_rate).mean()
        lpm = self.lower_partial_moment(returns, target, 1)['lpm']
        omega_ratio = (
            np.sqrt(interval) * np.sqrt(np.floor_divide(len(returns), interval) + 1) *
            excess_returns / lpm
        )
        return pd.DataFrame(omega_ratio, columns=['omega_ratio'])

    def sortino_ratio(self, returns, target=0):
        returns = pd.DataFrame(returns)
        risk_free_rate, interval = self.set_risk_free_rate(returns)
        excess_returns = (returns - risk_free_rate).mean()
        lpm = self.lower_partial_moment(returns, target, 2)['lpm']
        sortino_ratio = (
            np.sqrt(interval) * np.sqrt(np.floor_divide(len(returns), interval) + 1) *
            excess_returns / lpm
        )
        return pd.DataFrame(sortino_ratio, columns=['sortino_ratio'])

    def kappa_three_ratio(self, returns, target=0):
        returns = pd.DataFrame(returns)
        risk_free_rate, interval = self.set_risk_free_rate(returns)
        excess_returns = (returns - risk_free_rate).mean()
        lpm = self.lower_partial_moment(returns, target, 3)['lpm']
        kappa_three_ratio = (
            np.sqrt(interval) * np.sqrt(np.floor_divide(len(returns), interval) + 1) *
            excess_returns / lpm
        )
        return pd.DataFrame(kappa_three_ratio, columns=['kappa_three_ratio'])

    def gain_loss_ratio(self, returns, target=0):
        returns = pd.DataFrame(returns)
        lpm = self.lower_partial_moment(returns, target, 1)['lpm']
        hpm = self.higher_partial_moments(returns, target, 1)['hpm']
        gain_loss_ratio = hpm / lpm
        return pd.DataFrame(gain_loss_ratio, columns=['gain_loss_ratio'])

    def upside_potential_ratio(self, returns, target=0):
        returns = pd.DataFrame(returns)
        lpm = self.lower_partial_moment(returns, target, 2)['lpm']
        hpm = self.higher_partial_moments(returns, target, 2)['hpm']
        upside_potential_ratio = hpm / lpm
        return pd.DataFrame(upside_potential_ratio, columns=['upside_potential_ratio'])

    def calmar_ratio(self, returns):
        returns = pd.DataFrame(returns)
        risk_free_rate, interval = self.set_risk_free_rate(returns)
        excess_returns = (returns - risk_free_rate).mean()
        calmar_ratio = (
            np.sqrt(interval) * np.sqrt(np.floor_divide(len(returns), interval) + 1) *
            excess_returns / self.max_draw_down(returns)
        )
        return pd.DataFrame(calmar_ratio, columns='calmar_ratio')

    def sterling_ratio(self, returns, risk_free_rate=RISK_FREE_RATE, periods=2):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / self.average_draw_down(returns, periods)

    def burke_ratio(self, returns, risk_free_rate=RISK_FREE_RATE, periods=2):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / np.sqrt(self.average_draw_down(returns, periods))

    def describe_returns(self, returns, benchmark_returns):
        returns_describe = returns.describe().T
        returns_describe['count'] = returns_describe['count'].astype(int)
        returns_describe = returns_describe.rename(
            columns={
                'count': 'num_returns',
                'mean': 'mean_returns',
                'std': 'std_dev',
                'min': 'min_returns',
                'max': 'max_returns',
                '25%': '25_pctile',
                '50%': '50_pctile',
                '75%': '75_pctile'
            })
        percentiles = returns_describe.ix[:, 4:7].copy()
        returns_describe = returns_describe.drop(percentiles.columns, axis=1)
        returns_describe = returns_describe.join(returns.sum().rename('total_returns'))
        returns_describe = returns_describe.join(returns.median().rename('median_returns'))

        pos_pctile = pd.Series(name='pos_pctile')
        for symbol in returns.columns:
            symbol_ret = returns[symbol]
            n = symbol_ret.count()
            pos_pctile[symbol] = symbol_ret[symbol_ret > 0].count() / n
        returns_describe = returns_describe.join(pos_pctile)

        symbol_sharpe_ratio = returns.apply(self.sharpe_ratio)
        returns_describe = returns_describe.join(symbol_sharpe_ratio.rename('sharpe_ratio'))
        returns_describe = returns_describe.join(percentiles)
        return returns_describe.round(4)
