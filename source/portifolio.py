import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats

import concurrent.futures
from symbol import Index, Symbol

TEMP_DATA_PATH = 'temp_data.h5'

# Schema for required dataframes
BETA_SCHEMA = ['symbol', 'benchmark', 'interval',
               'alpha', 'beta', 'std_dev', 'r_square', 'p_value', 'std_error']

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())
RISK_FREE_RATE = np.log(1 + 0.075) / 250


class Portifolio(Symbol, Index):

    def __init__(self, symbol_list=None, start=None, end=None,
                 index=None, benchmark=None, null_count=None,
                 min_rows=None, volume=None, mcap=None,
                 force_load=False):

        self.symbol_meta = self.get_symbol_meta()
        self.symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index,
            null_count=null_count, start=start, min_rows=min_rows,
            volume=volume, mcap=mcap
        )
        self.index_list = self.get_index_list(
            index_list=benchmark
        )
        self.start = self.get_date(start, start=True)
        self.end = self.get_date(end, start=False)

    def capm_symbol_to_index(self, symbol_list=None,
                             start=None, end=TODAY,
                             benchmark=None, index=None,
                             null_count=None, risk_free_rate=RISK_FREE_RATE,
                             periods=[1, 3, 5, 10, 15], period_type='years'):
        start = self.get_date(start, start=True)
        end = self.get_date(end, start=False)

        symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index,
            null_count=null_count, start=start
        )
        benchmark = self.get_index_list(
            index_list=benchmark, null_count=null_count, start=start
        )
        sret = self.get_symbol_returns(symbol_list=symbol_list,
                                       start=start, end=end,
                                       null_count=null_count)
        sret = self.handle_abnormal_returns(sret)
        iret = self.get_index_returns(index_list=benchmark,
                                      start=start, end=end,
                                      null_count=null_count)
        beta = pd.DataFrame(columns=BETA_SCHEMA)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in periods:
                to_date = end
                if period_type == 'years':
                    from_date = end - relativedelta(years=i)
                elif period_type == 'months':
                    from_date = end - relativedelta(months=i)
                elif period_type == 'days':
                    from_date = end - relativedelta(days=i)
                else:
                    return self.calculate_capm(
                        symbol_returns=sret, index_returns=iret,
                        risk_free_rate=risk_free_rate
                    )
                from_date = self.get_date(from_date, 'str')
                to_date = self.get_date(to_date, 'str')
                sret_temp = sret[from_date:to_date]
                p = executor.submit(
                    self.calculate_capm, returns=sret_temp,
                    benchmark_returns=iret, risk_free_rate=risk_free_rate
                )
                capm_variables = p.result()
                capm_variables['interval'] = [str(i).zfill(2) + period_type for n in range(0, len(capm_variables))]
                beta = beta.append(capm_variables)
        beta = beta.sort_values(['symbol', 'benchmark', 'interval'])
        beta = beta[BETA_SCHEMA]
        beta = beta.reset_index(drop=True)
        return beta

    def create_portifolio(self, start=None, symbol_list=None, index=None, ret=None,
                          initial_value=10000, weights=None, null_count=None,
                          hist_data=None):
        if ret is None:
            symbol_list = self.get_symbol_list(
                symbol_list=symbol_list, index=index,
                null_count=null_count, start=start
            )
            ret = self.get_symbol_returns(symbol_list=symbol_list, start=start)
            symbol_list = symbol_list[symbol_list.index.intersection(ret.columns)]
            column_list = self.get_symbol_list(symbol_list=symbol_list, ret=ret,
                                               hist_data=hist_data, start=start)
            ret = ret.ix[start:][column_list.index]

        if weights is None:
            weights = np.ones(len(ret.columns)) / len(ret.columns)

        portifolio = pd.DataFrame(0, index=ret.index.copy(),
                                  columns=ret.columns.copy())

        portifolio.ix[0, :] = [initial_value * weight for weight in weights]
        portifolio = portifolio.ix[0] * np.exp(ret.cumsum())
        portifolio['value'] = portifolio.sum(axis=1)
        return portifolio

    def beta(self, returns, benchmark_returns):
        # Create a matrix of [returns, market]
        returns_matrix = pd.DataFrame(returns.rename('symbol')).join(benchmark_returns.rename('bench'))
        co_var = returns_matrix.cov()
        # Return the covariance of m divided by the standard deviation of the market returns
        return co_var.ix[0, 1] / benchmark_returns.var()

    def calculate_capm(self, returns, benchmark_returns,
                       risk_free_rate=RISK_FREE_RATE):
        capm_schema = BETA_SCHEMA.copy()
        capm_schema.remove('interval')
        if isinstance(benchmark_returns, pd.Series):
            benchmark = benchmark_returns.name
        elif isinstance(benchmark_returns, pd.DataFrame):
            capm_variables = pd.DataFrame(columns=capm_schema)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for benchmark in benchmark_returns.columns:
                    p = executor.submit(
                        self.calculate_capm, returns=returns,
                        benchmark_returns=benchmark_returns[benchmark],
                        risk_free_rate=risk_free_rate
                    )
                    capm = p.result()
                    capm_variables = capm_variables.append(capm)
                capm_variables = capm_variables.reset_index(drop=True)
                capm_variables = capm_variables.sort_values(['symbol'])
                return capm_variables
        else:
            raise ValueError(
                'Index returns must be a series or dataframe only'
            )
        if isinstance(returns, pd.Series):
            symbol = returns.name
            returns = pd.DataFrame(returns)
            returns = returns.join(benchmark_returns)
            returns = returns.dropna(subset=[benchmark])
            risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
            returns = returns - risk_free_rate
            slope, intercept, r_value, p_value, std_err = stats.linregress(y=returns[symbol], x=returns[benchmark])
            std_dev = returns[symbol].std()
            capm_variables = pd.DataFrame(
                [[symbol, benchmark, intercept, slope, std_dev, r_value**2, p_value, std_err]],
                columns=capm_schema
            )
            return capm_variables
        elif isinstance(returns, pd.DataFrame):
            capm_variables = pd.DataFrame(
                columns=capm_schema
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for symbol in returns.columns:
                    p = executor.submit(
                        self.calculate_capm, returns=returns[symbol],
                        benchmark_returns=benchmark_returns, risk_free_rate=risk_free_rate
                    )
                    capm = p.result()
                    capm_variables = capm_variables.append(capm)
                capm_variables = capm_variables.sort_values(['symbol'])
                capm_variables = capm_variables.reset_index(drop=True)
                return capm_variables
        else:
            raise ValueError(
                'Index returns must be a series or dataframe only'
            )

    def calculate_capm_interval(self, returns, benchmark_returns,
                                risk_free_rate=0.075, frequency='A'):
        if isinstance(returns, pd.Series):
            returns = pd.DataFrame(returns)
        if frequency == 'A':
            years = returns.index.year
            interval_beta = returns.groupby(years).apply(
                self.calculate_capm, benchmark_returns=benchmark_returns,
                risk_free_rate=risk_free_rate
            )
            interval_beta.index = interval_beta.index.droplevel(1)
            interval_beta.index.rename('year', inplace=True)
            interval_beta = interval_beta.reset_index()
        elif frequency == 'M':
            years = returns.index.year
            months = returns.index.month
            interval_beta = returns.groupby([years, months]).apply(
                self.calculate_capm, benchmark_returns=benchmark_returns,
                risk_free_rate=risk_free_rate
            )
            interval_beta.index = interval_beta.index.droplevel(2)
            interval_beta.index.rename(['year', 'month'], inplace=True)
            interval_beta = interval_beta.reset_index()
            interval_beta = interval_beta.sort_values(['symbol', 'year', 'month'])
            interval_beta['day'] = 1
            interval_beta.index = pd.to_datetime(interval_beta[['year', 'month', 'day']])
            interval_beta = interval_beta.drop(['year', 'month', 'day'], axis=1)
        return interval_beta

    def lower_partial_moment(self, returns, threshold=0, order=1):
        # Calculate the difference between the threshold and the returns
        diff = threshold - returns
        # Set the minimum of each to 0
        diff = diff.clip(lower=0)

        # Return the mean of the different to the power of order
        return diff.pow(order).mean()

    def higher_partial_moments(self, returns, threshold=0, order=1):
        # Calculate the difference between the threshold and the returns
        diff = returns - threshold
        # Set the minimum of each to 0
        diff = diff.clip(lower=0)

        # Return the mean of the different to the power of order
        return diff.pow(order).mean()

    def prices(self, returns, base=1000):
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

    def treynor_ratio(self, returns, benchmark_returns, risk_free_rate=RISK_FREE_RATE):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / self.beta(returns, benchmark_returns)

    def sharpe_ratio(self, returns, risk_free_rate=RISK_FREE_RATE):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def information_ratio(self, returns, benchmark):
        excess_returns = returns - benchmark
        return excess_returns.mean() / excess_returns.std()

    def modigliani_ratio(self, returns, benchmark_returns, risk_free_rate=RISK_FREE_RATE):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        excess_benchmark_returns = benchmark_returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() *\
            ((excess_returns).std() / (excess_benchmark_returns).std()) + np.exp(risk_free_rate)**252

    def omega_ratio(self, returns, risk_free_rate=RISK_FREE_RATE, target=0):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / self.lower_partial_moment(returns, target, 1)

    def sortino_ratio(self, returns, risk_free_rate=RISK_FREE_RATE, target=0):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / np.sqrt(self.lower_partial_moment(returns, target, 2))

    def kappa_three_ratio(self, returns, risk_free_rate=RISK_FREE_RATE, target=0):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / np.power(self.lower_partial_moment(returns, target, 3), 1 / 3)

    def gain_loss_ratio(self, returns, target=0):
        return self.higher_partial_moments(returns, target, 1) / self.lower_partial_moment(returns, target, 1)

    def upside_potential_ratio(self, returns, target=0):
        return self.higher_partial_moments(returns, target, 1) / np.sqrt(self.lower_partial_moment(returns, target, 2))

    def calmar_ratio(self, returns, risk_free_rate=RISK_FREE_RATE):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / self.max_draw_down(returns)

    def sterling_ratio(self, returns, risk_free_rate=RISK_FREE_RATE, periods=2):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / self.average_draw_down(returns, periods)

    def burke_ratio(self, returns, risk_free_rate=RISK_FREE_RATE, periods=2):
        risk_free_rate = self.set_risk_free_rate(returns, risk_free_rate)
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / np.sqrt(self.average_draw_down(returns, periods))
