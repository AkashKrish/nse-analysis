import warnings
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
from nsepy import get_history

import concurrent.futures
from helpers import rename_columns

# URLs for getting data
INDEX_URL = 'https://www.nseindia.com/content/indices/{0}.csv'
SYMBOL_LIST_URL = 'https://www.nseindia.com/content/equities/EQUITY_L.csv'
YAHOO_URL = 'http://real-chart.finance.yahoo.com/table.csv?'
TECHPAISA_URL = 'http://techpaisa.com/sector/'

# File path
SYMBOL_DATA_PATH = 'symbol_data.h5'
INDEX_DATA_PATH = 'index_data.h5'
TEMP_DATA_PATH = 'temp_data.h5'

# Schema for required dataframes
HIST_SYMBOL_SCHEMA = ['symbol', 'date', 'prev_close', 'open', 'high',
                      'low', 'last', 'close', 'vwap', 'volume',
                      'turnover', 'pct_deliverble', 'simple_returns',
                      'log_returns', 'daily_volatility']
DIVIDEND_SCHEMA = ['symbol', 'date', 'action', 'value']
TECH_STRENGTH_SCHEMA = ['date', 'symbol', 'tech_strength']
BETA_SCHEMA = ['symbol', 'benchmark', 'interval',
               'alpha', 'beta', 'std_dev', 'r_square', 'p_value', 'std_error']

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())
INDEX_META = pd.read_hdf('constants.h5', 'index_meta')
INDUSTRY_META = pd.read_hdf('constants.h5', 'industry_meta')
INDEX_LIST = pd.read_hdf('constants.h5', 'index_list')
RISK_FREE_RATE = np.log(1 + 0.075) / 250


class Market(object):

    def get_date(self, date=None, out='dt', start=True):
        if date is None:
            if start:
                try:
                    date = self.start
                except:
                    date = datetime(1996, 1, 1)
            else:
                try:
                    date = self.end
                except:
                    date = TODAY
        elif isinstance(date, str) and len(date) == 4:
            date = datetime(int(date), 1, 1)
        elif isinstance(date, str) and len(date) == 7:
            date = datetime(int(date[0:4]), int(date[-2:]), 1)
        elif isinstance(date, str) and len(date) == 10:
            date = datetime.strptime(date, '%Y-%m-%d')
        elif isinstance(date, int) and date > 1900:
            date = datetime(date, 1, 1)
        elif isinstance(date, datetime):
            pass
        else:
            raise ValueError('Invalid Value for date')
        if out == 'str':
            date = date.strftime('%Y-%m-%d')
        return date

    def impute_returns(self, data, limit=3):
        '''
        If returns is greater than 25 nullify it and
        interpolate to make returns data homogeneous
        '''
        data = data[data.first_valid_index():]
        outliers = abs(data) > 0.25
        data[outliers] = np.nan
        data = data.interpolate(method='time', limit=limit)
        data[outliers] = data[~outliers].mean()
        data = data.fillna(0)
        return data.round(4)

    def get_daily_volatility(self, returns_series):
        daily_volatility = pd.Series(0, index=returns_series.index,
                                     name=returns_series.name + '_dv')
        daily_volatility = np.sqrt(0.94 * np.square(daily_volatility) +
                                   0.06 * np.square(returns_series))
        return daily_volatility

    def handle_abnormal_returns(self, returns=None, symbol_data=None):
        if returns is not None:
            if isinstance(returns, pd.Series):
                returns = returns.copy()
                returns = returns[returns.first_valid_index():]
                abnormal_ret = abs(returns) > 0.25
                returns[abnormal_ret] = np.nan
                try:
                    returns = returns.interpolate(method='time', limit=2)
                except:
                    returns = returns.interpolate(limit=2)
                returns[abnormal_ret] = returns[~abnormal_ret].mean()
                returns = returns.fillna(0)

            elif isinstance(returns, pd.DataFrame):
                returns = returns.copy()
                returns = returns.apply(self.handle_abnormal_returns)
            return returns

        elif symbol_data is not None and isinstance(symbol_data, pd.DataFrame):
            symbol_data = symbol_data.copy()
            abnormal_ret = (abs(symbol_data.simple_returns) > 0.25) |\
                           (abs(symbol_data.log_returns) > 0.25) &\
                           (symbol_data.close > 1)

            symbol_data.ix[abnormal_ret, 'simple_returns'] = (
                (symbol_data.close.ix[abnormal_ret] - symbol_data.vwap.ix[abnormal_ret]) /
                symbol_data.open.ix[abnormal_ret]
            )
            symbol_data.ix[abnormal_ret, 'log_returns'] = np.log(
                symbol_data.close.ix[abnormal_ret] / symbol_data.vwap.ix[abnormal_ret]
            )
            return symbol_data
        else:
            raise ValueError(
                'Atleast one of symbol list of returns should be passed'
            )

    def fetch_index_list_symbols(self, req_index):
        '''
        Fetch list of symbols in req_index
        '''
        index = req_index.Index
        url = req_index.url
        r = requests.get(INDEX_URL.format(url))
        try:
            index_data = pd.read_csv(StringIO(r.text), index_col='Symbol')
            if index_data.empty:
                warnings.warn('No data recieved for {0} index'.format(index))
                return pd.Datafame()
            else:
                print('Data loaded for {0} index'.format(index))
        except Exception as e:
            warnings.warn(
                'Unable to get index meta for {0} index'.format(index)
            )
            return pd.Datafame()
        rename_columns(index_data)
        index_data[index] = True
        return index_data[index]

    def fetch_symbol_meta(self):
        '''
        Method to grab symbol meta data from NSE website
        '''
        r = requests.get(SYMBOL_LIST_URL)
        symbol_meta = pd.read_csv(StringIO(r.text), index_col='SYMBOL')

        rename_columns(symbol_meta)
        symbol_meta['from_date'] = pd.to_datetime('1994-01-01')
        symbol_meta['to_date'] = pd.to_datetime('1994-01-01')
        symbol_meta['row_count'] = np.nan
        symbol_meta['industry'] = 0
        symbol_meta['mcap'] = 0
        symbol_meta['p_e_ratio'] = 0
        symbol_meta = symbol_meta[(symbol_meta['series'] == 'EQ')]
        symbol_meta = symbol_meta[['name_of_company', 'date_of_listing',
                                   'from_date', 'to_date', 'row_count',
                                   'industry', 'mcap', 'p_e_ratio']]
        symbol_meta.date_of_listing = pd.to_datetime(symbol_meta.date_of_listing)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for index in INDEX_META.itertuples():
                p = executor.submit(self.fetch_index_list_symbols, index)
                res = p.result()
                if not res.empty:
                    symbol_meta = symbol_meta.join(res)
        symbol_meta.fillna(value=False, inplace=True)
        symbol_meta.index = symbol_meta.index.str.lower()
        symbol_meta = symbol_meta.sort_index()
        return symbol_meta

    def fetch_historical_data(self, symbol, start=None):
        '''Fetch all data from NSE and Adj Close from Yahoo Finance'''
        from_date = self.get_date(start, start=True)
        to_date = self.get_date(start=False)
        # Get data from NSE
        try:
            nse_data = get_history(symbol=symbol,
                                   start=from_date,
                                   end=to_date)
            if nse_data.empty:
                warnings.warn(
                    'No data recieved from NSE for {0} from {1} to {2}'.
                    format(symbol, from_date.date(), to_date.date())
                )
                return nse_data
            else:
                print(
                    'Recieved {0} records from NSE for {1} from {2} to {3}'.
                    format(len(nse_data), symbol,
                           nse_data.index.min(),
                           nse_data.index.max())
                )
            nse_data.drop(
                ['Series', 'Trades', 'Deliverable Volume'], 1, inplace=True
            )
            nse_data.index = pd.to_datetime(nse_data.index)
        except Exception as e:
            warnings.warn(
                'Could not get data for {0} from NSE due to {1}'.format(symbol, e)
            )
            return pd.DataFrame()

        rename_columns(nse_data)
        nse_data['symbol'] = [symbol for i in range(len(nse_data))]
        nse_data = nse_data.reset_index().sort_values(['symbol', 'date', 'close'])
        nse_data = nse_data.drop_duplicates(
            subset=['symbol', 'date'], keep='last'
        )

        nse_data['simple_returns'] = (
            (nse_data.close - nse_data.prev_close) / nse_data.prev_close
        )
        nse_data['log_returns'] = np.log(nse_data.close / nse_data.prev_close)
        nse_data = self.handle_abnormal_returns(symbol_data=nse_data)
        nse_data['daily_volatility'] = self.get_daily_volatility(nse_data.log_returns)

        # Adjusting other columns for maintaining integrity
        nse_data.volume = nse_data.volume.astype(np.float)
        nse_data['pct_deliverble'] = nse_data['pct_deliverble'] * 100
        return nse_data

    def fetch_dividend_data(self, symbol_list):
        '''Fetch dividend from Yahoo'''
        from_date = self.get_date(start=True)
        to_date = self.get_date(start=False)
        symbol = symbol_list.upper()

        try:
            from pandas_datareader.data import DataReader
            dividend_data = DataReader(
                symbol + '.NS', data_source='yahoo-actions',
                start=from_date, end=to_date
            )
            if dividend_data.empty:
                warnings.warn(
                    'No dividend data recieved for {0} from Yahoo'.format(symbol)
                )
                return dividend_data
            else:
                print(
                    'Loaded dividend data for {0} from {1} to {2}'.
                    format(symbol, from_date.date(), to_date.date())
                )
        except Exception as e:
            warnings.warn(
                'Warning: Could not get dividend data for {0} from Yahoo'.format(symbol)
            )
            return pd.DataFrame()
        rename_columns(dividend_data)
        dividend_data.index.name = 'date'
        dividend_data.value = round(dividend_data.value, 4)
        return dividend_data

    def fetch_tech_data_sector(self, sector_url):
        '''
        Fetch technical data for sector.
        '''
        try:
            tech_data = pd.read_html(sector_url)[0]
            if tech_data.empty:
                warnings.warn('Recieved empty data for {0}'.format(sector_url))
                return pd.DataFame()
            print('Loaded {0} sector list'.format(sector_url[28:-1]))
            return tech_data
        except:
            warnings.warn('Unable to fetch data for {0}'.format(sector_url))
            return pd.DataFrame()

    def fetch_tech_data(self, symbol_meta=None):
        '''Fetch technical data from techpaisa.com'''
        from bs4 import BeautifulSoup
        r = requests.get(TECHPAISA_URL)
        soup = BeautifulSoup(r.text, 'lxml')
        sector_link_list = []
        for links in soup.find_all('a'):
            link = links.get('href')
            if link is not None and link.startswith('/sector/'):
                sector_link_list.append(
                    TECHPAISA_URL + link[8:]
                )
        sector_link_list = list(set(sector_link_list))

        tech_data_daily = pd.DataFrame()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for sector_url in sector_link_list:
                p = executor.submit(self.fetch_tech_data_sector, sector_url)
                tech_data = p.result()
                if tech_data.empty:
                    continue
                tech_data['Sector'] = [
                    sector_url[28:-1].replace('-', '_') for i in range(0, len(tech_data))
                ]
                if len(tech_data_daily.columns) == 0:
                    tech_data_daily = tech_data
                else:
                    tech_data_daily = tech_data_daily.append(tech_data)

        tech_data_daily.rename(columns={
            tech_data_daily.columns[0]: 'symbol',
            tech_data_daily.columns[1]: 'close',
            tech_data_daily.columns[2]: 'simple_returns',
            tech_data_daily.columns[3]: 'futures_close',
            tech_data_daily.columns[4]: 'rsi',
            tech_data_daily.columns[5]: 'macd',
            tech_data_daily.columns[6]: 'sma',
            tech_data_daily.columns[7]: 'b_bands',
            tech_data_daily.columns[8]: 'fibo_retracement',
            tech_data_daily.columns[9]: 'corp_action',
            tech_data_daily.columns[10]: 'tech_strength',
            tech_data_daily.columns[11]: 'p_e_ratio',
            tech_data_daily.columns[12]: 'mcap',
            tech_data_daily.columns[13]: 'industry'
        }, inplace=True)

        tech_data_daily.symbol = tech_data_daily.symbol.str.strip().str.lower()
        # If symbol_meta is not passed just send tech_strength
        if symbol_meta is None:
            tech_data_daily['date'] = pd.to_datetime(TODAY.date())
            tech_data_daily = tech_data_daily[TECH_STRENGTH_SCHEMA]
            return tech_data_daily

        tech_data_daily = tech_data_daily[[
            'symbol', 'industry', 'mcap', 'p_e_ratio', 'tech_strength'
        ]]
        tech_data_daily = tech_data_daily.drop_duplicates('symbol', keep='last').set_index('symbol')

        industry_meta_dict = INDUSTRY_META.to_dict()
        symbol_meta.industry = tech_data_daily.industry.replace(industry_meta_dict)
        symbol_meta.industry = symbol_meta.industry.fillna('unknown')
        symbol_meta['mcap'] = tech_data_daily.mcap.replace(0, np.nan)
        symbol_meta['p_e_ratio'] = tech_data_daily.p_e_ratio
        symbol_meta.to_hdf(SYMBOL_DATA_PATH, 'symbol_meta')

    def fetch_index_data(self, index):
        '''
        Fetch historical price data for index.
        '''
        start = self.get_date(1995)
        index = index.upper()
        try:
            index_data = get_history(index, index=True,
                                     start=start, end=TODAY)
            if index_data.empty:
                warnings.warn('No data available for {0}'.format(index))
            else:
                print('Loaded data for {0}'.format(index))
        except:
            warnings.warn('unable to fetch data for {0}'.format(index))
            index_data = pd.DataFrame()
        return index_data

    def consecutive_nulls(self, returns):

        if isinstance(returns, pd.Series):
            symbol = returns.name
            returns.name = 'symbol'
            null_series = pd.DataFrame(returns)
            null_series['null_count'] = 0
            null_series = null_series[null_series['symbol'].first_valid_index():]
            for i in null_series.itertuples():
                if np.isnan(i[1]):
                    current_index = null_series.index.get_loc(i.Index)
                    try:
                        next_index = null_series.index.get_loc(null_series[i[0]:].symbol.first_valid_index())
                        counts = (next_index - current_index)
                        null_series.ix[current_index:next_index] = counts
                    except KeyError as e:
                        null_series.ix[current_index] = 1
            null_count = pd.Series(null_series.null_count, name=symbol)
            return null_count.fillna(0).astype(int)
        elif isinstance(returns, pd.DataFrame) and len(returns.columns) == 1:
            returns = returns.ix[:, 0]
            null_count = self.consecutive_nulls(returns)
            return null_count.fillna(0).astype(int)
        elif isinstance(returns, pd.DataFrame):
            null_count = returns.apply(self.consecutive_nulls).fillna(0).astype(int)
            return null_count
        else:
            raise ValueError(
                'Index returns must be a series or dataframe only'
            )

    def set_risk_free_rate(self, returns, risk_free_rate=RISK_FREE_RATE):
        index = returns.index
        if isinstance(index, pd.DatetimeIndex):
            pass
        else:
            raise ValueError('Invalid index of returns')
        if index.inferred_freq is None:
            if (returns.index[1] - returns.index[0]).days < 10:
                n = returns.resample('A').count().max(axis=1).max()
                n = int(np.maximum(n, 252))
                risk_free_rate = np.log(1 + risk_free_rate) / n
            else:
                pass
        elif str(index.inferred_freq)[0] == 'W':
            risk_free_rate = np.log(1 + risk_free_rate) / 54
        elif str(index.inferred_freq)[0] == 'M':
            risk_free_rate = np.log(1 + risk_free_rate) / 12
        elif str(index.inferred_freq)[0] == 'Q':
            risk_free_rate = np.log(1 + risk_free_rate) / 4
        return risk_free_rate
