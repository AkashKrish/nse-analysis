'''Module for loading the symbol metadata'''  # noqa
import os
import warnings
from datetime import datetime
from market import Market

import pandas as pd
from dateutil.relativedelta import relativedelta
from helpers import clean_file, SafeHDFStore
import concurrent.futures


# File path
SYMBOL_DATA_PATH = 'symbol_data.h5'
TEMP_DATA_PATH = 'temp_data.h5'

# Schema for required dataframes

DIVIDEND_SCHEMA = ['symbol', 'date', 'action', 'value']
TECH_STRENGTH_SCHEMA = ['date', 'symbol', 'tech_strength']

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())


class Symbol(Market):

    def index_to_symbol_list(self, index, type='union'):
        symbol_meta = self.get_symbol_meta()

        if isinstance(index, list):
            if type == 'union':
                slist_temp = pd.DataFrame()
                for ind in index:
                    if ind not in symbol_meta.columns:
                        warnings.warn('Index not available in symbol meta')
                        raise KeyError
                    symbol_list = symbol_meta[symbol_meta[ind]]
                    slist_temp = pd.DataFrame(index=slist_temp.index.union(symbol_list.index))
                symbol_list = symbol_meta.ix[slist_temp.index, 'from_date']
            elif type == 'intersection':
                slist_temp = pd.DataFrame(index=symbol_meta.index)
                for ind in index:
                    if ind not in symbol_meta.columns:
                        warnings.warn('Index not available in symbol meta')
                        raise KeyError
                    symbol_list = symbol_meta[symbol_meta[ind]]
                    slist_temp = pd.DataFrame(index=slist_temp.index.intersection(symbol_list.index))
                symbol_list = symbol_meta.ix[slist_temp.index, 'from_date']
            else:
                raise ValueError('Invalid type specified')
        elif isinstance(index, 'str'):
            try:
                symbol_list = symbol_meta.from_date[symbol_meta[index]]
            except:
                warnings.warn(
                    'Invalid index passed as string, loading nifty_50 symbol list'
                )
                symbol_list = symbol_meta.from_date[symbol_meta['nifty_50']]
        else:
            warnings.warn(
                'Invalid value passed as index, loading nifty_50 symbol list'
            )
            symbol_list = symbol_meta.from_date[symbol_meta['nifty_50']]
        return symbol_list

    def get_symbol_list(self, symbol_list=None, index=None,
                        start=None, min_rows=None,
                        volume=None, mcap=None):
        '''
        Get symbol list based on criteria provided.
        Pass index for getting symbols in index.
        List of indexes to get union of symbols of all indexes in list.
        start: {year as int or string, string format of date, None}
        hist_data: hist_data to be used for filtering symbol list
        null_count: {True, False} load symbols listed before start date
        '''
        symbol_meta = self.get_symbol_meta()

        if symbol_list is None:
            try:
                symbol_list = self.symbol_list.copy()
            except:
                symbol_list = symbol_meta.from_date.copy()
        elif isinstance(symbol_list, str):
            symbol_list = pd.Series([symbol_list])
            symbol_list = symbol_meta.from_date[symbol_meta.index.isin(symbol_list)].copy()
        elif isinstance(symbol_list, list):
            symbol_list = pd.Series(symbol_list)
            symbol_list = symbol_meta.from_date[symbol_meta.index.isin(symbol_list)].copy()
        elif isinstance(symbol_list, pd.Series):
            symbol_list = symbol_list.copy()
        elif isinstance(symbol_list, pd.DataFrame):
            try:
                symbol_list = symbol_list.date.copy()
            except:
                pass
            try:
                symbol_list = symbol_list.from_date.copy()
            except:
                warnings.warn('Could not fetch symbol_list with proper dates.\
                               Loading default symbol_list')
                symbol_list = symbol_meta.from_date.copy()
        else:
            warnings.warn('Could not fetch symbol_list with proper dates.\
                           Loading default symbol_list')
            symbol_list = symbol_meta.from_date.copy()

        if start is not None:
            start = self.get_date(start, 'dt')
            symbol_list = symbol_list[symbol_list <= start]

        if index is not None:
            index = index.lower().replace(' ', '_')
            symbol_list = self.index_to_symbol_list(index)

        if min_rows is not None:
            temp_smeta = symbol_meta[symbol_meta.index.isin(symbol_list.index)]
            symbol_list = symbol_list[temp_smeta.row_count >= min_rows]

        if volume is not None:
            hist_data = self.get_symbol_hist(symbol_list=symbol_list, start=start)
            symbol_list = symbol_list[
                hist_data.groupby('symbol').aggregate({'volume': 'min'}).volume >= volume
            ]
        if mcap is not None:
            temp_smeta = symbol_meta[symbol_meta.index.isin(symbol_list.index)]
            symbol_list = symbol_list[temp_smeta.mcap >= mcap]
        return symbol_list

    def get_symbol_meta(self):
        '''
        If symbol meta data exists grab data from file.
        Else fetch symbol meta data from NSE website.
        '''
        try:
            symbol_meta = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_meta')
        except:
            self.force_load_data(force_load='symbol_meta')
            symbol_meta = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_meta')
        return symbol_meta

    def update_symbol_meta_dates(self):
        # Update from_date, to_date, row_count fields
        symbol_meta = self.get_symbol_meta()
        try:
            hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')
        except:
            symbol_meta['from_date'] = pd.to_datetime('1994-01-01')
            symbol_meta['to_date'] = pd.to_datetime('1994-01-01')
            symbol_meta['row_count'] = 0
            symbol_meta.to_hdf(SYMBOL_DATA_PATH, 'symbol_meta')
            clean_file(SYMBOL_DATA_PATH)
            return

        def counts(data):
            count_data = pd.DataFrame(index=data.symbol.unique())
            count_data['from_date'] = data.date.min()
            count_data['to_date'] = data.date.max()
            count_data['row_count'] = len(data)
            return count_data
        count_data = hist_data.groupby('symbol').apply(counts)
        count_data = count_data.reset_index(1, drop=True)
        symbol_meta['from_date'] = count_data['from_date']
        symbol_meta['to_date'] = count_data['to_date']
        symbol_meta['row_count'] = count_data['row_count']
        symbol_meta['from_date'] = symbol_meta['from_date'].fillna(datetime(1994, 1, 1))
        symbol_meta['to_date'] = symbol_meta['to_date'].fillna(datetime(1994, 1, 1))
        symbol_meta['row_count'] = symbol_meta['row_count'].fillna(0).astype(int)
        symbol_meta.to_hdf(SYMBOL_DATA_PATH, 'symbol_meta')
        clean_file(SYMBOL_DATA_PATH)

    def get_symbol_hist(self, symbol_list=None, index=None,
                        start=None, end=None,
                        volume=None, mcap=None):
        '''
        If SYMBOL_DATA_PATH exists grab data from file.
        Update data if data in the file is older than 5 days.
        Else fetch symbol data from NSE website.
        '''

        try:
            hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')
        except Exception as e:
            self.force_load_data(force_load='symbol_hist')
            hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')

        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           index=index, start=start,
                                           volume=volume, mcap=mcap)

        hist_data = hist_data[hist_data.symbol.isin(symbol_list.index)]
        start = self.get_date(start, out='dt', start=True)
        end = self.get_date(end, out='dt', start=False)
        hist_data = hist_data.ix[
            (hist_data.date >= start) & (hist_data.date <= end)
        ]
        return hist_data

    def get_dividend_data(self, symbol_list=None):
        '''
        If Dividend data exists grab data from file.
        Else fetch data from YAHOO.
        '''
        symbol_list = self.get_symbol_list(symbol_list=symbol_list)
        try:
            div_data = pd.read_hdf(SYMBOL_DATA_PATH, 'dividend_data')
        except:
            self.force_load_data(force_load='symbol_dividend')
            div_data = pd.read_hdf(SYMBOL_DATA_PATH, 'dividend_data')

        div_data = div_data[div_data.symbol.isin(symbol_list.index)]
        return div_data

    def get_tech_data(self, symbol_list=None, index=None,
                      start=None, end=TODAY):
        '''
        If TECH_DATA exists grab data from file.
        Update data if data in the file is older than 3 days.
        Else fetch data from TECHPAISA website.
        '''
        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           start=start, index=index)
        try:
            tech_data = pd.read_hdf(SYMBOL_DATA_PATH, 'tech_data_daily')
        except:
            self.force_load_data(force_load='symbol_tech')
            tech_data = pd.read_hdf(SYMBOL_DATA_PATH, 'tech_data_daily')

        tech_data = tech_data[tech_data.symbol.isin(symbol_list.index)]
        start = self.get_date(start, start=True)
        end = self.get_date(end, start=False)
        tech_data = tech_data.ix[(tech_data.date >= start) & (tech_data.date <= end)]
        return tech_data

    def get_symbol_data(self, data='returns', symbol_list=None, start=None,
                        end=None, index=None, null_count=None,
                        volume=None, mcap=None):
        '''Get Close prices for historical as a separate dataframe'''

        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           start=start, index=index,
                                           mcap=mcap, volume=volume)
        hist_data_columns = [
            'open', 'high', 'low', 'last', 'close', 'vwap',
            'simple_returns', 'log_returns', 'daily_volatility',
            'volume', 'turnover', 'pct_deliverble'
        ]
        if data in hist_data_columns:
            values = data
        elif data == 'returns':
            values = 'log_returns'
        elif data == 'volatility':
            values = 'daily_volatility'
        elif data == 'deliverable':
            values = 'pct_deliverable'
        else:
            warnings.warn(
                'Invalid type of data requested. Returning returns data'
            )
            values = 'log_returns'
        try:
            data = pd.read_hdf(
                SYMBOL_DATA_PATH, 'symbol_{0}'.format(values)
            )
        except:
            self.force_load_data(force_load='symbol_close', values=values)
            data = pd.read_hdf(
                SYMBOL_DATA_PATH, 'symbol_{0}'.format(values)
            )

        data = data[symbol_list.index]
        start = self.get_date(start, 'str', True)
        end = self.get_date(end, 'str', False)
        data = data[start:end]
        if null_count is not None:
            data = data.interpolate(
                method='time', limit=null_count,
                limit_direction='backward'
            )
            symbols = data.columns[data.count() == len(data)]
            data = data[symbols]
        data = data.dropna(how='all', axis=1)
        return data

    def get_symbol_dat(self, symbol, start=None, end=None):
        '''Get the historical data of a single symbol.'''
        data = self.get_symbol_hist([symbol], start=start, end=end)
        data = data.drop(['symbol'], axis=1).set_index('date')
        return data

    def get_correlation_matrix(self, symbol_list=None, index=None,
                               returns=None, null_count=None, end=None,
                               periods=[1, 3, 5, 10, 15, 20],
                               period_type='years'):
        '''Get correlation matrix for all symbols in symbol_ist'''
        end = self.get_date(end, start=False)
        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           index=index, start=end)
        if returns is None:
            returns = self.get_symbol_returns(symbol_list=symbol_list,
                                              end=end, null_count=null_count)
        elif null_count is None:
            returns = returns.interpolate(
                method='time', limit=null_count,
                limit_direction='backward'
            )
            symbols = returns.columns[returns.count() == len(returns)]
            returns = returns[symbols]
        correlation_matrix = pd.DataFrame(index=returns.columns)
        for period in periods:
            to_date = end

            if period_type == 'years':
                from_date = end - relativedelta(years=period)
            elif period_type == 'months':
                from_date = end - relativedelta(months=period)
            elif period_type == 'days':
                from_date = end - relativedelta(days=period)

            from_date = self.get_date(from_date, 'str')
            to_date = self.get_date(to_date, 'str')

            slice_ret = returns[from_date:to_date]
            slice_corr = slice_ret.corr()
            correlation_matrix['coeff_{0}'.format(period)] = slice_corr.min()
            correlation_matrix['symbol_{0}'.format(period)] = slice_corr.idxmin()

        return correlation_matrix.dropna(how='all')

    def industry_to_symbol_list(self, industry='technology'):
        symbol_meta = self.get_symbol_meta()
        if isinstance(industry, list):
            industry = pd.Series(industry)
        symbol_list = symbol_meta.from_date[symbol_meta.industry.isin(industry)]
        return symbol_list

    def describe_returns(self, returns):
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

    def force_load_data(self, force_load, values=None):
        try:
            os.remove(TEMP_DATA_PATH)
        except:
            pass

        if force_load == 'symbol_meta':
            print('Loading Symbol Meta data from NSE website')
            symbol_meta = self.fetch_symbol_meta()
            symbol_meta.ix[symbol_meta.date_of_listing < datetime(1996, 1, 1), 'date_of_listing'] = datetime(1996, 1, 1)
            symbol_meta.to_hdf(SYMBOL_DATA_PATH, 'symbol_meta')
            # Update industry field
            try:
                self.fetch_tech_data(symbol_meta=symbol_meta)
            except Exception as e:
                warnings.warn(
                    'Unable to connect to techpaisa site due to {0}'.format(e)
                )
            # Update dates for symbol meta with dates from symbol_hist
            self.update_symbol_meta_dates()
            symbol_meta = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_meta')

        elif force_load == 'symbol_hist':
            self.update_symbol_meta_dates()
            symbol_meta = self.get_symbol_meta()
            date_diff = (TODAY - symbol_meta.to_date).days
            symbol_meta = symbol_meta[
                date_diff > 5 | (symbol_meta.row_count == 0)
            ]
            print('Fetching Data from NSE website for {0} symbols'.format(len(symbol_meta)))
            try:
                os.remove(TEMP_DATA_PATH)
            except:
                pass
            # symbol_list = symbol_list[symbol_list.index.isin(['arvinfra'])]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for symbol in symbol_meta.itertuples():
                    symbol_executor = executor.submit(
                        self.fetch_historical_data,
                        symbol=symbol.Index,
                        start=symbol.to_date
                    )
                    nse_data = symbol_executor.result()
                    if nse_data.empty:
                        continue
                    nse_data['symbol'] = symbol.Index.lower()
                    nse_data = nse_data[nse_data.date >= symbol.date_of_listing]
                    with SafeHDFStore(TEMP_DATA_PATH) as store:
                        store.put('symbol_hist_data_temp', value=nse_data, format='t',
                                  append=True, min_itemsize={'symbol': 15})

            hist_data_temp = pd.read_hdf(TEMP_DATA_PATH, 'symbol_hist_data_temp')
            try:
                hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')
            except:
                hist_data = pd.DataFrame(columns=hist_data_temp.columns)

            hist_data = hist_data.append(hist_data_temp)
            del hist_data_temp
            os.remove(TEMP_DATA_PATH)

            hist_data = hist_data.drop_duplicates(['symbol', 'date'], keep='last')
            hist_data = hist_data.sort_values(['symbol', 'date'])
            hist_symbol_schema = [
                'symbol', 'date', 'prev_close', 'open', 'high',
                'low', 'last', 'close', 'vwap',
                'simple_returns', 'log_returns', 'daily_volatility',
                'volume', 'turnover', 'pct_deliverble'
            ]
            hist_data = hist_data.reset_index()[hist_symbol_schema]
            hist_data.to_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')
            self.update_symbol_meta_dates()
            hist_data_columns = [
                'open', 'high', 'low', 'last', 'close', 'vwap',
                'simple_returns', 'log_returns', 'daily_volatility',
                'volume', 'turnover', 'pct_deliverble'
            ]
            for col in hist_data_columns:
                self.force_load_data(force_load='symbol_data', values=col)

        elif force_load == 'symbol_dividend':
            print('Fetching Dividend data for all symbols from Yahoo Finance')
            symbol_list = self.get_symbol_list()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor: # noqa
                for symbol in symbol_list.index:
                    p = executor.submit(self.fetch_dividend_data, symbol)
                    dividend_data = p.result()
                    if dividend_data.empty:
                        continue
                    dividend_data = dividend_data.reset_index().drop_duplicates(subset=['date'],
                                                                                keep='last')
                    dividend_data = dividend_data.sort_values('date')
                    dividend_data['symbol'] = [symbol for i in range(0, len(dividend_data))]
                    dividend_data = dividend_data.reset_index()[DIVIDEND_SCHEMA]
                    with SafeHDFStore(TEMP_DATA_PATH) as store:
                        store.put('temp_dividend_data', value=dividend_data, format='t',
                                  append=True, min_itemsize={'symbol': 15, 'action': 20})

            div_data = pd.read_hdf(TEMP_DATA_PATH, 'temp_dividend_data')
            os.remove(TEMP_DATA_PATH)

            div_data = div_data.sort_values(['symbol', 'date'])
            div_data = div_data.reset_index()[DIVIDEND_SCHEMA]
            div_data = div_data.drop_duplicates(['symbol', 'date'], keep='last')
            div_data.to_hdf(SYMBOL_DATA_PATH, 'dividend_data')

        elif force_load == 'symbol_tech':
            print('Fetching Tech Data for all symbols for today from techpaisa website')
            tech_data_temp = self.fetch_tech_data()
            try:
                tech_data = pd.read_hdf(SYMBOL_DATA_PATH, 'tech_data_daily')
            except:
                tech_data = pd.DataFrame(columns=TECH_STRENGTH_SCHEMA)

            tech_data = tech_data.append(tech_data_temp)
            del tech_data_temp
            os.remove(TEMP_DATA_PATH)

            tech_data.symbol = tech_data.symbol.str.lower().str.strip()
            tech_data = tech_data.drop_duplicates(['symbol', 'date'], keep='last')
            tech_data = tech_data.sort_values(['date', 'symbol'])
            tech_data = tech_data.reset_index()[TECH_STRENGTH_SCHEMA]
            tech_data.to_hdf(SYMBOL_DATA_PATH, 'tech_data_daily')

        elif force_load == 'symbol_data':
            print('Generating time series data for {0} from local data'.format(values))
            hist_data = self.get_symbol_hist(start='1990')

            data = pd.pivot_table(data=hist_data, index='date',
                                  columns='symbol', values=values)
            data.to_hdf(SYMBOL_DATA_PATH, 'symbol_{0}'.format(values))
        elif force_load == 'all':
            self.force_load_data('symbol_meta')
            self.force_load_data('symbol_hist')
            self.force_load_data('symbol_dividend')
            self.force_load_data('symbol_tech')
        clean_file(SYMBOL_DATA_PATH)

    def __init__(self, symbol_list=None, index=None,
                 null_count=None, start=None, end=None, min_rows=None,
                 force_load=False, volume=None, mcap=None):
        '''
        Symbol Object containing all the necessary methods
        for handling symbol related functions
        '''
        if force_load is not None:
            self.force_load_data(force_load)

        self.symbol_meta = self.get_symbol_meta()
        self.start = self.get_date(start, start=True)
        self.end = self.get_date(end, start=False)
        self.symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index,
            start=start, min_rows=min_rows,
            volume=volume, mcap=mcap
        )
