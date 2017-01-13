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
HIST_SYMBOL_SCHEMA = ['symbol', 'date', 'prev_close', 'open', 'high',
                      'low', 'last', 'close', 'vwap', 'volume',
                      'turnover', 'pct_deliverble', 'simple_returns',
                      'log_returns', 'daily_volatility']
DIVIDEND_SCHEMA = ['symbol', 'date', 'action', 'value']
TECH_STRENGTH_SCHEMA = ['date', 'symbol', 'tech_strength']

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())


class Symbol(Market):

    def get_symbol_list(self, symbol_list=None, index=None, hist_data=None,
                        null_count=None, start=None, min_rows=None,
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

        if index is not None:
            index = index.lower().replace(' ', '_')
            symbol_list = self.index_to_symbol_list(index)

        if hist_data is not None:
            first_index = hist_data.groupby('symbol').date.min()
            symbol_list = symbol_list[first_index.index[first_index <= start]]

        if min_rows is not None:
            temp_smeta = symbol_meta[symbol_meta.index.isin(symbol_list.index)]
            symbol_list = symbol_list[temp_smeta.row_count >= min_rows]

        if null_count is not None:
            start = self.get_date(start, 'dt')
            returns = self.get_symbol_returns(symbol_list=symbol_list, start=start)
            returns = returns.interpolate(method='time', limit=null_count, limit_direction='backward')
            symbols = returns.columns[returns.count() == len(returns)]
            print(returns.count()['rblbank'])
            symbol_list = symbol_list[symbol_list.index.isin(symbols)]
            if symbol_list.empty:
                symbol_list = symbol_list[symbol_list == symbol_list.min()]

        if volume is not None:
            start = self.get_date(start, 'dt')
            symbol_list = symbol_list[symbol_list <= start]
            hist_data = self.get_symbol_hist(symbol_list=symbol_list, start=start)
            symbol_list = symbol_list[
                hist_data.groupby('symbol').aggregate({'volume': 'min'}).volume >= volume
            ]
        if mcap is not None:
            temp_smeta = symbol_meta[symbol_meta.index.isin(symbol_list.index)]
            symbol_list = symbol_list[temp_smeta.mcap >= mcap]
        return symbol_list

    def get_symbol_meta(self, force_load=False):
        '''
        If symbol meta data exists grab data from file.
        Else fetch symbol meta data from NSE website.
        '''
        if not force_load:
            try:
                symbol_meta = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_meta')
            except:
                warnings.warn('symbol meta is not available locally, Fetching data from NSE website')
                symbol_meta = self.get_symbol_meta(force_load=True)
        else:
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

        return symbol_meta

    def update_symbol_meta_dates(self,):
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
                        volume=None, mcap=None,
                        force_load=False):
        '''
        If SYMBOL_DATA_PATH exists grab data from file.
        Update data if data in the file is older than 5 days.
        Else fetch symbol data from NSE website.
        '''
        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           index=index,
                                           volume=volume, mcap=mcap,
                                           start=start)

        if not force_load:
            try:
                hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')
            except Exception as e:
                warnings.warn('Unable to open HDF file due to {0}'.format(e))
                hist_data = self.get_symbol_hist(force_load=True)
        else:
            self.update_symbol_meta_dates()
            symbol_meta = self.get_symbol_meta()
            symbol_meta = symbol_meta[
                (symbol_meta.to_date < TODAY) | (symbol_meta.row_count == 0)
            ]
            try:
                os.remove(TEMP_DATA_PATH)
            except:
                pass
            # symbol_list = symbol_list[symbol_list.index.isin(['arvinfra'])]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for symbol in symbol_meta.itertuples():
                    p = executor.submit(
                        self.fetch_historical_data,
                        symbol=symbol.Index,
                        start=symbol.to_date
                    )
                    nse_data = p.result()
                    if nse_data.empty:
                        nse_data = pd.DataFrame(
                            columns=HIST_SYMBOL_SCHEMA, dtype=float
                        )
                        nse_data['date'] = pd.to_datetime('1994-01-01')
                    nse_data['symbol'] = symbol.Index.lower()
                    nse_data = nse_data[nse_data.date >= symbol.date_of_listing]
                    symbol_meta.ix[symbol.Index, ['from_date', 'to_date', 'row_count']] = (
                        nse_data.date.min(), nse_data.date.max(), len(nse_data)
                    )
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
            hist_data = hist_data.reset_index()[HIST_SYMBOL_SCHEMA]
            hist_data.to_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')
            self.update_symbol_meta_dates()
            clean_file(SYMBOL_DATA_PATH)
            return hist_data

        hist_data = hist_data[hist_data.symbol.isin(symbol_list.index)]
        start = self.get_date(start, start=True)
        end = self.get_date(end, start=False)
        hist_data = hist_data.ix[(hist_data.date >= start) & (hist_data.date <= end)]
        return hist_data

    def get_dividend_data(self, symbol_list=None, force_load=False):
        '''
        If Dividend data exists grab data from file.
        Else fetch data from YAHOO.
        '''
        symbol_list = self.get_symbol_list(symbol_list=symbol_list)

        if not force_load:
            try:
                div_data = pd.read_hdf(SYMBOL_DATA_PATH, 'dividend_data')
            except Exception as e:
                warnings.warn('Unable to open Dividend file due to {0}'.format(e))
                div_data = self.get_dividend_data(force_load=True)
        else:
            try:
                os.remove(TEMP_DATA_PATH)
            except:
                pass
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
            clean_file(SYMBOL_DATA_PATH)
            return div_data

        div_data = div_data[div_data.symbol.isin(symbol_list.index)]
        return div_data

    def get_tech_data(self, symbol_list=None, index=None,
                      start=None, end=TODAY,
                      force_load=False):
        '''
        If TECH_DATA exists grab data from file.
        Update data if data in the file is older than 3 days.
        Else fetch data from TECHPAISA website.
        '''
        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           start=start, index=index)
        if not force_load:
            try:
                tech_data = pd.read_hdf(SYMBOL_DATA_PATH, 'tech_data_daily')
            except Exception as e:
                warnings.warn('Unable to open HDF file due to {0}'.format(e))
                tech_data = pd.DataFrame(columns=TECH_STRENGTH_SCHEMA)
            if int((TODAY - tech_data.date.max()).days) > 1:
                print('Getting tech data from techpaisa website')
                tech_data = self.get_tech_data(force_load=True)

        else:
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
            clean_file(SYMBOL_DATA_PATH)

        tech_data = tech_data[tech_data.symbol.isin(symbol_list.index)]
        start = self.get_date(start, start=True)
        end = self.get_date(end, start=False)
        tech_data = tech_data.ix[(tech_data.date >= start) & (tech_data.date <= end)]
        return tech_data

    def get_symbol_close(self, start=None, end=None,
                         symbol_list=None, hist_data=None,
                         volume=None, mcap=None):
        '''Get Close prices for historical as a separate dataframe'''
        start = self.get_date(start, start=True)
        end = self.get_date(end, start=False)
        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           start=start,
                                           hist_data=hist_data)

        if hist_data is None:
            hist_data = self.get_symbol_hist(symbol_list=symbol_list)

        close = pd.pivot_table(data=hist_data, index='date',
                               columns='symbol', values='close')
        start = self.get_date(start, start=True, out='str')
        end = self.get_date(end, start=False, out='str')
        return close[start:end]

    def get_symbol_returns(self, symbol_list=None, start=None,
                           end=None, index=None, null_count=None,
                           force_load=False,
                           volume=None, mcap=None):
        '''Get returns data'''

        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           start=start, index=index)
        if not force_load:
            try:
                ret = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_returns')
            except:
                ret = self.get_symbol_returns(force_load=True)
        else:
            hist_data = self.get_symbol_hist(start='1990')

            ret = pd.pivot_table(data=hist_data, index='date',
                                 columns='symbol', values='log_returns')
            ret.to_hdf(SYMBOL_DATA_PATH, 'symbol_returns')
            clean_file(SYMBOL_DATA_PATH)
            return ret

        ret = ret[ret.columns[ret.columns.isin(symbol_list.index)]]
        start = self.get_date(start, 'str', True)
        end = self.get_date(end, 'str', False)
        ret = ret[start:end]
        if null_count is not None:
            ret = ret.interpolate(method='time', limit=null_count, limit_direction='backward')
            symbols = ret.columns[ret.count() == len(ret)]
            ret = ret[symbols]
        ret = ret.dropna(how='all', axis=1)
        return ret

    def symbol_returns_null_count(self, ret=None, symbol_list=None,
                                  index=None, force_load=False):
        '''Get returns data'''
        if ret is not None:
            returns_null_count = self.consecutive_nulls(ret)
            return returns_null_count

        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           index=index)
        if not force_load:
            try:
                returns_null_count = pd.read_hdf(SYMBOL_DATA_PATH, 'returns_null_count')
            except:
                returns_null_count = self.consecutive_nulls(force_load=True)
        else:
            returns = self.get_symbol_returns(start='1994')

            returns_null_count = self.consecutive_nulls(returns)
            returns_null_count.to_hdf(SYMBOL_DATA_PATH, 'returns_null_count')
            clean_file(SYMBOL_DATA_PATH)
            return returns_null_count

        return returns_null_count

    def get_symbol_data(self, symbol, start=None, end=None):
        '''Get the historical data of a single symbol.'''
        data = self.get_symbol_hist([symbol], start=start, end=end)
        data = data.drop(['symbol'], axis=1).set_index('date')
        return data

    def get_correlation_matrix(self, symbol_list=None, index=None, null_count=None,
                               end=None, ret=None,
                               periods=[1, 3, 5, 10, 15, 20], period_type='years'):
        '''Get correlation matrix for all symbols in symbol_ist'''
        end = self.get_date(end, start=False)
        symbol_list = self.get_symbol_list(symbol_list=symbol_list,
                                           index=index,
                                           start=end,
                                           null_count=null_count)
        if ret is None:
            ret = self.get_symbol_returns(symbol_list=symbol_list,
                                          end=end, null_count=null_count)
        correlation_matrix = pd.DataFrame(index=ret.columns)
        for i in periods:
            to_date = end

            if period_type == 'years':
                from_date = end - relativedelta(years=i)
            elif period_type == 'months':
                from_date = end - relativedelta(months=i)
            elif period_type == 'days':
                from_date = end - relativedelta(days=i)

            from_date = self.get_date(from_date, 'str')
            to_date = self.get_date(to_date, 'str')

            slice_ret = ret[from_date:to_date]
            slice_corr = slice_ret.corr()
            correlation_matrix['coeff_{0}'.format(i)] = slice_corr.min()
            correlation_matrix['symbol_{0}'.format(i)] = slice_corr.idxmin()

        return correlation_matrix.dropna(how='all')

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
                raise ValueError('Invalid type')
        else:
            symbol_list = symbol_meta.from_date[symbol_meta[index]]
        return symbol_list

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
        returns_describe = returns_describe.drop(returns_describe.ix[:, 4:7].columns, axis=1)
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

    def __init__(self, symbol_list=None, index=None,
                 null_count=None, start=None, end=None, min_rows=None,
                 force_load=False, volume=None, mcap=None):
        '''
        Symbol Object containing all the necessary methods
        for handling symbol related functions
        '''
        if str(force_load) == 'symbol_meta':
            self.get_symbol_meta(force_load=True)
        elif str(force_load) == 'hist_data':
            self.get_symbol_hist(force_load=True)
        elif str(force_load) == 'dividend_data':
            self.get_dividend_data(force_load=True)
        elif str(force_load) == 'returns':
            self.get_symbol_returns(force_load=True)
        elif force_load:
            self.get_symbol_meta(force_load=True)
            self.get_symbol_hist(force_load=True)
            self.get_dividend_data(force_load=True)
            self.get_symbol_returns(force_load=True)

        self.symbol_meta = self.get_symbol_meta()
        self.start = self.get_date(start, start=True)
        self.end = self.get_date(end, start=False)
        self.symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index,
            null_count=null_count, start=start, min_rows=min_rows,
            volume=volume, mcap=mcap
        )
