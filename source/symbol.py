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

    def get_symbol_index(self, index_type='all'):
        if isinstance(index_type, str):
            if index_type == 'all':
                index_symbol_list = self.get_symbol_index(
                    ['mkt_index', 'sec_index', 'stg_index', 'thematic_index']
                )
            else:
                try:
                    index_symbol_list = pd.read_hdf(SYMBOL_DATA_PATH, index_type)
                except:
                    self.force_load_data(force_load='index_to_symbol')
                    index_symbol_list = pd.read_hdf(SYMBOL_DATA_PATH, index_type)
        elif isinstance(index_type, list):
            index_symbol_list = pd.DataFrame()
            for index_type_element in index_type:
                temp_index_symbol_list = pd.read_hdf('symbol_data.h5', index_type_element)
                if index_symbol_list.empty:
                    index_symbol_list = temp_index_symbol_list
                else:
                    index_symbol_list = index_symbol_list.join(temp_index_symbol_list)
            return index_symbol_list

        return index_symbol_list
 
    def index_to_symbol_list(self, index=None, index_type=None, fetch_type='union'):
        if index_type is not None:
            index_ = self.get_symbol_index(index_type)
        elif index is not None:
            index_ = self.get_symbol_index('all')
            index_ = index_[index]
            index_ = pd.DataFrame(index_)
        col_list = index_.columns

        # list to query
        if fetch_type == 'union':
            query = ' | '.join(col_list)
        elif fetch_type == 'intersection':
            query = ' & '.join(col_list)
        symbol_list = index_.query(query)
        return symbol_list.index.tolist()

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
            symbol_list = self.index_to_symbol_list(index)

        if min_rows is not None:
            temp_smeta = symbol_meta[symbol_meta.index.isin(symbol_list.index)]
            symbol_list = symbol_list[temp_smeta.row_count >= min_rows]

        if volume is not None:
            volume_data = self.get_symbol_data(data='volume', symbol_list=symbol_list, start=start)
            volume_data = volume_data.min()
            symbol_list = symbol_list[symbol_list.index.isin(volume_data.index)]
            symbol_list = symbol_list[
                volume_data >= volume
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
            hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_hist_data')
        except Exception as e:
            self.force_load_data(force_load='symbol_hist')
            hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_hist_data')

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
                SYMBOL_DATA_PATH, 'symbol_data_{0}'.format(values)
            )
        except:
            self.force_load_data(force_load='symbol_data', values=values)
            data = pd.read_hdf(
                SYMBOL_DATA_PATH, 'symbol_data_{0}'.format(values)
            )
        column_list = data.columns
        column_list = data.columns.intersection(symbol_list.index)
        data = data[column_list]
        start = self.get_date(start, 'str', True)
        end = self.get_date(end, 'str', False)
        data = data[start:end]
        if null_count is not None:
            data_na = data.fillna(0, limit=null_count)
            symbols = data_na.columns[data_na.count() == len(data_na)]
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

    def force_load_data(self, force_load, values=None):
        try:
            os.remove(TEMP_DATA_PATH)
        except:
            pass

        if force_load == 'symbol_meta':
            print('Loading Symbol Meta data from NSE website')
            symbol_meta = self.fetch_symbol_meta()

            # Change date of listing of symbols if lower than Jan 1 1996
            # to Jan 1 1996
            symbol_meta.ix[symbol_meta.date_of_listing < datetime(1996, 1, 1), 'date_of_listing'] = datetime(1996, 1, 1)
            symbol_meta.to_hdf(SYMBOL_DATA_PATH, 'symbol_meta')

            # Update dates for symbol meta with dates from symbol_hist
            self.update_symbol_meta_dates()

        elif force_load == 'index_to_symbol':
            INDEX_META = pd.read_hdf('constants.h5', 'index_meta')
            symbol_meta = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_meta')
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                for index_type in INDEX_META.category.unique():
                    index_type_data = pd.DataFrame(index=symbol_meta.index)
                    index_list = INDEX_META.query(
                        'category == "{0}"'.format(index_type)
                    )['url']
                    for index_name in index_list.index:
                        index_data = executor.submit(self.fetch_index_list_of_symbols, index_name)
                        index_data = index_data.result()
                        if index_data.empty:
                            continue
                        index_data[index_name] = True
                        index_data = index_data[index_name]
                        print(
                            'Updated index to symbol list for {0} index'.format(index_name)
                        )
                        index_type_data = index_type_data.join(index_data)
                    index_type_data = index_type_data.fillna(False)
                    with SafeHDFStore(SYMBOL_DATA_PATH) as store:
                        store.put(index_type, value=index_type_data)

        elif force_load == 'symbol_hist':
            self.update_symbol_meta_dates()
            symbol_meta = self.get_symbol_meta()
            date_diff = (TODAY - symbol_meta.to_date).dt.days
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
                hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_hist_data')
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
            hist_data.to_hdf(SYMBOL_DATA_PATH, 'symbol_hist_data')
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
            hist_data = self.get_symbol_hist(symbol_list=self.symbol_meta.from_date)

            data = pd.pivot_table(data=hist_data, index='date',
                                  columns='symbol', values=values)
            data.to_hdf(SYMBOL_DATA_PATH, 'symbol_data_{0}'.format(values))
        elif force_load == 'all':
            self.force_load_data('symbol_meta')
            self.force_load_data('symbol_hist')
            self.force_load_data('symbol_dividend')
            self.force_load_data('symbol_tech')
        clean_file(SYMBOL_DATA_PATH)

    def __init__(self, symbol_list=None, index=None,
                 start=None, end=None, min_rows=None,
                 force_load=None, volume=None, mcap=None):
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
