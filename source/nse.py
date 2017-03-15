'''Module for loading data from NSE website'''
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from nsepy import get_history

from helpers import clean_file, get_date, get_store_keys, rename_columns
from market import Market

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())


class NSE(Market):
    '''
    Module for NSE Data
    '''

    __CURRENT_PATH = os.path.dirname(__file__)
    __NSE_DATA_PATH = os.path.join(__CURRENT_PATH, 'data{0}nse_data.h5'.format(os.sep))
    NSE_DATA_PATH = __NSE_DATA_PATH
    __CONSTANTS_PATH = os.path.join(__CURRENT_PATH, 'data{0}constants.h5'.format(os.sep))

    __SYMBOL_EOD_DATA_KEY = 'symbol_eod_data'
    __SYMBOL_EOD_META_KEY = 'symbol_eod_meta'
    __INDEX_EOD_DATA_KEY = 'index_eod_data'
    __INDEX_EOD_META_KEY = 'index_eod_meta'

    def fetch_eod_data(
            self, symbol, start=None, index=False
        ):
        'Fetch all End of Day(EOD) data from NSE'
        from_date = get_date(start, start=True)
        to_date = get_date(start=False)
        # Get data from NSE
        try:
            eod_data = get_history(
                symbol=symbol, index=index,
                start=from_date, end=to_date, series='EQ'
            )
            if eod_data.empty:
                warnings.warn(
                    'No data recieved from NSE for {0} from {1} to {2}'.
                    format(symbol, from_date.date(), to_date.date())
                )
                return eod_data
            eod_data.index = pd.to_datetime(eod_data.index)
            if index:
                eod_data['prev_close'] = eod_data['Close'].shift(1)
                eod_data['pct_deliverble'] = 100
                eod_data['vwap'] = eod_data['Close']
                eod_data['last'] = eod_data['Close']
                eod_data['trades'] = 0
            else:
                eod_data.drop(
                    ['Series', 'Deliverable Volume'], 1, inplace=True
                )
        except Exception as exception:
            warnings.warn(
                'Could not get data for {0} from NSE due to {1}'.format(symbol, exception)
            )
            return pd.DataFrame()

        rename_columns(eod_data)
        eod_data['symbol'] = [symbol for i in range(len(eod_data))]
        eod_data = eod_data.reset_index().sort_values(['symbol', 'date', 'close'])
        eod_data = eod_data.drop_duplicates(
            subset=['symbol', 'date'], keep='last'
        )

        # Handle prev_close = 0
        eod_data['prev_close_test'] = eod_data.close.shift(1)
        eod_data.loc[eod_data.prev_close == 0, 'prev_close'] = eod_data.loc[eod_data.prev_close == 0, 'prev_close_test']
        eod_data.drop(['prev_close_test'], axis=1, inplace=True)

        from_date = eod_data.date.min()
        to_date = eod_data.date.max()
        traded_dates = self.get_traded_dates(
            start=from_date,
            end=to_date
        )
        traded_dates = pd.DataFrame(index=traded_dates.index)
        missing_dates = traded_dates.index.difference(eod_data.date)
        eod_data = traded_dates.join(eod_data.set_index('date'), how='outer')
        traded_dates = pd.DataFrame(index=eod_data.index)
        traded_dates['date_count'] = [i+1 for i in range(len(traded_dates))]

        if len(missing_dates) > 0:
            for i in missing_dates:
                date_count = traded_dates.loc[i]['date_count']
                back_date = traded_dates[traded_dates.date_count == date_count-1].index.values[0]
                next_first_valid_date = eod_data.loc[i:].symbol.first_valid_index()
                if next_first_valid_date is None:
                    next_first_valid_date = TODAY
                if eod_data.loc[back_date, 'close'] == eod_data.loc[next_first_valid_date, 'prev_close']:
                    close = eod_data.loc[back_date, 'close']
                    eod_data.loc[i, ['symbol']] = symbol
                    eod_data.loc[i, ['prev_close', 'open', 'high', 'low', 'last', 'close', 'vwap']] = close
                    eod_data.loc[i, ['volume', 'turnover', 'trades', 'pct_deliverble']] = 0
        missing_count = len(traded_dates) - eod_data.symbol.count()
        if missing_count > 0:
            warnings.warn(
                ' {0} missing rows in {1}'.format(missing_count, symbol)
            )
        eod_data['simple_returns'] = (
            (eod_data.close - eod_data.prev_close) / eod_data.prev_close
        )
        eod_data['log_returns'] = np.log(eod_data.close / eod_data.prev_close)
        eod_data['high_low_spread'] = (eod_data.high - eod_data.low) / eod_data.low * 100
        eod_data['open_close_spread'] = (eod_data.close - eod_data.open) / eod_data.open * 100
        eod_data['pct_deliverble'] = eod_data['pct_deliverble'] * 100

        # Handle abnormal returns. i.e Splits
        abnormal_condition = (eod_data.simple_returns < -0.35) | (eod_data.simple_returns > 0.35)
        eod_data.loc[abnormal_condition, ['simple_returns']] = (
            (eod_data[abnormal_condition].high_low_spread + eod_data[abnormal_condition].open_close_spread) / (2 * 100)
        )
        eod_data.loc[abnormal_condition, ['log_returns']] = (
            (eod_data[abnormal_condition].high_low_spread + eod_data[abnormal_condition].open_close_spread) / (2 * 100)
        )
        eod_data.index.name = 'date'
        eod_data = eod_data.reset_index().set_index(['symbol', 'date'])
        eod_data = eod_data.astype(np.float)
        if index:
            eod_data = eod_data.drop(['pct_deliverble', 'vwap', 'last', 'trades'], axis=1)
        return eod_data

    def get_symbol_list(
            self, symbol_list=None, index=None, index_type=None, start=None,
            min_rows=0, missing_count=0
        ):
        '''
        Get symbol list based on criteria provided.
        Pass index for getting symbols in index.
        List of indexes to get union of symbols of all indexes in list.
        start: {year as int or string, string format of date, None}
        min_rows: int- get symbols with rows more than min_rows
        '''
        symbol_list = super().get_symbol_list(
            symbol_list=symbol_list, index=index, index_type=index_type, start=start
        )

        if min_rows != 0 or missing_count != 0:
            symbol_data_meta = self.get_eod_meta(eod_type='symbol')
            symbol_list = symbol_data_meta.query(
                'row_count >= @min_rows and missing_count >= @missing_count'
            )
            symbol_list = symbol_list.index.tolist()

        # if volume is not None:
        #    volume_data = self.get_symbol_data(data='volume', symbol_list=symbol_list, start=start)
        #     volume_data = volume_data.median()
        #     symbol_list = symbol_list[symbol_list.index.isin(volume_data.index)]
        #     symbol_list = symbol_list[
        #         volume_data >= volume
        #     ]

        # if mcap is not None:
        #     temp_smeta = symbol_meta[symbol_meta.index.isin(symbol_list.index)]
        #     symbol_list = symbol_list[temp_smeta.mcap >= mcap]
        return symbol_list

    def get_symbol_eod_data(
            self, symbol_list=None,
            index=None, index_type=None, start=None, end=None,
            min_rows=0, missing_count=0
        ):
        '''
        If SYMBOL_DATA_PATH exists grab data from file.
        Update data if data in the file is older than 5 days.
        Else fetch symbol data from NSE website.
        '''
        if NSE.__SYMBOL_EOD_DATA_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
            eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__SYMBOL_EOD_DATA_KEY)
            eod_data = eod_data.reset_index()
        else:
            self.force_load_data(force_load='symbol_eod_data')
            eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__SYMBOL_EOD_DATA_KEY)
            eod_data = eod_data.reset_index()
        symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index, index_type=index_type,
            start=start, missing_count=missing_count, min_rows=min_rows
        )
        eod_data = eod_data[eod_data.symbol.isin(symbol_list)]
        start = get_date(start, out='dt', start=True)
        end = get_date(end, out='dt', start=False)
        eod_data = eod_data.loc[
            (eod_data.date >= start) & (eod_data.date <= end)
        ]
        return eod_data

    def get_symbol_eod_values(
            self, data='returns', symbol_list=None,
            index=None, index_type=None, start=None, end=None,
            min_rows=0, missing_count=0
        ):
        '''Get Close prices for historical as a separate dataframe'''

        symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index, index_type=index_type,
            start=start, missing_count=missing_count, min_rows=min_rows
        )
        eod_data_schema = [
            'symbol', 'date', 'prev_close', 'open', 'high',
            'low', 'last', 'close', 'vwap',
            'trades', 'volume', 'turnover', 'pct_deliverble',
            'simple_returns', 'log_returns', 'high_low_spread', 'open_close_spread'
        ]
        if data in eod_data_schema:
            values = data
        elif data == 'returns':
            values = 'log_returns'
        elif data == 'deliverble':
            values = 'pct_deliverble'
        else:
            warnings.warn(
                'Invalid type of data requested. Returning returns data'
            )
            values = 'log_returns'
        if 'symbol_eod_values_{0}'.format(values) in get_store_keys(NSE.__NSE_DATA_PATH):
            data = pd.read_hdf(
                NSE.__NSE_DATA_PATH, 'symbol_eod_values_{0}'.format(values)
            )
        else:
            self.force_load_data(force_load='symbol_eod_values', values=values)
            data = pd.read_hdf(
                NSE.__NSE_DATA_PATH, 'symbol_eod_values_{0}'.format(values)
            )
        column_list = data.columns
        column_list = data.columns.intersection(symbol_list)
        data = data[column_list]
        start = get_date(start, 'str', True)
        end = get_date(end, 'str', False)
        data = data[start:end]
        data = data.dropna(how='all', axis=1)
        return data

    def get_index_eod_data(
            self, index_list=None, index_type=None,
            start=None, end=None,
        ):
        '''
        TODO
        If SYMBOL_DATA_PATH exists grab data from file.
        Update data if data in the file is older than 5 days.
        Else fetch symbol data from NSE website.
        '''
        if NSE.__INDEX_EOD_DATA_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
            eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__INDEX_EOD_DATA_KEY)
            eod_data = eod_data.reset_index()
        else:
            self.force_load_data(force_load='index_eod_data')
            eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__INDEX_EOD_DATA_KEY)
            eod_data = eod_data.reset_index()

        index_list = self.get_index_list(
            index_list=index_list, index_type=index_type
        )

        eod_data = eod_data[eod_data.symbol.isin(index_list)]
        start = get_date(start, out='dt', start=True)
        end = get_date(end, out='dt', start=False)
        eod_data = eod_data.ix[
            (eod_data.date >= start) & (eod_data.date <= end)
        ]
        return eod_data

    def get_index_eod_values(
            self, data='returns', index_list=None, index_type=None,
            start=None, end=None,
        ):
        '''Get Close prices for historical as a separate dataframe'''

        index_list = self.get_index_list(
            index_list=index_list, index_type=index_type
        )
        eod_data_schema = [
            'symbol', 'date', 'open', 'high',
            'low', 'last', 'close', 'volume', 'turnover',
            'simple_returns', 'log_returns', 'high_low_spread', 'open_close_spread'
        ]
        if data in eod_data_schema:
            values = data
        elif data == 'returns':
            values = 'log_returns'
        else:
            warnings.warn(
                'Invalid type of data requested. Returning returns data'
            )
            values = 'log_returns'
        if 'index_eod_values_{0}'.format(values) in get_store_keys(NSE.__NSE_DATA_PATH):
            data = pd.read_hdf(
                NSE.__NSE_DATA_PATH, 'index_eod_values_{0}'.format(values)
            )
        else:
            self.force_load_data(force_load='index_eod_values', values=values)
            data = pd.read_hdf(
                NSE.__NSE_DATA_PATH, 'index_eod_values_{0}'.format(values)
            )
        column_list = data.columns
        column_list = data.columns.intersection(index_list)
        data = data[column_list]
        start = get_date(start, 'str', True)
        end = get_date(end, 'str', False)
        data = data[start:end]
        data = data.dropna(how='all', axis=1)
        return data

    def get_eod_meta(self, eod_data=None, eod_type='symbol'):
        'Calculate meta data for EOD Data'

        if eod_data is None:
            if eod_type == 'symbol':
                symbol_meta = self.get_symbol_meta()
                eod_data_meta = pd.DataFrame(
                    index=symbol_meta.index.copy(),
                )
                if NSE.__SYMBOL_EOD_META_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
                    eod_data_meta = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__SYMBOL_EOD_META_KEY)
                    return eod_data_meta
                elif NSE.__SYMBOL_EOD_DATA_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
                    eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__SYMBOL_EOD_DATA_KEY)
                    eod_data = eod_data.reset_index()
                else:
                    eod_data_meta['from_date'] = pd.to_datetime('1994-01-01')
                    eod_data_meta['to_date'] = pd.to_datetime('1994-01-01')
                    eod_data_meta['row_count'] = 0
                    eod_data_meta['missing_count'] = np.inf
                    eod_data_meta['non_traded_dates'] = np.inf
                    eod_data_meta['missing_dates'] = np.nan
                    return eod_data_meta
            elif eod_type == 'index':
                index_meta = self.get_index_meta()
                eod_data_meta = pd.DataFrame(
                    index=index_meta.index.copy(),
                )
                if NSE.__INDEX_EOD_META_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
                    eod_data_meta = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__INDEX_EOD_META_KEY)
                    return eod_data_meta
                elif NSE.__INDEX_EOD_DATA_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
                    eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__INDEX_EOD_DATA_KEY)
                    eod_data = eod_data.reset_index()
                else:
                    eod_data_meta['from_date'] = pd.to_datetime('1994-01-01')
                    eod_data_meta['to_date'] = pd.to_datetime('1994-01-01')
                    eod_data_meta['row_count'] = 0
                    eod_data_meta['missing_count'] = np.inf
                    eod_data_meta['non_traded_dates'] = np.inf
                    eod_data_meta['missing_dates'] = np.nan
                    return eod_data_meta
            else:
                raise KeyError(
                    'Wrong eod_type'
                )
        else:
            if eod_type == 'symbol':
                symbol_meta = self.get_symbol_meta()
                eod_data_meta = pd.DataFrame(
                    index=symbol_meta.index.copy(),
                )
            elif eod_type == 'index':
                index_meta = self.get_index_meta()
                eod_data_meta = pd.DataFrame(
                    index=index_meta.index.copy(),
                )
        def counts(data):
            '''Calculate count data'''
            data = data.set_index('date')
            name = data.symbol.unique()[0]
            count_data = pd.Series(name=name)
            count_data['from_date'] = data.index.min()
            count_data['to_date'] = data.index.max()
            count_data['row_count'] = len(data)

            traded_dates = self.get_traded_dates(
                start=count_data['from_date'],
                end=count_data['to_date']
            )
            missing_dates = traded_dates.index.difference(data.index)
            count_data['missing_count'] = len(traded_dates) - len(data)
            count_data['non_traded_dates'] = len(data.query('volume == 0'))
            count_data['missing_dates'] = missing_dates.tolist()
            return count_data

        count_data = eod_data.groupby('symbol').apply(counts)
        eod_data_meta = eod_data_meta.join(count_data)
        eod_data_meta['from_date'] = eod_data_meta['from_date'].fillna(datetime(1994, 1, 1))
        eod_data_meta['to_date'] = eod_data_meta['to_date'].fillna(datetime(1994, 1, 1))
        eod_data_meta['row_count'] = eod_data_meta['row_count'].fillna(0).astype(int)
        eod_data_meta['missing_count'] = eod_data_meta['missing_count'].fillna(np.inf).astype(np.float)
        eod_data_meta['non_traded_dates'] = eod_data_meta['non_traded_dates'].fillna(np.inf).astype(np.float)

        return eod_data_meta

    def force_load_data(self, force_load, values=None):
        '''
        Force loading helper method for saving EOD data from NSE Website to local HDFStores
        '''

        if force_load == 'symbol_eod_meta':
            print('Updating symbol eod metadata')
            if NSE.__SYMBOL_EOD_DATA_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
                eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__SYMBOL_EOD_DATA_KEY)
                eod_data = eod_data.reset_index()
            else:
                eod_data = None

            eod_data_meta = self.get_eod_meta(eod_data, eod_type='symbol')
            eod_data_meta.to_hdf(NSE.__NSE_DATA_PATH, NSE.__SYMBOL_EOD_META_KEY)

        elif force_load == 'symbol_eod_data':
            eod_data_meta = self.get_eod_meta(eod_type='symbol')
            date_diff = (TODAY - eod_data_meta.to_date).dt.days
            eod_data_meta = eod_data_meta[
                (date_diff >= 5) | (eod_data_meta.row_count == 0)
            ]

            # return if less indices need to be refreshed
            if len(eod_data_meta) < 120:
                print(eod_data_meta.row_count)
                return

            if len(eod_data_meta) > 500:
                eod_data_meta = eod_data_meta.ix[0:200]

            print('Fetching Data from NSE website for {0} symbols'.format(len(eod_data_meta)))

            fresh_eod_data = pd.DataFrame()
            for symbol in eod_data_meta.itertuples():
                eod_data = self.fetch_eod_data(
                    symbol=symbol.Index,
                    start=symbol.to_date,
                    index=False,
                )
                if eod_data.empty:
                    continue
                else:
                    eod_data = eod_data.reset_index()
                    print(
                        'Recieved {0} records from NSE for {1} from {2} to {3}'.
                        format(len(eod_data), symbol.Index,
                               eod_data.date.min().date(),
                               eod_data.date.max().date())
                    )
                    fresh_eod_data = fresh_eod_data.append(eod_data)

            if NSE.__SYMBOL_EOD_DATA_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
                old_eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__SYMBOL_EOD_DATA_KEY)
                old_eod_data = old_eod_data.reset_index()
            else:
                old_eod_data = pd.DataFrame()

            fresh_eod_data = fresh_eod_data.append(old_eod_data)
            del old_eod_data

            fresh_eod_data = fresh_eod_data.drop_duplicates(['symbol', 'date'], keep='last')
            fresh_eod_data = fresh_eod_data.sort_values(['symbol', 'date'])
            eod_data_schema = [
                'symbol', 'date', 'prev_close', 'open', 'high',
                'low', 'last', 'close', 'vwap',
                'trades', 'volume', 'turnover', 'pct_deliverble',
                'simple_returns', 'log_returns', 'high_low_spread', 'open_close_spread'
            ]
            fresh_eod_data = fresh_eod_data.reset_index()[eod_data_schema]
            fresh_eod_data = fresh_eod_data.set_index(['symbol', 'date'])
            fresh_eod_data.to_hdf(NSE.__NSE_DATA_PATH, NSE.__SYMBOL_EOD_DATA_KEY)
            del fresh_eod_data
            self.force_load_data('symbol_eod_meta')
            self.force_load_data('traded_dates')
            eod_data_meta = self.get_eod_meta(eod_type='symbol')
            date_diff = (TODAY - eod_data_meta.to_date).dt.days
            eod_data_meta = eod_data_meta[
                (date_diff >= 5) | (eod_data_meta.row_count == 0)
            ]
            if len(eod_data_meta) > 20:
                self.force_load_data('symbol_eod_data')
            eod_data_columns = [
                'open', 'high', 'low', 'close', 'vwap',
                'simple_returns', 'log_returns',
                'high_low_spread', 'open_close_spread'
            ]
            for column in eod_data_columns:
                self.force_load_data(force_load='symbol_eod_values', values=column)
            # clean_file(NSE.__NSE_DATA_PATH)

        elif force_load == 'symbol_eod_values':
            print('Generating time series data for {0} from local data'.format(values))
            eod_data = self.get_symbol_eod_data(symbol_list='all')

            data = pd.pivot_table(data=eod_data, index='date',
                                  columns='symbol', values=values)
            data.to_hdf(NSE.__NSE_DATA_PATH, 'symbol_eod_values_{0}'.format(values))

        elif force_load == 'index_eod_meta':
            print('Updating index eod metadata')
            if NSE.__INDEX_EOD_DATA_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
                eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__INDEX_EOD_DATA_KEY)
                eod_data = eod_data.reset_index()
            else:
                eod_data = None

            eod_data_meta = self.get_eod_meta(eod_data, eod_type='index')
            eod_data_meta.to_hdf(NSE.__NSE_DATA_PATH, NSE.__INDEX_EOD_META_KEY)

        elif force_load == 'index_eod_data':
            eod_data_meta = self.get_eod_meta(eod_type='index')
            index_meta = self.get_index_meta()
            eod_data_meta = eod_data_meta.join(index_meta.index_code)
            date_diff = (TODAY - eod_data_meta.to_date).dt.days
            eod_data_meta = eod_data_meta[
                (date_diff >= 5) | (eod_data_meta.row_count == 0)
            ]
            eod_data_meta = eod_data_meta.dropna(subset=['index_code'])

            # return if less indices need to be refreshed
            if len(eod_data_meta) < 20:
                return

            print('Fetching Data from NSE website for {0} indices'.format(len(eod_data_meta)))
            fresh_eod_data = pd.DataFrame()
            for index in eod_data_meta.itertuples():
                eod_data = self.fetch_eod_data(
                    symbol=index.index_code,
                    start=index.to_date,
                    index=True,
                )
                if eod_data.empty:
                    continue
                else:
                    eod_data = eod_data.reset_index()
                    eod_data['symbol'] = [index.Index for i in range(len(eod_data))]
                    print(
                        'Recieved {0} records from NSE for {1} from {2} to {3}'.
                        format(len(eod_data), index.Index,
                               eod_data.date.min().date(),
                               eod_data.date.max().date())
                    )
                    fresh_eod_data = fresh_eod_data.append(eod_data)
            if fresh_eod_data.empty:
                return

            if NSE.__INDEX_EOD_DATA_KEY in get_store_keys(NSE.__NSE_DATA_PATH):
                old_eod_data = pd.read_hdf(NSE.__NSE_DATA_PATH, NSE.__INDEX_EOD_DATA_KEY)
                old_eod_data = old_eod_data.reset_index()
            else:
                old_eod_data = pd.DataFrame()

            fresh_eod_data = fresh_eod_data.append(old_eod_data)
            del old_eod_data

            fresh_eod_data = fresh_eod_data.drop_duplicates(['symbol', 'date'], keep='last')
            fresh_eod_data = fresh_eod_data.sort_values(['symbol', 'date'])
            eod_data_schema = [
                'symbol', 'date', 'open', 'high',
                'low', 'close', 'volume', 'turnover',
                'simple_returns', 'log_returns', 'high_low_spread', 'open_close_spread'
            ]
            fresh_eod_data = fresh_eod_data.reset_index()[eod_data_schema]
            fresh_eod_data = fresh_eod_data.set_index(['symbol', 'date'])
            fresh_eod_data.to_hdf(NSE.__NSE_DATA_PATH, NSE.__INDEX_EOD_DATA_KEY)
            del fresh_eod_data
            self.force_load_data('index_eod_meta')
            self.force_load_data('traded_dates')
            eod_data_meta = self.get_eod_meta(eod_type='index')
            date_diff = (TODAY - eod_data_meta.to_date).dt.days
            eod_data_meta = eod_data_meta[
                (date_diff >= 5) | (eod_data_meta.row_count == 0)
            ]
            if len(eod_data_meta) > 5:
                self.force_load_data('index_eod_data')
            eod_data_columns = [
                'open', 'high', 'low', 'close',
                'simple_returns', 'log_returns',
                'high_low_spread', 'open_close_spread'
            ]
            for column in eod_data_columns:
                self.force_load_data(force_load='index_eod_values', values=column)
            clean_file(NSE.__NSE_DATA_PATH)

        elif force_load == 'index_eod_values':
            print('Generating time series data for {0} from local data'.format(values))
            eod_data = self.get_index_eod_data(index_list='all')

            data = pd.pivot_table(data=eod_data, index='date',
                                  columns='symbol', values=values)
            data.to_hdf(NSE.__NSE_DATA_PATH, 'index_eod_values_{0}'.format(values))


        elif force_load == 'all':
            self.force_load_data('traded_dates')
            self.force_load_data('symbol_eod_data')
            self.force_load_data('index_eod_data')
        else:
            super().force_load_data(force_load)
    def __init__(
            self, symbol_list=None, index=None, index_type=None,
            start=None, end=None,
            min_rows=0, missing_count=0,
            force_load=None,
        ):
        super().__init__(
            symbol_list=symbol_list, index=index, index_type=index_type,
            start=start, end=end, force_load=force_load
        )
        if force_load is not None:
            self.force_load_data(force_load)

        self.symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index, index_type=index_type,
            start=start, min_rows=min_rows, missing_count=missing_count
        )
