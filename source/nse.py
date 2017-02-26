'''Module for loading the symbol metadata'''  # noqa
import os
import re
import warnings
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
# from dateutil.relativedelta import relativedelta
from nsepy import get_history

from helpers import get_date, rename_columns, get_store_keys, clean_file

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())


class NSE(object):
    '''
    Module for NSE Data
    '''

    CURRENT_PATH = os.path.dirname(__file__)
    NSE_DATA_PATH = os.path.join(CURRENT_PATH, 'data{0}symbol_data.h5'.format(os.sep))
    CONSTANTS_PATH = os.path.join(CURRENT_PATH, 'data{0}constants.h5'.format(os.sep))

    INDEX_META_KEY = 'index_meta'
    SYMBOL_META_KEY = 'symbol_meta'
    EOD_DATA_KEY = 'eod_data'
    EOD_DATA_META_KEY = 'eod_data_meta'
    TRADED_DATES_KEY = 'traded_dates'

    @classmethod
    def fetch_symbol_meta(cls):
        'Function to grab symbol meta data from NSE website'

        symbol_list_url = 'https://www.nseindia.com/content/equities/EQUITY_L.csv'
        response = requests.get(symbol_list_url)
        symbol_meta = pd.read_csv(StringIO(response.text), index_col='SYMBOL')

        rename_columns(symbol_meta)
        symbol_meta = symbol_meta.query('series == "EQ"')
        symbol_meta = symbol_meta[[
            'name_of_company', 'isin_number', 'date_of_listing'
        ]]
        symbol_meta.date_of_listing = pd.to_datetime(symbol_meta.date_of_listing)
        symbol_meta.index = symbol_meta.index.str.lower()
        symbol_meta = symbol_meta.sort_index()
        symbol_meta['name_of_company'] = symbol_meta['name_of_company'].astype(str)
        symbol_meta['isin_number'] = symbol_meta['isin_number'].astype(str)

        return symbol_meta

    def fetch_eod_data(
            self, symbol, start=None
        ):
        'Fetch all End of Day(EOD) data from NSE'
        from_date = get_date(start, start=True)
        to_date = get_date(start=False)

        # Get data from NSE
        try:
            eod_data = get_history(
                symbol=symbol, start=from_date,
                end=to_date, series='EQ'
            )
            if eod_data.empty:
                warnings.warn(
                    'No data recieved from NSE for {0} from {1} to {2}'.
                    format(symbol, from_date.date(), to_date.date())
                )
                return eod_data
            eod_data.drop(
                ['Series', 'Deliverable Volume'], 1, inplace=True
            )
            eod_data.index = pd.to_datetime(eod_data.index)
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
        eod_data = eod_data.reset_index().set_index(['symbol', 'date'])
        eod_data = eod_data.astype(np.float)
        return eod_data

    def get_symbol_meta(self):
        '''
        If symbol meta data exists grab data from file.
        Else fetch symbol meta data from NSE website.
        '''
        if NSE.SYMBOL_META_KEY in get_store_keys(NSE.NSE_DATA_PATH):
            symbol_meta = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.SYMBOL_META_KEY)
        else:
            warnings.warn(
                'Unable to read symbol_meta locally. Fetching data from NSE website'
            )
            self.force_load_data(force_load='symbol_meta')
            symbol_meta = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.SYMBOL_META_KEY)
        return symbol_meta

    def get_index_components(
            self, index=None, index_type=None
        ):
        'Returns the symbols components of passed index or index_type'

        if NSE.INDEX_META_KEY in get_store_keys(NSE.NSE_DATA_PATH):
            index_meta = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.INDEX_META_KEY)
        else:
            warnings.warn(
                'Unable to read symbol_meta locally. Fetching data from NSE website'
            )
            self.force_load_data(force_load='index_components')
            index_meta = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.INDEX_META_KEY)

        index_meta = index_meta.replace('nan', np.nan)
        index_meta = index_meta.dropna()
        index_type_list = index_meta.index_type.unique()

        if index is not None:
            if isinstance(index, str) and index == 'all':
                index_list = index_meta.index.tolist()
            elif isinstance(index, str):
                if index in index_meta.index:
                    index_list = [index]
                else:
                    warnings.warn(
                        'Could not find {0} index. Loading NIFTY 50'.format(index)
                    )
                    index_list = ['nifty_50']
            elif isinstance(index, list):
                index_list = index
        elif index_type is not None:
            if isinstance(index_type, str) and index_type == 'all':
                pass
            elif isinstance(index_type, str):
                index_meta = index_meta.query(
                    'index_type == "{0}"'.format(index_type)
                )
            elif isinstance(index_type, list):
                index_meta = index_meta[index_meta.index_type.isin(index_type)]
            index_list = index_meta.dropna().index.tolist()

        else:
            warnings.warn(
                'No suitable index found. Loading Nifty 50 Index components'
            )
            index_list = ['nifty_50']

        index_components = pd.DataFrame()
        for index_type in index_type_list:
            hdf_key = index_type + '_components'
            temp_index_components = pd.read_hdf(NSE.NSE_DATA_PATH, hdf_key)
            temp_index_components = pd.DataFrame(temp_index_components)
            if index_components.empty:
                index_components = temp_index_components
            else:
                index_components = index_components.join(temp_index_components)

        index_components = index_components[index_list].replace(False, np.nan)
        index_components = index_components.dropna(how='all')
        symbol_list = index_components.index.tolist()
        return symbol_list

    def get_symbol_list(
            self, symbol_list=None, index=None, index_type=None, start=None,
            min_rows=None, missing_count=None
        ):
        '''
        Get symbol list based on criteria provided.
        Pass index for getting symbols in index.
        List of indexes to get union of symbols of all indexes in list.
        start: {year as int or string, string format of date, None}
        null_count: {True, False} load symbols listed before start date
        '''
        symbol_meta = self.get_symbol_meta()
        if symbol_list is None:
            try:
                symbol_list = self.symbol_list
            except:
                symbol_list = symbol_meta.date_of_listing
        elif isinstance(symbol_list, str) and symbol_list == 'all':
            symbol_list = symbol_meta.date_of_listing
            return symbol_list
        elif isinstance(symbol_list, str):
            symbol_list = symbol_meta[symbol_meta.index == symbol_list].date_of_listing
        elif isinstance(symbol_list, list):
            symbol_list = symbol_meta[symbol_meta.index.isin(symbol_list)].date_of_listing
        elif isinstance(symbol_list, pd.Series):
            symbol_list = symbol_list.copy()
        elif isinstance(symbol_list, pd.DataFrame):
            try:
                symbol_list = symbol_list.date_of_listing.copy()
            except:
                warnings.warn('Could not fetch symbol_list with proper dates.\
                               Loading default symbol_list')
                symbol_list = symbol_meta.date_of_listing.copy()
        else:
            warnings.warn('Could not fetch symbol_list with proper dates.\
                           Loading default symbol_list')
            symbol_list = symbol_meta.date_of_listing.copy()

        symbol_list = symbol_list.copy()
        if index is not None or index_type is not None:
            symbol_list_temp = self.get_index_components(
                index=index, index_type=index_type
            )
            symbol_list = symbol_list[symbol_list.index.isin(symbol_list_temp.index)]

        if start is not None:
            start = get_date(start, 'dt')
            symbol_list = symbol_list[symbol_list <= start]

        if min_rows is not None:
            symbol_data_meta = self.get_symbol_data_meta()
            symbol_list = symbol_list[symbol_data_meta.row_count >= min_rows]

        if missing_count is not None:
            symbol_data_meta = self.get_symbol_data_meta()
            symbol_list = symbol_list[symbol_data_meta.missing_count >= missing_count]

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

    def get_eod_data(
            self, symbol_list=None,
            index=None, index_type=None, start=None, end=None,
            min_rows=None, missing_count=None
        ):
        '''
        If SYMBOL_DATA_PATH exists grab data from file.
        Update data if data in the file is older than 5 days.
        Else fetch symbol data from NSE website.
        '''
        if NSE.EOD_DATA_KEY in get_store_keys(NSE.NSE_DATA_PATH):
            eod_data = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.EOD_DATA_KEY)
            eod_data = eod_data.reset_index()
        else:
            self.force_load_data(force_load='eod_data')
            eod_data = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.EOD_DATA_KEY)
            eod_data = eod_data.reset_index()

        symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index, index_type=index_type,
            start=start, missing_count=missing_count, min_rows=min_rows
        )

        eod_data = eod_data[eod_data.symbol.isin(symbol_list.index)]
        start = get_date(start, out='dt', start=True)
        end = get_date(end, out='dt', start=False)
        eod_data = eod_data.ix[
            (eod_data.date >= start) & (eod_data.date <= end)
        ]
        return eod_data

    def get_eod_column_data(
            self, data='returns', symbol_list=None,
            index=None, index_type=None, start=None, end=None,
            min_rows=None, missing_count=None
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
        try:
            data = pd.read_hdf(
                NSE.NSE_DATA_PATH, 'eod_column_data_{0}'.format(values)
            )
        except:
            self.force_load_data(force_load='eod_column_data', values=values)
            data = pd.read_hdf(
                NSE.NSE_DATA_PATH, 'eod_column_data_{0}'.format(values)
            )
        column_list = data.columns
        column_list = data.columns.intersection(symbol_list.index)
        data = data[column_list]
        start = get_date(start, 'str', True)
        end = get_date(end, 'str', False)
        data = data[start:end]
        data = data.dropna(how='all', axis=1)
        return data

    def get_traded_dates(self, start=None, end=None):
        'Generate Traded dates for NSE'

        if NSE.TRADED_DATES_KEY in get_store_keys(NSE.NSE_DATA_PATH):
            traded_dates = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.TRADED_DATES_KEY)
        else:
            self.force_load_data('traded_dates')
            traded_dates = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.TRADED_DATES_KEY)

        start = get_date(start, 'str', True)
        end = get_date(end, 'str', False)

        traded_dates = traded_dates[start:end]
        traded_dates['specific_date_count'] = [i+1 for i in range(len(traded_dates))]

        return traded_dates

    def get_eod_data_meta(self, eod_data=None):
        'Calculate meta data for EOD Data'
        symbol_meta = self.get_symbol_meta()
        eod_data_meta = pd.DataFrame(
            index=symbol_meta.index.copy(),
        )

        if eod_data is None:
            if NSE.EOD_DATA_META_KEY in get_store_keys(NSE.NSE_DATA_PATH):
                eod_data_meta = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.EOD_DATA_META_KEY)
                return eod_data_meta
            elif NSE.EOD_DATA_KEY in get_store_keys(NSE.NSE_DATA_PATH):
                eod_data = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.EOD_DATA_KEY)
                eod_data = eod_data.reset_index()
            else:
                eod_data_meta['from_date'] = pd.to_datetime('1994-01-01')
                eod_data_meta['to_date'] = pd.to_datetime('1994-01-01')
                eod_data_meta['row_count'] = 0
                eod_data_meta['missing_count'] = np.inf
                eod_data_meta['missing_dates'] = np.nan
                return eod_data_meta

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
        Force loading helper method for saving data from NSE Website to local HDFStores
        '''

        if force_load == 'symbol_meta':
            print('Loading Symbol Meta data from NSE website')
            symbol_meta = self.fetch_symbol_meta()
            if not os.path.isdir(os.path.join(NSE.CURRENT_PATH, 'data')):
                os.mkdir(os.path.join(NSE.CURRENT_PATH, 'data'))
            symbol_meta.to_hdf(NSE.NSE_DATA_PATH, NSE.SYMBOL_META_KEY)
            self.force_load_data('index_components')

        elif force_load == 'index_components':
            print('Loading Index components data from NSE website')
            symbol_meta = self.get_symbol_meta()
            symbol_meta['industry'] = np.nan
            session = requests.session()
            url = 'https://www.nseindia.com/products/content/equities/indices/historical_index_data.htm'
            response = session.get(url)
            if response.status_code != 200:
                print(
                    'Unable to load base url data due to {0} status code'
                    .format(response.status_code)
                )
                return
            soup = BeautifulSoup(response.text, 'html.parser')
            index_meta = pd.DataFrame(
                columns=['index_code', 'index_name', 'index_type', 'url', 'number_of_symbols']
            )
            index_type_group = soup.find('select', {'id': 'indexType'})
            index_type_group = index_type_group.find_all('optgroup')
            for index_type in index_type_group:
                index_type_label = index_type['label'].strip()
                index_type_label = index_type_label.split(' ')
                index_type_label = '_'.join(index_type_label[0:-1]).lower()
                index_type_label = index_type_label.replace('strategy', 'strategic')
                index_list = index_type.find_all('option')
                for index in index_list:
                    index_code = index['value'].strip()
                    index_code_asindex = index_code.lower().replace(' ', '_').replace('%', '')
                    index_name = index.text.strip()
                    index_meta.loc[index_code_asindex] = [
                        index_code, index_name, index_type_label, np.nan, np.nan
                    ]
                if index_type_label != 'broad_market':
                    index_components_data = pd.DataFrame(index=symbol_meta.index)
                    info_url = 'https://www.nseindia.com/products/content/equities/indices/{0}_indices.htm'
                    response = session.get(info_url.format(index_type_label))
                    if response.status_code != 200:
                        print(
                            'Unable to load url data for {0} index type due to {1} status code'
                            .format(index_type_label, response.status_code)
                        )
                        continue
                    soup = BeautifulSoup(response.text, 'html.parser')
                    content = soup.find('div', {'class': 'abt_equities_content'})
                    download_links = content.find_all('a', {'class': 'download'})
                    for link in download_links:
                        text = link.text
                        text = re.sub(r'\r\n', ' ', text)
                        text = re.sub(' +', ' ', text)
                        text = text[text.find('NIFTY'): text.find('Index')]
                        if text[-3:] == 'csv':
                            text = text[text.find('NIFTY'): text.find('stocks')]
                        if text[-2:] == 'cs':
                            text = text[text.find('NIFTY'): text.find('Indices')]
                        text = text.lower().strip()
                        link = link['href']
                        link = 'https://www.nseindia.com' + link
                        if link[-3:] == 'csv':
                            try:
                                index = index_meta[index_meta.index_name.str.lower() == text].index[0]
                            except:
                                warnings.warn(
                                    '{0} index not found in index_meta table'.format(text)
                                )
                                continue
                            response = session.get(link)
                            if response.status_code != 200:
                                print(
                                    'Unable to fetch csv data for {0} index due to {1} status code'
                                    .format(index, response.status_code)
                                )
                                continue
                            index_components = pd.read_csv(StringIO(response.text), index_col='Symbol')
                            index_components.index = index_components.index.str.lower()
                            symbol_meta['industry'] = symbol_meta['industry'].fillna(index_components['Industry'])
                            index_meta.loc[index, 'url'] = link
                            index_meta.loc[index, 'number_of_symbols'] = len(index_components)
                            index_components = pd.Series(True, index=index_components.index, name=index)
                            index_components_data = index_components_data.join(index_components)
                            print(
                                'Component data loaded successfully for {0}'.format(index)
                            )
                elif index_type_label == 'broad_market':
                    index_components_data = pd.DataFrame(index=symbol_meta.index)
                    info_url = 'https://www.nseindia.com/products/content/equities/indices/broad_indices.htm'
                    response = session.get(info_url.format(index_type_label))
                    soup = BeautifulSoup(response.text, 'html.parser')
                    content = soup.find('div', {'class': 'content'})
                    download_links = content.find_all('a')
                    for link in download_links:
                        text = link.text
                        text = re.sub(r'\r\n', ' ', text)
                        text = re.sub(' +', ' ', text)
                        text = text[text.find('NIFTY'): text.find('Index')]
                        if text[-3:] == 'csv':
                            text = text[text.find('NIFTY'): text.find('stocks')]
                        if text[-2:] == 'cs':
                            text = text[text.find('NIFTY'): text.find('Indices')]
                        text = text.lower().strip()
                        link = link['href']
                        link = 'https://www.nseindia.com/products/content/equities/indices/' + link
                        try:
                            index = index_meta[index_meta.index_name.str.lower() == text].index[0]
                        except:
                            warnings.warn(
                                '{0} index not found in index_meta table'.format(text)
                            )
                            continue
                        response = session.get(link)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        link_list = soup.find_all('a', {'class': 'download'})
                        for link in link_list:
                            link = link['href']
                            if link[-3:] == 'csv':
                                csv_link = 'https://www.nseindia.com' + link
                                break
                        response = session.get(csv_link)
                        if response.status_code != 200:
                            print(
                                'Unable to fetch csv data for {0} index due to {1} status code'
                                .format(index, response.status_code)
                            )
                            continue
                        index_components = pd.read_csv(StringIO(response.text), index_col='Symbol')
                        index_components.index = index_components.index.str.lower()
                        symbol_meta['industry'] = symbol_meta['industry'].fillna(index_components['Industry'])
                        index_meta.loc[index, 'url'] = link
                        index_meta.loc[index, 'number_of_symbols'] = len(index_components)
                        index_components = pd.Series(True, index=index_components.index, name=index)
                        index_components_data = index_components_data.join(index_components)
                        print(
                            'Component data loaded successfully for {0}'.format(index)
                        )
                index_components_data = index_components_data.fillna(False).astype(bool)
                hdf_key = index_type_label + '_components'
                index_components_data.to_hdf(NSE.NSE_DATA_PATH, hdf_key)
            symbol_meta['name_of_company'] = symbol_meta['name_of_company'].astype(str)
            symbol_meta['isin_number'] = symbol_meta['isin_number'].astype(str)
            symbol_meta['industry'] = symbol_meta['industry'].fillna('unknown')
            symbol_meta['industry'] = symbol_meta['industry'].str.lower().str.replace(' ', '_')
            symbol_meta.to_hdf(NSE.NSE_DATA_PATH, NSE.SYMBOL_META_KEY)
            index_meta = index_meta.astype(str)
            index_meta.to_hdf(NSE.NSE_DATA_PATH, NSE.INDEX_META_KEY)

        elif force_load == 'traded_dates':
            if NSE.EOD_DATA_KEY in get_store_keys(NSE.NSE_DATA_PATH):
                eod_data = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.EOD_DATA_KEY)
                eod_data = eod_data.reset_index()
                traded_dates = eod_data.date.unique()
                traded_dates = pd.to_datetime(traded_dates)
            else:
                traded_dates = pd.read_hdf(NSE.CONSTANTS_PATH, NSE.TRADED_DATES_KEY)
                traded_dates = traded_dates.index

            traded_dates = pd.read_hdf(NSE.CONSTANTS_PATH, NSE.TRADED_DATES_KEY)
            traded_dates = pd.DataFrame(index=traded_dates.index)
            traded_dates['date'] = traded_dates.index
            traded_dates['date_count'] = [i+1 for i in range(len(traded_dates))]
            traded_dates['day'] = traded_dates.date.dt.day
            traded_dates['month'] = traded_dates.date.dt.month
            traded_dates['year'] = traded_dates.date.dt.year
            traded_dates['day_of_week'] = traded_dates.date.dt.dayofweek
            traded_dates['day_of_year'] = traded_dates.date.dt.dayofyear
            traded_dates['week_of_year'] = traded_dates.date.dt.week
            traded_dates = traded_dates.drop(['date'], axis=1)
            traded_dates.to_hdf(NSE.NSE_DATA_PATH, NSE.TRADED_DATES_KEY)

        elif force_load == 'eod_data_meta':
            if NSE.EOD_DATA_KEY in get_store_keys(NSE.NSE_DATA_PATH):
                eod_data = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.EOD_DATA_KEY)
                eod_data = eod_data.reset_index()
            else:
                eod_data = None

            eod_data_meta = self.get_eod_data_meta(eod_data)
            eod_data_meta.to_hdf(NSE.NSE_DATA_PATH, NSE.EOD_DATA_META_KEY)

        elif force_load == 'eod_data':
            eod_data_meta = self.get_eod_data_meta()
            date_diff = (TODAY - eod_data_meta.to_date).dt.days
            eod_data_meta = eod_data_meta[
                (date_diff >= 5) | (eod_data_meta.row_count == 0)
            ]
            print('Fetching Data from NSE website for {0} symbols'.format(len(eod_data_meta)))
            # eod_data_meta = eod_data_meta[eod_data_meta.index.isin(['arvinfra', 'infy', 'hindalco'])]

            fresh_eod_data = pd.DataFrame()
            for symbol in eod_data_meta.itertuples():
                eod_data = self.fetch_eod_data(
                    symbol=symbol.Index,
                    start=symbol.to_date
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

            if NSE.EOD_DATA_KEY in get_store_keys(NSE.NSE_DATA_PATH):
                old_eod_data = pd.read_hdf(NSE.NSE_DATA_PATH, NSE.EOD_DATA_KEY)
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
            fresh_eod_data.to_hdf(NSE.NSE_DATA_PATH, NSE.EOD_DATA_KEY)
            del fresh_eod_data
            self.force_load_data('eod_data_meta')
            eod_data_columns = [
                'open', 'high', 'low', 'close', 'vwap',
                'simple_returns', 'log_returns',
                'high_low_spread', 'open_close_spread'
            ]
            for column in eod_data_columns:
                self.force_load_data(force_load='eod_column_data', values=column)
            clean_file(NSE.NSE_DATA_PATH)

        elif force_load == 'eod_column_data':
            print('Generating time series data for {0} from local data'.format(values))
            eod_data = self.get_eod_data(symbol_list='all')

            data = pd.pivot_table(data=eod_data, index='date',
                                  columns='symbol', values=values)
            data.to_hdf(NSE.NSE_DATA_PATH, 'eod_column_data_{0}'.format(values))

        elif force_load == 'all':
            self.force_load_data('symbol_meta')
            self.force_load_data('traded_dates')
            self.force_load_data('eod_data')
