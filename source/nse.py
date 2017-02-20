'''Module for loading the symbol metadata'''  # noqa
import re
import os
import warnings
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
# from dateutil.relativedelta import relativedelta
from nsepy import get_history

from helpers import get_daily_volatility, get_date, rename_columns

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())


class NSE(object):
    '''
    Module for NSE Data
    '''

    CURRENT_PATH = os.path.dirname(__file__)
    SYMBOL_DATA_PATH = os.path.join(CURRENT_PATH, 'data/symbol_data.h5')

    INDEX_META_KEY = 'index_meta'
    SYMBOL_META_KEY = 'symbol_meta'
    EOD_DATA_KEY = 'eod_data'
    EOD_DATA_META_KEY = 'eod_data_meta'

    def fetch_symbol_meta(self):
        '''
        Method to grab symbol meta data from NSE website
        '''
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

    def fetch_eod_data(self, symbol, start=None):
        '''Fetch all End of Day(EOD) data from NSE'''
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
        eod_data = eod_data.set_index(['symbol', 'date'])
        eod_data['high_low_spread'] = (eod_data.high - eod_data.low) / eod_data.low
        eod_data['open_close_spread'] = (eod_data.close - eod_data.open) / eod_data.open
        eod_data['simple_returns'] = (
            (eod_data.close - eod_data.prev_close) / eod_data.prev_close
        )
        eod_data['log_returns'] = np.log(eod_data.close / eod_data.prev_close)
        eod_data['daily_volatility'] = get_daily_volatility(eod_data.log_returns)

        # Adjusting other columns for maintaining integrity
        eod_data['pct_deliverble'] = eod_data['pct_deliverble'] * 100
        eod_data = eod_data.astype(np.float)
        return eod_data

    def get_symbol_meta(self):
        '''
        If symbol meta data exists grab data from file.
        Else fetch symbol meta data from NSE website.
        '''
        try:
            symbol_meta = pd.read_hdf(NSE.SYMBOL_DATA_PATH, NSE.SYMBOL_META_KEY)
        except:
            warnings.warn(
                'Unable to read symbol_meta locally. Fetching data from NSE website'
            )
            self.force_load_data(force_load='symbol_meta')
            symbol_meta = pd.read_hdf(NSE.SYMBOL_DATA_PATH, NSE.SYMBOL_META_KEY)
        return symbol_meta

    @classmethod
    def get_index_components(
            cls, index=None, index_type=None, fetch_type='union'
        ):
        'Retruns the symbols components of passed index or index_type'
        index_meta = pd.read_hdf(NSE.SYMBOL_DATA_PATH, NSE.INDEX_META_KEY)
        index_meta = index_meta.replace('nan', np.nan)
        index_type_list = index_meta.index_type.unique()

        if index is not None:
            if isinstance(index_type, str):
                index_list = [index]
            if isinstance(index_type, list):
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
            temp_index_components = pd.read_hdf(NSE.SYMBOL_DATA_PATH, hdf_key)
            temp_index_components = pd.DataFrame(temp_index_components)
            if index_components.empty:
                index_components = temp_index_components
            else:
                index_components = index_components.join(temp_index_components)

        # list to query
        if fetch_type == 'union':
            query = ' | '.join(index_list)
        elif fetch_type == 'intersection':
            query = ' & '.join(index_list)
        symbol_list = index_components.query(query)
        symbol_list = symbol_list.index.tolist()
        return symbol_list

    def get_symbol_list(
            self, symbol_list=None,
            index=None, index_type=None, start=None
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

        # if min_rows is not None:
        #     symbol_data_meta = self.get_symbol_data_meta()
        #     symbol_list = symbol_list[symbol_data_meta.row_count >= min_rows]

        # if null_count is not None:
        #     symbol_data_meta = self.get_symbol_data_meta()
        #     symbol_list = symbol_list[symbol_data_meta.missing_count >= null_count]

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
        # return symbol_list

    def get_traded_dates(self):
        'Generate Traded dates for NSE'
        try:
            eod_data = pd.read_hdf(NSE.SYMBOL_DATA_PATH, NSE.EOD_DATA_KEY)
        except (FileNotFoundError, KeyError) as data_not_found:
            print(data_not_found)

        return eod_data

    def get_eod_data_meta(self, eod_data=None):
        'Generate metadata for EOD Data'
        symbol_meta = self.get_symbol_meta()
        eod_data_meta = pd.DataFrame(index=symbol_meta.index.copy())

        if eod_data is None:
            try:
                eod_data = pd.read_hdf(NSE.SYMBOL_DATA_PATH, NSE.EOD_DATA_KEY)
                eod_data = eod_data.reset_index()
            except:
                eod_data_meta['from_date'] = pd.to_datetime('1994-01-01')
                eod_data_meta['to_date'] = pd.to_datetime('1994-01-01')
                eod_data_meta['row_count'] = 0
                eod_data_meta['missing_count'] = np.inf
                return eod_data_meta

        traded_dates = self.get_traded_dates()

        def counts(data):
            data = data.set_index('date')
            name = data.symbol.unique()[0]
            count_data = pd.Series(name=name)
            count_data['from_date'] = data.date.min()
            count_data['to_date'] = data.date.max()
            count_data['row_count'] = len(data)

            stock_specific_traded_dates = traded_dates.index.union(data.index)

            return count_data
        count_data = eod_data.groupby('symbol').apply(counts)
        count_data = count_data.reset_index(1, drop=True)
        symbol_meta['from_date'] = count_data['from_date']
        symbol_meta['to_date'] = count_data['to_date']
        symbol_meta['row_count'] = count_data['row_count']
        symbol_meta['from_date'] = symbol_meta['from_date'].fillna(datetime(1994, 1, 1))
        symbol_meta['to_date'] = symbol_meta['to_date'].fillna(datetime(1994, 1, 1))
        symbol_meta['row_count'] = symbol_meta['row_count'].fillna(0).astype(int)
        symbol_meta.to_hdf(NSE.SYMBOL_DATA_PATH, 'symbol_meta')

    def force_load_data(self, force_load, values=None):

        if force_load == 'symbol_meta':
            print('Loading Symbol Meta data from NSE website')
            symbol_meta = self.fetch_symbol_meta()
            if not os.path.isdir(os.path.join(NSE.CURRENT_PATH, 'data')):
                os.mkdir(os.path.join(NSE.CURRENT_PATH, 'data'))
            symbol_meta.to_hdf(NSE.SYMBOL_DATA_PATH, NSE.SYMBOL_META_KEY)
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
                index_components_data.to_hdf(NSE.SYMBOL_DATA_PATH, hdf_key)
            symbol_meta['name_of_company'] = symbol_meta['name_of_company'].astype(str)
            symbol_meta['isin_number'] = symbol_meta['isin_number'].astype(str)
            symbol_meta['industry'] = symbol_meta['industry'].fillna('unknown')
            symbol_meta['industry'] = symbol_meta['industry'].str.lower().str.replace(' ', '_')
            symbol_meta.to_hdf(NSE.SYMBOL_DATA_PATH, NSE.SYMBOL_META_KEY)
            index_meta = index_meta.astype(str)
            index_meta.to_hdf(NSE.SYMBOL_DATA_PATH, NSE.INDEX_META_KEY)

        elif force_load == 'historical_data':
            # self.force_load_data(force_load='symbol_data_meta')
            # symbol_meta = self.get_historical_data_meta()
            # date_diff = (TODAY - symbol_meta.to_date).dt.days
            # symbol_meta = symbol_meta[
            #     (date_diff >= 5) | (symbol_meta.row_count == 0)
            # ]
            # print('Fetching Data from NSE website for {0} symbols'.format(len(symbol_meta)))
            # try:
            #     os.remove(TEMP_DATA_PATH)
            # except:
            #     pass
            # # symbol_list = symbol_list[symbol_list.index.isin(['arvinfra'])]
            # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            #     for symbol in symbol_meta.itertuples():
            #         symbol_executor = executor.submit(
            #             self.fetch_historical_data,
            #             symbol=symbol.Index,
            #             start=symbol.to_date
            #         )
            #         nse_data = symbol_executor.result()
            #         if nse_data.empty:
            #             continue
            #         else:
            #             print(
            #                 'Recieved {0} records from NSE for {1} from {2} to {3}'.
            #                 format(len(nse_data), symbol.Index,
            #                        nse_data.date.min().date(),
            #                        nse_data.date.max().date())
            #             )
            #         # nse_data = nse_data[nse_data.date >= symbol.date_of_listing]
            #         with SafeHDFStore(TEMP_DATA_PATH) as store:
            #             store.put('symbol_hist_data_temp', value=nse_data, format='t',
            #                       append=True, min_itemsize={'symbol': 15})

            # hist_data_temp = pd.read_hdf(TEMP_DATA_PATH, 'symbol_hist_data_temp')
            # try:
            #     hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_hist_data')
            #     hist_data = hist_data.reset_index()
            # except:
            #     hist_data = pd.DataFrame(columns=hist_data_temp.columns)

            # hist_data = hist_data.append(hist_data_temp)
            # del hist_data_temp
            # os.remove(TEMP_DATA_PATH)

            # hist_data = hist_data.drop_duplicates(['symbol', 'date'], keep='last')
            # hist_data = hist_data.sort_values(['symbol', 'date'])
            # hist_symbol_schema = [
            #     'symbol', 'date', 'prev_close', 'open', 'high',
            #     'low', 'last', 'close', 'vwap',
            #     'simple_returns', 'log_returns', 'daily_volatility',
            #     'trades', 'volume', 'turnover', 'pct_deliverble'
            # ]
            # hist_data = hist_data.reset_index()[hist_symbol_schema]
            # hist_data = hist_data.set_index(['symbol', 'date'])
            # hist_data.to_hdf(SYMBOL_DATA_PATH, 'symbol_hist_data')
            # self.update_symbol_data_meta()
            # hist_data_columns = [
            #     'open', 'high', 'low', 'last', 'close', 'vwap',
            #     'simple_returns', 'log_returns', 'daily_volatility',
            #     'volume', 'turnover', 'pct_deliverble'
            # ]
            # for col in hist_data_columns:
            #     self.force_load_data(force_load='symbol_data', values=col)
            # clean_file(SYMBOL_DATA_PATH)
            pass

        elif force_load == 'symbol_data_meta':
            # TODO update this block like IIFL and modify update_symbol_data_meta similarly
            symbol_data_meta = self.update_symbol_data_meta()
            symbol_data_meta.to_hdf(NSE.SYMBOL_DATA_PATH, NSE.EOD_DATA_META_KEY)

        elif force_load == 'symbol_data':
            pass
            # print('Generating time series data for {0} from local data'.format(values))
            # symbol_meta = self.get_symbol_meta()
            # symbol_list = symbol_meta.index.tolist()
            # hist_data = self.get_symbol_hist(symbol_list=symbol_list)

            # data = pd.pivot_table(data=hist_data, index='date',
            #                       columns='symbol', values=values)
            # data.to_hdf(SYMBOL_DATA_PATH, 'symbol_data_{0}'.format(values))

        elif force_load == 'all':
            self.force_load_data('symbol_meta')
            self.force_load_data('index_components')
            self.force_load_data('historical_data')
