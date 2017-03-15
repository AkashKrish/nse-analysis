'''Module for loading symbol meta data from NSE website'''
import os
import re
import warnings
from io import StringIO
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import requests
from bs4 import BeautifulSoup

from helpers import get_date, rename_columns, get_store_keys

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())

class Market(object):
    '''Symbol module to hold all the bare bone data of symbol meta'''
    __CURRENT_PATH = os.path.dirname(__file__)
    __Market_PATH = os.path.join(__CURRENT_PATH, 'data{0}market.h5'.format(os.sep))
    __CONSTANTS_PATH = os.path.join(__CURRENT_PATH, 'data{0}constants.h5'.format(os.sep))

    __SYMBOL_META_KEY = 'symbol_meta'
    __INDEX_META_KEY = 'index_meta'
    __TRADED_DATES_KEY = 'traded_dates'
    __RISK_FREE_RATE_KEY = 'risk_free_rate'


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

    def get_symbol_meta(self):
        '''
        If symbol meta data exists grab data from file.
        Else fetch symbol meta data from NSE website.
        '''
        if Market.__SYMBOL_META_KEY in get_store_keys(Market.__Market_PATH):
            symbol_meta = pd.read_hdf(Market.__Market_PATH, Market.__SYMBOL_META_KEY)
        else:
            warnings.warn(
                'Unable to read symbol_meta locally. Fetching data from NSE website'
            )
            self.force_load_data('symbol')
            symbol_meta = pd.read_hdf(Market.__Market_PATH, Market.__SYMBOL_META_KEY)
        return symbol_meta

    def get_index_meta(self):
        'Get meta data for index and its components'
        if Market.__INDEX_META_KEY in get_store_keys(Market.__Market_PATH):
            index_meta = pd.read_hdf(Market.__Market_PATH, Market.__INDEX_META_KEY)
        else:
            warnings.warn(
                'Unable to read symbol_meta locally. Fetching data from NSE website'
            )
            self.force_load_data('index')
            index_meta = pd.read_hdf(Market.__Market_PATH, Market.__INDEX_META_KEY)

        index_meta = index_meta.replace('nan', np.nan)
        return index_meta

    def get_index_components(
            self, index=None, index_type=None
        ):
        'Returns the symbols components of passed index or index_type'

        index_meta = self.index_meta.copy()
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
            temp_index_components = pd.read_hdf(Market.__Market_PATH, hdf_key)
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
        ):
        '''
        Get symbol list based on criteria provided.
        Pass index for getting symbols in index.
        List of indexes to get union of symbols of all indexes in list.
        start: {year as int or string, string format of date, None}
        '''
        symbol_meta = self.symbol_meta.copy()
        if symbol_list is None:
            if 'symbol_list' in dir(self):
                symbol_list = symbol_meta[symbol_meta.index.isin(self.symbol_list)].date_of_listing
            else:
                symbol_list = symbol_meta.date_of_listing
        elif isinstance(symbol_list, str) and symbol_list == 'all':
            symbol_list = symbol_meta.date_of_listing
            return symbol_list
        elif isinstance(symbol_list, str):
            symbol_list = symbol_meta[symbol_meta.index == symbol_list].date_of_listing
        elif isinstance(symbol_list, list):
            symbol_list = symbol_meta[symbol_meta.index.isin(symbol_list)].date_of_listing
        else:
            warnings.warn('Could not fetch proper symbol_list.\
                           Loading default symbol_list')
            symbol_list = symbol_meta.date_of_listing.copy()

        symbol_list = symbol_list.copy()
        if index is not None or index_type is not None:
            symbol_list_temp = self.get_index_components(
                index=index, index_type=index_type
            )
            symbol_list = symbol_list[symbol_list.index.isin(symbol_list_temp)]

        if start is not None:
            start = get_date(start, 'dt')
            symbol_list = symbol_list[symbol_list <= start]
        return symbol_list.index.tolist()

    def get_index_list(
            self, index_list=None, index_type=None
        ):
        '''Get index list based on passed params'''

        index_meta = self.get_index_meta()
        if index_list is None:
            if 'index_list' in dir(self):
                index_list = index_meta[index_meta.index.isin(self.index_list)].index_name
            else:
                index_list = index_meta.index_name
        elif isinstance(index_list, str) and index_list == 'all':
            index_list = index_meta.index_name
            return index_list
        elif isinstance(index_list, str):
            index_list = index_meta[index_meta.index == index_list].index_name
        elif isinstance(index_list, list):
            index_list = index_meta[index_meta.index.isin(index_list)].index_name
        else:
            warnings.warn('Could not fetch index_list with proper dates.\
                           Loading default index_list')
            index_list = index_meta.index_name

        index_list = index_list.copy()
        if index_type is not None:
            if isinstance(index_type, str):
                index_type = [index_type]
            index_meta = index_meta[index_meta.index_type.isin(index_type)]

        index_list = index_list[index_list.index.isin(index_meta.index)]
        return index_list.index.tolist()

    def get_traded_dates(self, start=None, end=None):
        'Generate Traded dates for NSE'

        if Market.__TRADED_DATES_KEY in get_store_keys(Market.__Market_PATH):
            traded_dates = pd.read_hdf(Market.__Market_PATH, Market.__TRADED_DATES_KEY)
        else:
            self.force_load_data('traded_dates')
            traded_dates = pd.read_hdf(Market.__Market_PATH, Market.__TRADED_DATES_KEY)

        start = get_date(start, 'str', True)
        end = get_date(end, 'str', False)

        traded_dates = traded_dates[start:end]
        traded_dates['specific_date_count'] = [i+1 for i in range(len(traded_dates))]

        return traded_dates

    def get_risk_free_rate(
        self, returns=None, freq=None, start=None, end=None, excess=False
        ):
        '''Get risk free rate'''
        if Market.__RISK_FREE_RATE_KEY in get_store_keys(Market.__Market_PATH):
            risk_free_rate = pd.read_hdf(Market.__Market_PATH, Market.__RISK_FREE_RATE_KEY)
        else:
            self.force_load_data('rf')
            risk_free_rate = pd.read_hdf(Market.__Market_PATH, Market.__RISK_FREE_RATE_KEY)

        if returns is None:
            traded_dates = self.get_traded_dates(start, end)
            returns = pd.DataFrame(0, index=traded_dates.index, columns=['returns'])
        elif isinstance(returns, pd.Series):
            returns = pd.DataFrame(returns)
        else:
            returns = returns.copy()

        if freq in ['daily', 'd', None]:
            risk_free_rate = risk_free_rate['rf_daily']
        elif freq in ['monthly', 'm']:
            risk_free_rate = risk_free_rate['rf_monthly']
        elif freq in ['yearly', 'a', 'annual', 'y']:
            risk_free_rate = risk_free_rate['rf_yearly']

        for symbol in returns.columns:
            returns[symbol] = (returns[symbol] - risk_free_rate) if excess else risk_free_rate

        return returns

    def force_load_data(self, force_load, values=None):
        '''
        Force loading helper method for saving symbol data from NSE Website to local HDFStores
        '''
        if force_load == 'symbol':
            print('Loading Symbol Meta data from NSE website')
            symbol_meta = self.fetch_symbol_meta()
            if not os.path.isdir(os.path.join(Market.__CURRENT_PATH, 'data')):
                os.mkdir(os.path.join(Market.__CURRENT_PATH, 'data'))
            symbol_meta.to_hdf(Market.__Market_PATH, Market.__SYMBOL_META_KEY)

        elif force_load == 'index':
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
                index_components_data.to_hdf(Market.__Market_PATH, hdf_key)
            symbol_meta['name_of_company'] = symbol_meta['name_of_company'].astype(str)
            symbol_meta['isin_number'] = symbol_meta['isin_number'].astype(str)
            symbol_meta['industry'] = symbol_meta['industry'].fillna('unknown')
            symbol_meta['industry'] = symbol_meta['industry'].str.lower().str.replace(' ', '_')
            symbol_meta.to_hdf(Market.__Market_PATH, Market.__SYMBOL_META_KEY)
            index_meta = index_meta.astype(str)
            index_meta.to_hdf(Market.__Market_PATH, Market.__INDEX_META_KEY)

        elif force_load == 'traded_dates':
            print('Updating traded dates')
            from nse import NSE

            if (('symbol_eod_values_close' in get_store_keys(NSE.NSE_DATA_PATH)) and
                    ('index_eod_values_close' in get_store_keys(NSE.NSE_DATA_PATH))):

                nse = NSE(symbol_list='infy', index='nifty_50')
                symbol_returns = nse.get_symbol_eod_values()
                index_returns = nse.get_index_eod_values()

                traded_dates_symbol = symbol_returns.index
                traded_dates_index = index_returns.index

                traded_dates = traded_dates_symbol.union(traded_dates_index)
            else:
                traded_dates = pd.read_hdf(Market.__CONSTANTS_PATH, Market.__TRADED_DATES_KEY)
                traded_dates = traded_dates.index

            traded_dates = pd.DataFrame(index=traded_dates)
            traded_dates['date'] = traded_dates.index
            traded_dates['date_count'] = [i+1 for i in range(len(traded_dates))]
            traded_dates['day'] = traded_dates.date.dt.day
            traded_dates['month'] = traded_dates.date.dt.month
            traded_dates['year'] = traded_dates.date.dt.year
            traded_dates['day_of_week'] = traded_dates.date.dt.dayofweek
            traded_dates['day_of_year'] = traded_dates.date.dt.dayofyear
            traded_dates['week_of_year'] = traded_dates.date.dt.week
            traded_dates = traded_dates.drop(['date'], axis=1)
            traded_dates.to_hdf(Market.__Market_PATH, Market.__TRADED_DATES_KEY)

        elif force_load == 'rf' or force_load == 'risk_free_rate':
            intervals = ['Daily', 'Monthly', 'Yearly']
            date_formats = ['%Y%m%d', '%Y%m', '%Y']
            month_end = TODAY + MonthEnd(-1)
            month_end = datetime.strftime(month_end, format='%Y%m%d')
            link = 'http://www.iimahd.ernet.in/~iffm/Indian-Fama-French-Momentum/DATA/{0}_FourFactors_and_Market_Returns_{1}.csv'

            session = requests.session()
            traded_dates = self.get_traded_dates()
            fama_french_rf = pd.DataFrame(index=traded_dates.index)
            for interval, date_format in zip(intervals, date_formats):
                interval_link = link.format(month_end, interval)
                response = session.get(interval_link)
                interval_fama_french = pd.read_csv(StringIO(response.text))
                rename_columns(interval_fama_french)
                interval_fama_french.columns.values[0] = 'date'
                interval_fama_french = interval_fama_french[['date', 'rf_pct_']]
                interval_fama_french.columns = ['date', 'rf_{0}'.format(interval.lower())]

                interval_fama_french.date = pd.to_datetime(interval_fama_french.date, format=date_format)
                interval_fama_french['rf_{0}'.format(interval.lower())] = interval_fama_french['rf_{0}'.format(interval.lower())] / 100
                interval_fama_french = interval_fama_french.set_index('date')

                fama_french_rf = fama_french_rf.join(interval_fama_french, how='outer')
            fama_french_rf = fama_french_rf.ffill()
            fama_french_rf.to_hdf(Market.__Market_PATH, Market.__RISK_FREE_RATE_KEY)
        elif force_load == 'all':
            value = values
            self.force_load_data('symbol')
            self.force_load_data('index')
            self.force_load_data('traded_dates')
            self.force_load_data('rf')
            return value

    def __init__(
            self, symbol_list=None, index=None, index_type=None,
            start=None, end=None,
            force_load=None
        ):
        self.force_load_data(force_load)
        self.symbol_meta = self.get_symbol_meta()
        self.index_meta = self.get_index_meta()
        self.start = get_date(start, start=True)
        self.end = get_date(end, start=False)

        self.symbol_list = self.get_symbol_list(
            symbol_list=symbol_list, index=index, index_type=index_type,
            start=start
        )

        self.index_list = self.get_index_list(
            index_list=index, index_type=index_type
        )
