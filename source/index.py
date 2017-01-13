import warnings
from datetime import datetime
from market import Market

import numpy as np
import pandas as pd
from helpers import clean_file
import concurrent.futures


# File path
INDEX_DATA_PATH = 'index_data.h5'

# Constants
TODAY = datetime.combine(datetime.today().date(), datetime.min.time())
INDEX_LIST = pd.read_hdf('constants.h5', 'index_list')


class Index(Market):

    def get_index_list(self, index_list=None,
                       start=None):
        if index_list is None:
            try:
                index_list = self.index_list.copy()
            except:
                index_list = list(INDEX_LIST.index)
        elif isinstance(index_list, str):
            index_list = index_list.lower().replace(' ', '_')
            index_list = [index_list]
        elif isinstance(index_list, list):
            index_list = [index.lower().replace(' ', '_') for index in index_list]
        elif isinstance(index_list, pd.Series):
            index_list = index_list.str.lower().str.replace(' ', '_').tolist()
        else:
            warnings.warn('Unable to get index list. Creating a list with Nifty 50')
            index_list = ['nifty_50']

        return index_list

    def get_index_hist(self, start=None, end=None, index_list=None,
                       force_load=False):
        index_list = self.get_index_list(index_list=index_list,
                                         start=start)
        if not force_load:
            try:
                index_data = pd.read_hdf(INDEX_DATA_PATH, 'index_data_daily')
                if index_data.empty:
                    warnings.warn('No data present in local file. grabbing data from NSE')
                    index_data = self.get_index_hist(force_load=True)
            except:
                index_data = self.get_index_hist(force_load=True)
        else:
            idata = pd.DataFrame(index=pd.date_range(start=datetime(1994, 1, 1), end=TODAY))
            idata.index.rename('date', inplace=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for index in INDEX_LIST.values:
                    p = executor.submit(self.fetch_index_data, index)
                    res = p.result()
                    res = res.Close
                    res = res.rename(index.lower().replace(' ', '_'))
                    idata = idata.join(res)
            index_data = idata.dropna(how='all').astype(np.float64)
            index_data.to_hdf(INDEX_DATA_PATH, 'index_data_daily')
            clean_file(INDEX_DATA_PATH)
            return index_data
        start = self.get_date(start, 'str', start=True)
        end = self.get_date(end, 'str', start=False)
        index_list = list(set(index_list).intersection(index_data.columns))
        index_data = index_data[index_list]
        index_data = index_data[start:end]
        return index_data

    def get_index_returns(self, index_list=None, start=None,
                          end=TODAY,
                          force_load=False):
        index_list = self.get_index_list(index_list=index_list,
                                         start=start)
        if not force_load:
            try:
                index_returns = pd.read_hdf(INDEX_DATA_PATH, 'index_returns_daily')
            except:
                index_returns = self.get_index_returns(force_load=True)
        else:
            index_data = self.get_index_hist(index_list=INDEX_LIST.index.tolist(), start='1996')
            index_returns = np.log(index_data / index_data.shift(1))
            index_returns = index_returns.ix[1:]
            index_returns.to_hdf(INDEX_DATA_PATH, 'index_returns_daily')
            clean_file(INDEX_DATA_PATH)
            return index_returns
        start = self.get_date(start, start=True, out='str')
        end = self.get_date(end, start=False, out='str')
        index_list = list(set(index_list).intersection(index_returns.columns))
        index_returns = index_returns[index_list]
        index_returns = index_returns[start:end]
        return index_returns

    def __init__(self, index_list=None, start=None, end=None,
                 force_load=False):
        '''
        Symbol Object containing all the necessary methods
        for handling symbol related functions
        '''
        if force_load:
            self.get_index_hist(force_load=True)
            self.get_index_returns(force_load=True)
        self.start = self.get_date(start, start=True)
        self.end = self.get_date(end, start=False)
        self.index_list = self.get_index_list(index_list=index_list)