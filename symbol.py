'''Module for loading the symbol metadata'''
from pandas import HDFStore
from nsepy import get_history
# from pandas_datareader.data import DataReader
import urllib
import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from datetime import datetime

INDEX_URL = 'https://www.nseindia.com/content/indices/{0}.csv'
SYMBOL_LIST_URL = 'https://www.nseindia.com/content/equities/EQUITY_L.csv'
YAHOO_URL = 'http://real-chart.finance.yahoo.com/table.csv?'
SYMBOL_META_PATH = 'symbol_meta.h5'
SYMBOL_DATA_PATH = 'symbol_data_daily.h5'
TEMP_DATA_PATH = 'symbol_data_temp.h5'
HIST_SYMBOL_SCHEMA = ['symbol', 'date', 'prev_close', 'open', 'high',
                      'low', 'last', 'close', 'adj_close', 'vwap', 'volume',
                      'turnover', 'pct_deliverble', 'simple_returns',
                      'log_returns', 'adj_simple_returns', 'adj_log_returns']


class SafeHDFStore(HDFStore):

    def __init__(self, *args, **kwargs):
        import os
        probe_interval = kwargs.pop("probe_interval", 0.1)
        self._lock = "%s.lock" % args[0]
        while True:
            try:
                self._flock = os.open(self._lock, os.O_CREAT |
                                      os.O_EXCL |
                                      os.O_WRONLY)
                break
            except FileExistsError:
                import time
                time.sleep(probe_interval)

        HDFStore.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        import os
        HDFStore.__exit__(self, *args, **kwargs)
        os.close(self._flock)
        os.remove(self._lock)


def rename_columns(data):
    '''Rename columns and index to lowercase'''
    data.columns = [name.strip().lower().replace(' ', '_').replace('%', 'pct_')
                    for name in data.columns]
    if data.index.names[0] is not None:
        data.index.names = [name.strip().lower().replace(' ', '_')
                            for name in data.index.names]


def get_store_keys(path):
    store = pd.HDFStore(path)
    keys = store.keys()
    store.close()
    return keys


def get_symbol_meta():

    try:
        symbol_meta = pd.read_hdf(SYMBOL_META_PATH, 'symbol_meta')
    except:
        symbol_meta = load_symbol_meta()
        try:
            from os import remove
            remove(SYMBOL_META_PATH)
        except:
            pass
        symbol_meta.to_hdf(SYMBOL_META_PATH, 'symbol_meta')
    return symbol_meta


def load_symbol_meta():
    import requests
    from io import StringIO

    with open('index_urls.json', 'r') as fp:
        INDEX_DICT = json.load(fp, object_pairs_hook=OrderedDict)

    r = requests.get(SYMBOL_LIST_URL)
    symbol_meta = pd.read_csv(StringIO(r.text), index_col='SYMBOL')

    r = requests.get(INDEX_URL.format(INDEX_DICT['mkt_index']['Nifty 500']))
    nifty_500 = pd.read_csv(StringIO(r.text), index_col='Symbol')
    symbol_meta = symbol_meta.join(nifty_500['Industry']).fillna('UNKONOWN')
    rename_columns(symbol_meta)
    symbol_meta = symbol_meta[(symbol_meta['series'] == 'EQ')]
    symbol_meta = symbol_meta[['name_of_company', 'date_of_listing',
                              'industry']]
    symbol_meta.date_of_listing = pd.to_datetime(symbol_meta.date_of_listing)

    for index_category in INDEX_DICT:
        for index in INDEX_DICT[index_category]:
            ind = INDEX_DICT[index_category][index]
            index = index.lower().replace(' ', '_')

            r = requests.get(INDEX_URL.format(ind))
            try:
                index_data = pd.read_csv(StringIO(r.text), index_col='Symbol')
            except Exception as e:
                print('Unable to get symbol_meta list for {0} due to {1} error'
                      .format(index, e))
                continue

            index_data[index] = True
            index_data[index]
            symbol_meta = symbol_meta.join(index_data[index])
    symbol_meta.fillna(value=False, inplace=True)
    return symbol_meta


def fetch_historical_data_nse(symbol):
    # from dateutil.relativedelta import relativedelta

    '''Fetch data from NSE'''
    from_date = symbol[1]
    to_date = datetime.today()
    symbol = symbol[0]
    print('Loading the data from NSE for {0} from {1} to {2}'.
          format(symbol, from_date.date(), to_date.date()))

    # Get data from NSE
    try:
        nse_data = get_history(symbol=symbol,
                               start=from_date,
                               end=to_date)
        nse_data.drop(['Series', 'Trades',
                      'Deliverable Volume'],
                      1, inplace=True)
        nse_data.index = pd.to_datetime(nse_data.index)
    except Exception as e:
        print(e)
        print('Warning: Could not get data for {0} from NSE'.format(symbol))
        return False

    try:
        enco = urllib.parse.urlencode
        base = enco((('a', from_date.month), ('b', from_date.day),
                    ('c', from_date.year), ('d', to_date.month),
                    ('e', to_date.day), ('f', to_date.year),
                    ('g', 'd'), ('ignore', '.csv')))

        url = YAHOO_URL + 's=' + symbol + '.NS&' + base
        yahoo_data = pd.read_csv(url, parse_dates=['Date'],
                                 index_col='Date',
                                 usecols=['Date', 'Adj Close'])
        nse_data = nse_data.join(yahoo_data)
    except Exception as e:
        print('Warning: Could not get data for {0} from Yahoo'.format(symbol))
        nse_data['adj_close'] = nse_data['Close']

    rename_columns(nse_data)
    nse_data['symbol'] = nse_data['symbol'].replace({';': ''}, regex=True)
    nse_data['symbol'] = [symbol for i in range(0, len(nse_data))]
    nse_data = nse_data.ix[nse_data.index.drop_duplicates(), :]
    nse_data = nse_data.sort_index()
    nse_data['simple_returns'] = ((nse_data.close - nse_data.prev_close) /
                                  nse_data.prev_close)*100
    nse_data['log_returns'] = np.log(nse_data.close/nse_data.prev_close)*100
    nse_data['adj_simple_returns'] = ((nse_data.adj_close -
                                      nse_data.adj_close.shift(1)) /
                                      nse_data.adj_close.shift(1))*100
    nse_data['adj_log_returns'] = np.log(nse_data.adj_close /
                                         nse_data.adj_close.shift(1))*100

    # # Generate values for Adjusted Close Column
    # nse_data['adj_close'] = nse_data.close

    # # Adjustments for splits
    # for i in nse_data.itertuples():
    #     returns = round(((i.close - i.prev_close)/i.prev_close), 1)
    #     date = i.Index - relativedelta(days=1)
    #     if returns <= -0.5:
    # noqa        nse_data.adj_close[date:] = nse_data.adj_close[date:]*abs(returns)

    # Adjusting other columns for maintaining integrity
    nse_data.volume = nse_data.volume.astype(int)
    nse_data['pct_deliverble'] = nse_data['pct_deliverble']*100
    nse_data.reset_index(inplace=True)
    nse_data = nse_data[HIST_SYMBOL_SCHEMA]
    # nse_data.date = nse_data.date.apply(lambda x: x.toordinal())
    with SafeHDFStore(TEMP_DATA_PATH) as store:
        store.put('daily_symbol', value=nse_data, format='t',
                  append=True, data_columns=['symbol'],
                  min_itemsize={'symbol': 15})


def get_hist_data(symbol_list):

    if len(symbol_list.columns) == 1:
        try:
            hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')
        except Exception as e:
            print('Unable to open HDF file due to {0}'.format(e))
            symbol_list['date'] = symbol_list.date_of_listing
            hist_data = get_hist_data(symbol_list)
        try:
            symbol_list = symbol_list.drop(['date'], axis=1)
        except:
            pass
        date_list = hist_data.groupby('symbol')['date'].max()
        date_list = symbol_list.join(date_list)
        date_list.date[date_list.date.isnull()] = date_list.date_of_listing[date_list.date.isnull()] # noqa
        date_list = date_list[(date_list.date < datetime(datetime.today().year, datetime.today().month, 1))] # noqa
        if len(date_list) > 10:
            print('Getting stock data for {0} companies again'.
                  format(len(date_list)))
            hist_data = get_hist_data(date_list)
    else:
        symbol_list = symbol_list.date
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for symbol in symbol_list.iteritems():
                p = executor.submit(fetch_historical_data_nse, symbol)
                p.result()
        # for symbol in symbol_list.itertuples():
        #     print(symbol)
        #     fetch_historical_data_nse(symbol)
        hist_data_temp = pd.read_hdf(TEMP_DATA_PATH, 'daily_symbol')
        try:
            hist_data = pd.read_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')
        except:
            hist_data = pd.DataFrame(columns=hist_data_temp.columns)

        hist_data = hist_data.append(hist_data_temp)
        del hist_data_temp
        hist_data = hist_data.sort_values(['symbol', 'date'])
        hist_data = hist_data
        hist_data = hist_data.reset_index()[HIST_SYMBOL_SCHEMA]
        try:
            from os import remove
            remove(TEMP_DATA_PATH)
            remove(SYMBOL_DATA_PATH)
        except:
            pass
        hist_data = hist_data.drop_duplicates(['symbol', 'date'])
        hist_data.to_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')

    return hist_data
