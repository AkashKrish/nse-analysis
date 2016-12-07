'''Module for loading the symbol metadata'''  # noqa

import json
import os
from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from nsepy import get_history
from pandas import HDFStore

# URLs for getting data
INDEX_URL = 'https://www.nseindia.com/content/indices/{0}.csv'
SYMBOL_LIST_URL = 'https://www.nseindia.com/content/equities/EQUITY_L.csv'
YAHOO_URL = 'http://real-chart.finance.yahoo.com/table.csv?'

# File path
SYMBOL_META_PATH = 'symbol_meta.h5'
SYMBOL_DATA_PATH = 'symbol_data_daily.h5'
SYMBOL_DIVIDEND_PATH = 'dividend_data.h5'
TEMP_DATA_PATH = 'symbol_data_temp.h5'

# Schema for required dataframes
HIST_SYMBOL_SCHEMA = ['symbol', 'date', 'prev_close', 'open', 'high',
                      'low', 'last', 'close', 'adj_close', 'vwap', 'volume',
                      'turnover', 'pct_deliverble', 'simple_returns',
                      'log_returns', 'adj_simple_returns', 'adj_log_returns']

DIVIDEND_SCHEMA = ['symbol', 'date', 'action', 'value']


class SafeHDFStore(HDFStore):
    '''
    Helper class for proper writing of data to H5 files
    while using multi-threads by making only one thread
    to write at a given time.
    '''

    def __init__(self, *args, **kwargs):
        '''Initiate lock for prebinting other threads to access file'''
        probe_interval = kwargs.pop("probe_interval", 0.1)
        self._lock = "%s.lock" % args[0]
        while True:
            try:
                self._flock = os.open(self._lock, os.O_CREAT |
                                      os.O_EXCL |
                                      os.O_WRONLY)
                break
            except FileExistsError:
                '''Delete lock on file'''
                import time
                time.sleep(probe_interval)

        HDFStore.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        HDFStore.__exit__(self, *args, **kwargs)
        os.close(self._flock)
        os.remove(self._lock)


def rename_columns(data):
    '''Rename columns and index to lowercase for easier access'''
    data.columns = [name.strip().lower().replace(' ', '_').replace('%', 'pct_')
                    for name in data.columns]
    if data.index.names[0] is not None:
        data.index.names = [name.strip().lower().replace(' ', '_')
                            for name in data.index.names]


def get_store_keys(path):
    '''Get keys in the provided H5 file path'''
    store = pd.HDFStore(path)
    keys = store.keys()
    store.close()
    return keys


def fetch_symbol_meta():
    '''
    Method to grab symbol meta data from NSE website
    '''
    import requests
    from io import StringIO
    from collections import OrderedDict

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


def fetch_historical_data(symbol):
    '''Fetch all data from NSE and Adj Close from Yahoo Finance'''
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
        print('Warning: Could not get data for {0} from NSE'.format(symbol))
        return False

    # Generate values for Adjusted Close Column
    try:
        enco = urlencode
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
    nse_data = nse_data.reset_index().drop_duplicates(
        subset='date', keep='last').set_index('date')

    nse_data = nse_data.sort_index()
    nse_data['simple_returns'] = ((nse_data.close - nse_data.prev_close) /
                                  nse_data.prev_close) * 100

    nse_data['log_returns'] = np.log(
        nse_data.close / nse_data.prev_close) * 100

    nse_data['adj_simple_returns'] = ((nse_data.adj_close -
                                       nse_data.adj_close.shift(1)) /

                                      nse_data.adj_close.shift(1)) * 100

    nse_data['adj_log_returns'] = np.log(nse_data.adj_close /
                                         nse_data.adj_close.shift(1)) * 100

    # Generate values for Adjusted Close Column
    # nse_data['adj_close'] = nse_data.close

    # # Adjustments for splits
    # for i in nse_data.itertuples():
    #     returns = round(((i.close - i.prev_close)/i.prev_close), 1)
    #     date = i.Index - relativedelta(days=1)
    #     if returns <= -0.5:
    # noqa        nse_data.adj_close[date:] = nse_data.adj_close[date:]*abs(returns)

    # Adjusting other columns for maintaining integrity
    nse_data.volume = nse_data.volume.astype(int)
    nse_data['pct_deliverble'] = nse_data['pct_deliverble'] * 100

    nse_data.reset_index(inplace=True)
    nse_data = nse_data[HIST_SYMBOL_SCHEMA]
    with SafeHDFStore(TEMP_DATA_PATH) as store:
        store.put('daily_symbol', value=nse_data, format='t',
                  append=True, min_itemsize={'symbol': 15})


def fetch_dividend_data(symbol):
    '''Fetch dividend from Yahoo'''
    from_date = symbol[1]
    to_date = datetime.today()
    symbol = symbol[0]
    print('Loading dividend data for {0} from {1} to {2}'.
          format(symbol, from_date.date(), to_date.date()))

    try:
        from pandas_datareader.data import DataReader
        dividend_data = DataReader(symbol+'.NS', data_source='yahoo-actions',
                                   start=from_date, end=to_date)
        if dividend_data.empty:
            dividend_data = pd.DataFrame([['N',0.0]], index=[from_date],
                                         columns=['action', 'value'])
    except Exception as e:
        print(e)
        print('Warning: Could not get dividend data for {0} from Yahoo'.
              format(symbol))
        return False
    dividend_data.index.name = 'date'
    rename_columns(dividend_data)
    dividend_data['symbol'] = [symbol for i in range(0, len(dividend_data))]
    dividend_data = dividend_data.reset_index().drop_duplicates(subset='date',
                                                                keep='last')
    dividend_data = dividend_data.sort_values('date')

    dividend_data.reset_index(inplace=True)
    dividend_data = dividend_data[DIVIDEND_SCHEMA]
    with SafeHDFStore(TEMP_DATA_PATH) as store:
        store.put('dividend_data', value=dividend_data, format='t',
                  append=True, min_itemsize={'symbol': 15,'action':20})


def get_symbol_meta():
    '''
    If SYMBOL_META_PATH exists grab data from file.
    Else fetch symbol meta data from NSE website.
    '''
    try:
        symbol_meta = pd.read_hdf(SYMBOL_META_PATH, 'symbol_meta')
    except:
        symbol_meta = fetch_symbol_meta()
        try:
            os.remove(SYMBOL_META_PATH)
        except:
            pass
        symbol_meta.to_hdf(SYMBOL_META_PATH, 'symbol_meta')
    return symbol_meta


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
        date_list.date[date_list.date.isnull()] = date_list.date_of_listing[date_list.date.isnull()]  # noqa

        date_list = date_list[(date_list.date < datetime(datetime.today().year, datetime.today().month - 1, 1))]  # noqa

        if len(date_list) > 10:
            print('Getting stock data for {0} companies again'.
                  format(len(date_list)))
            print(date_list.index)
            hist_data = get_hist_data(
                date_list[~date_list.index.isin(['DPL', 'ZENITHEXPO'])])

    else:
        symbol_list = symbol_list.date
        try:
            os.remove(TEMP_DATA_PATH)
        except:
            pass
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for symbol in symbol_list.iteritems():
                p = executor.submit(fetch_historical_data, symbol)
                p.result()
        # for symbol in symbol_list.iteritems():
        #     fetch_historical_data(symbol)
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
            os.remove(TEMP_DATA_PATH)
            os.remove(SYMBOL_DATA_PATH)
        except:
            pass
        hist_data = hist_data.drop_duplicates(['symbol', 'date'])
        hist_data.to_hdf(SYMBOL_DATA_PATH, 'symbol_data_daily')

    return hist_data


def get_dividend_data(symbol_list):

    try:
        div_data = pd.read_hdf(SYMBOL_DIVIDEND_PATH, 'dividend_data')
    except Exception as e:
        print('Unable to open Dividend file due to {0}'.format(e))
        div_data = pd.DataFrame(columns=DIVIDEND_SCHEMA)

    sym_div_list = div_data.symbol.unique()
    fetch_list = symbol_list[~symbol_list.index.isin(sym_div_list)]
    fetch_list = pd.Series(fetch_list, index=fetch_list.index)
    if len(fetch_list) > 25:
        print('Getting dividend data for {0} companies'.
              format(len(fetch_list)))
        try:
            os.remove(TEMP_DATA_PATH)
        except:
            pass
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for symbol in fetch_list.iteritems():
                p = executor.submit(fetch_dividend_data, symbol)
                p.result()
        # for symbol in symbol_list.iteritems():
        #     fetch_historical_data(symbol)

        div_data_temp = pd.read_hdf(TEMP_DATA_PATH, 'dividend_data')
        div_data = div_data.append(div_data_temp)
        del div_data_temp
        div_data = div_data.sort_values(['symbol', 'date'])
        div_data = div_data.reset_index()[DIVIDEND_SCHEMA]
        div_data = div_data.drop_duplicates(['symbol', 'date'])
        try:
            os.remove(TEMP_DATA_PATH)
            os.remove(SYMBOL_DIVIDEND_PATH)
        except:
            pass
        div_data.to_hdf(SYMBOL_DIVIDEND_PATH, 'dividend_data')

    return div_data
