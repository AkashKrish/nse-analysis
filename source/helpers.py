'''Helper Methods for analysis'''
import os
import warnings
from datetime import datetime

# import numpy as np
import pandas as pd

CURRENT_PATH = os.path.dirname(__file__)
TEMP_DATA_PATH = os.path.join(CURRENT_PATH, 'data{0}temp_data.h5'.format(os.sep))


class SafeHDFStore(pd.HDFStore):
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
                # Delete lock on file
                import time
                time.sleep(probe_interval)

        pd.HDFStore.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        pd.HDFStore.__exit__(self, *args, **kwargs)
        os.close(self._flock)
        os.remove(self._lock)


def rename_columns(data):
    '''Rename columns and index to lowercase for easier access'''
    data.columns = [name.strip().lower().replace(' ', '_').replace('%', 'pct_')
                    for name in data.columns]
    if data.index.names[0] is not None:
        data.index.names = [name.strip().lower().replace(' ', '_')
                            for name in data.index.names]


def num_missing(series, start='beginning'):
    '''Find number of mising values in a series'''
    if start == 'valid_index':
        series = series[series.first_valid_index():]
    return sum(series.isnull())


def get_store_keys(path):
    '''Get keys in the provided H5 file path'''
    if os.path.isfile(path):
        store = pd.HDFStore(path)
        keys = pd.Series(store.keys())
        keys = keys.str[1:].tolist()
        store.close()
    else:
        keys = []
    return keys


def delete_key(path, key):
    '''Delete "key" from HDFStore in "path"'''
    keys = get_store_keys(path)
    if key in keys:
        store = pd.HDFStore(path)
        del store[key]
        store.close()
    else:
        raise KeyError('Key not present in path')


def null_count_returns(data):
    '''Get null counts of columns'''
    total_nulls = data.apply(num_missing, start='valid_index')
    return total_nulls


def clean_file(path):
    '''Clean file to recreate all dataframes in the path file'''
    print('Cleaning file {0}'.format(path))
    keys = get_store_keys(path)
    if len(keys) == 0:
        print('No data present in path')
        return
    if os.path.isfile(TEMP_DATA_PATH):
        os.remove(TEMP_DATA_PATH)
    store = pd.HDFStore(path)
    keys = store.keys()
    store.close()
    for key in keys:
        temp = pd.read_hdf(path, key)
        temp.to_hdf(TEMP_DATA_PATH, key)
    os.remove(path)
    os.rename(src=TEMP_DATA_PATH, dst=path)


def export_csv(path, key):
    '''Export data in HDFStore to csv in the path passed'''
    if isinstance(key, list):
        keys = key
    elif isinstance(key, str) and key.lower() == 'all':
        # Get all store keys from path
        keys = get_store_keys(path)
    elif isinstance(key, str) and bool(key in keys):
        keys = [key]
    else:
        raise KeyError(
            'Invalid Key specified'
        )
    directory_path = os.path.dirname(path)
    for key in keys:
        data = pd.read_hdf(path, key)
        data.to_csv(directory_path+ os.sep + key + '.csv')


def get_date(date=None, out='dt', start=True):
    '''Generate date in proper format for "date" passed'''
    if date is None:
        if start:
            date = datetime(1996, 1, 1)
        else:
            date = datetime.combine(datetime.today().date(), datetime.min.time())
    elif isinstance(date, str) and len(date) == 4:
        date = datetime(int(date), 1, 1)
    elif isinstance(date, str) and len(date) == 7:
        date = datetime(int(date[0:4]), int(date[-2:]), 1)
    elif isinstance(date, str) and len(date) == 10:
        date = datetime.strptime(date, '%Y-%m-%d')
    elif isinstance(date, int) and date > 1900:
        date = datetime(date, 1, 1)
    elif isinstance(date, datetime):
        pass
    else:
        raise ValueError('Invalid Value for date')
    if out == 'str':
        date = date.strftime('%Y-%m-%d')
    return date
