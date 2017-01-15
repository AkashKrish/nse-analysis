import os
import warnings

import pandas as pd
from pandas import HDFStore


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


def num_missing(x, start='beginning'):
    if start == 'valid_index':
        x = x[x.first_valid_index():]
    return sum(x.isnull())


def get_store_keys(path):
    '''Get keys in the provided H5 file path'''
    store = pd.HDFStore(path)
    keys = store.keys()
    store.close()
    return keys


def null_count_returns(data):
    '''Get null counts of columns'''
    total_nulls = data.apply(num_missing, start='valid_index')
    return total_nulls


def clean_file(path):
    '''Clean file to recreate all dataframes in the path file'''
    try:
        keys = get_store_keys(path)
    except:
        warnings.warn(
            'path file does not exist'
        )
        return
    try:
        os.remove('temp_data.h5')
    except:
        pass
    for frame in keys:
        temp = pd.read_hdf(path, frame[1:])
        temp.to_hdf('temp_data.h5', frame[1:])
    os.remove(path)
    os.rename(src='temp_data.h5', dst=path)


def export_csv(path, key=None):
    '''Clean file to recreate all dataframes in the path file'''
    try:
        keys = get_store_keys(path)
    except:
        warnings.warn(
            'path file does not exist'
        )
        return
    if key is not None:
        try:
            data = pd.read_hdf(path, key)
        except Exception as e:
            warnings.warn(
                'Unable to read specified key from path due to {0}'.format(e)
            )
        data.to_csv(key + '.csv')
        return
    for frame in keys:
        data = pd.read_hdf(path, frame[1:])
        data.to_csv(frame[1:] + '.csv')
