import json
import warnings

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

import concurrent.futures
from helpers import get_date

INVESTING_FILE_PATH = 'investing.h5'


def fetch_investing_list(isin):
    search_url = 'https://www.investing.com/search/service/search'

    search_headers = {
        'Accept': 'text/plain, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive',
        'Content-Length': '103',
        'Content-Type': 'application/x-www-form-urlencoded',
        'DNT': '1',
        'Host': 'www.investing.com',
        'Origin': 'https://www.investing.com',
        'Referer': 'https://www.investing.com/equities/infosy-tech-historical-data?cid=18217',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'

    }
    session = requests.session()
    search_data = {
        'search_text': isin,
        'term': isin,
        'country_id': '14',
        'tab_id': 'All',
        'exchange_popular_symbol': 'NS'
    }
    try:
        search_response = session.post(
            search_url, headers=search_headers, data=search_data, verify=False
        )
        search_json = json.loads(search_response.text)['All'][0]
        symbol_series = pd.Series(search_json)
    except Exception as e:
        warnings.warn(
            'Unable to load list data for {0} due to {1}'.format(isin, e)
        )
        symbol_series = pd.Series()
    return symbol_series


def fetch_company_info(symbol):
    symbol_series = pd.Series()

    info_url = 'https://in.investing.com' + symbol.link
    info_headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Host': 'www.investing.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'Upgrade-Insecure-Requests': '1'
    }

    session = requests.session()
    response = session.get(info_url, headers=info_headers, verify=False)
    if response.status_code != 200:
        warnings.warn(
            '''
            Unable to establish connection to investing.com for {0}.
            Failed with responce code{1}
            '''.format(symbol, response.status_code)
        )
        return pd.DataFrame()
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        head = soup.find('div', {'class': 'clear overviewDataTable'}).find_all('span', {'class': 'float_lang_base_1'})
        value = soup.find('div', {'class': 'clear overviewDataTable'}).find_all('span', {'class': 'float_lang_base_2 bold'})
    except Exception as e:
        warnings.warn(
            'Unable to load data for {0} due to {1}'.format(symbol.Index, e)
        )
        return pd.DataFrame()
    if len(head) != len(value):
        warnings.warn(
            'Unable to load data for {0} as lengths of head and value were different'.format(symbol.Index)
        )
        return pd.DataFrame()
    for h, v in zip(head, value):
        column_name = h.text.strip().lower().replace(' ', '_').replace("'", '').replace('-', '_').replace('.', '')
        column_value = v.text.strip()
        symbol_series.loc[column_name] = column_value
    try:
        comp_info = soup.find('div', {'class': 'companyProfileHeader'}).find_all('p')
    except:
        warnings.warn(
            'Unable to load employee data for {0}'.format(symbol.Index)
        )
        return symbol_series
    if len(comp_info) == 2:
        symbol_series.loc['num_employees'] = comp_info[0].text.strip()
        symbol_series.loc['equity_type'] = comp_info[1].text.strip()
    return symbol_series


def fetch_historical_data(pair_id, from_date=None, to_date=None,
                          interval=None):
    session = requests.session()
    from_date = get_date(from_date).strftime('%d/%m/%Y')
    to_date = get_date(to_date).strftime('%d/%m/%Y')
    if interval is None:
        interval = 'Daily'

    hist_url = 'https://in.investing.com/instruments/HistoricalDataAjax'
    hist_headers = {
        'Accept': 'text/plain, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive',
        'Content-Length': '103',
        'Content-Type': 'application/x-www-form-urlencoded',
        'DNT': '1',
        'Host': 'www.investing.com',
        'Origin': 'https://www.investing.com',
        'Referer': 'https://www.investing.com/equities/infosy-tech-historical-data?cid=18217',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }
    data = {
        'action': 'historical_data',
        'curr_id': str(pair_id),
        'st_date': from_date,
        'end_date': to_date,
        'interval_sec': interval
    }
    session = requests.session()
    response = session.post(hist_url, headers=hist_headers, data=data, verify=False)
    if response.status_code != 200:
        warnings.warn(
            '''
            Unable to establish connection to investing.com for {0}.
            Failed with responce code{1}
            '''.format(pair_id, response.status_code)
        )
        return pd.DataFrame()
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        hist_data = pd.read_html(str(soup.table))[0]
        hist_data.columns = ['date', 'close', 'open', 'high', 'low', 'vol', 'simple_returns']
        hist_data.loc[:, 'date'] = pd.to_datetime(hist_data.date)
        hist_data = hist_data.sort_values('date')
    except Exception as e:
        warnings.warn(
            'Unable to load historical data for {0} due to {1}'.format(pair_id, e)
        )


def get_investing_list():
    try:
        investing_list = pd.read_hdf(INVESTING_FILE_PATH, 'list')
    except:
        force_load_data(force_load='list')
        investing_list = pd.read_hdf(INVESTING_FILE_PATH, 'list')
    return investing_list


def get_investing_company_info():
    try:
        company_info = pd.read_hdf(INVESTING_FILE_PATH, 'company_info')
    except:
        force_load_data(force_load='company_info')
        company_info = pd.read_hdf(INVESTING_FILE_PATH, 'company_info')
    return company_info


def get_investing_hist_data():
    try:
        hist_data = pd.read_hdf(INVESTING_FILE_PATH, 'hist_data')
    except:
        force_load_data(force_load='hist_data')
        hist_data = pd.read_hdf(INVESTING_FILE_PATH, 'hist_data')
    return hist_data


def force_load_data(force_load):

    if force_load == 'list':
        from symbol import Symbol
        s = Symbol()
        slist = s.symbol_meta[['isin_number']]

        investing_list = pd.DataFrame(index=slist.index.copy(), columns=[
            'flag', 'country_ID', 'exchange_name_short', 'symbol',
            'popularity_rank', 'pair_type', 'pair_type_label', 'pair_ID',
            'link', 'aql_pre_link', 'aql_link', 'trans_name', 'name', 'tab_ID',
        ])

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for symbol, isin in slist.itertuples():
                search_executor = executor.submit(
                    fetch_investing_list, isin=isin
                )
                search_result = search_executor.result()
                if search_result.empty:
                    continue
                print('Loaded investing site meta successfully for {0}'.format(symbol))
                investing_list.loc[symbol] = search_result
        investing_list = investing_list.convert_objects(convert_numeric=True)
        investing_list.to_hdf(INVESTING_FILE_PATH, 'list')

    elif force_load == 'company_info':
        investing_list = get_investing_list()
        company_info = pd.DataFrame(index=investing_list.index, columns=[
            'prev_close', 'days_range', 'revenue', 'open', '52_wk_range', 'eps',
            'volume', 'market_cap', 'dividend_(yield)', 'average_volume_(3m)',
            'p/e_ratio', 'beta', '1_year_change', 'shares_outstanding',
            'next_earnings_date', 'num_employees', 'equity_type'
        ])

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for symbol in investing_list.itertuples():
                symbol_executor = executor.submit(
                    fetch_company_info, symbol=symbol
                )
                symbol_result = symbol_executor.result()
                if symbol_result.empty:
                    continue
                company_info.loc[symbol.Index] = symbol_result
        company_info = company_info.drop(['days_range', 'open', '52_wk_range', 'volume', ], axis=1)
        company_info.to_hdf(INVESTING_FILE_PATH, 'company_info')
        try:
            column_list = ['revenue', 'market_cap', 'beta', 'p/e_ratio', '1_year_change', 'num_employees']
            for column in column_list:
                company_info[column] = company_info[column].replace('N/A', np.nan).replace(' ', np.nan)
                company_info[column] = company_info[column].replace('-', np.nan).replace('', np.nan)
                company_info[column] = company_info[column].str.replace(' ','')
                company_info[column] = company_info[column].str.replace('%','')

                import re
                powers = {'T': 10 ** 12, 'B': 10 ** 9, 'M': 10 ** 6, 'K': 10 ** 3}

                def string_to_numeric(num_str):
                    match = re.search(r"([0-9\.]+)\s?(M|B|K|T)", num_str)
                    if match is not None:
                        quantity = match.group(1) 
                        magnitude = match.group(0)[-1]
                        return float(quantity) * powers[magnitude]
                    else:
                        return float(num_str)
                company_info[column] = company_info[column].dropna().apply(string_to_numeric)
            company_info = company_info.convert_objects(convert_numeric=True)
            company_info.to_hdf(INVESTING_FILE_PATH, 'company_info')
        except Exception as e:
            warnings.warn(
                'Unable to convert columns in company info due to {1}'.format( e)
            )

    elif force_load == 'hist_data':
        investing_list = get_investing_list()
        pair_id_list = investing_list.pair_ID
        symbol_data = pd.DataFrame(columns=[
            'symbol', 'date', 'close', 'open', 'high', 'low', 'vol', 'simple_returns'
        ])

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for symbol, pair_id in pair_id_list.iteritems():
                hist_data_executor = executor.submit(
                    fetch_company_info, symbol=symbol
                )
                hist_data_result = hist_data_executor.result()
                if hist_data_result.empty:
                    continue
                hist_data_result.insert(0, 'symbol', symbol)
                symbol_data = symbol_data.append(hist_data_result)
                print('Loaded historical data succesfully for {0}'.format(symbol))
        symbol_data = symbol_data.reset_index(drop=True)
        symbol_data = symbol_data.sort_values(['symbol', 'date'])
        symbol_data.to_hdf(INVESTING_FILE_PATH, 'hist_data')

    elif force_load == 'all':
        force_load_data('list')
        force_load_data('company_info')
        force_load_data('hist_data')
