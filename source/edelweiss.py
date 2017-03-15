'''Module for data downloading from edelweiss'''
import json

import pandas as pd
import requests

from nse import NSE


class Edelweiss(NSE):

    def load_url_headers(self, request_type):
        '''
        Header and url links required for different requests made to Edelweiss
        '''

        if request_type == 'search':
            url = 'https://ewmw.edelweiss.in/ewreports/api/search/gsa/suggestions'
            headers = {
                'Accept': 'application/json, text/plain, */*',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive',
                'Content-Length': '47',
                'Content-Type': 'application/json;charset=UTF-8',
                'DNT': '1',
                'Host': 'ewmw.edelweiss.in',
                'Origin': 'https://www.edelweiss.in',
                'Referer': 'https://www.edelweiss.in',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
            }
        return url, headers

    def fetch_symbol_codes(self):
        '''
        Fetch Edelweiss symbol meta for symbol.
        '''
        symbol_meta = self.symbol_meta.copy()
        edelweiss_link = pd.DataFrame(index=symbol_meta.index.copy(), columns=['url', 'code'])

        search_url, search_headers = self.load_url_headers('search')
        session = requests.session()
        for symbol, isin in zip(symbol_meta.index, symbol_meta.isin_number):
            search_payload = {
                'SearchString': isin
            }
            response = session.post(
                url=search_url,
                json=search_payload,
                headers=search_headers
            )
            if response.status_code != 200:
                print('Unable to load for {0}'.format(symbol))
                continue

            try:
                link = json.loads(response.text)[0]['Route'].strip()
            except Exception as e:
                print('Unable to load for {0} due to {1}'.format(symbol, e))
                continue

            link = 'https://www.edelweiss.in' + link
            edelweiss_link.loc[symbol, 'url'] = link
            print('Loaded Successfully for {0}'.format(symbol))
        regex = edelweiss_link.url.str.extractall(r'(\d+)')
        regex = regex.reset_index().groupby('symbol').last().drop(['match'], axis=1)
        regex.columns = ['code']
        edelweiss_link['code'] = regex['code']
        edelweiss_link.to_hdf('edelweiss.h5', 'links')

    def get_company_codes(self, symbol_list=None):

        try:
            company_codes = pd.read_hdf('edelweiss.h5', 'links')
        except:
            self.load_symbol_codes()
            company_codes = pd.read_hdf('edelweiss.h5', 'links')
        symbol_list = self.get_symbol_list(symbol_list=symbol_list)
        company_codes = company_codes[company_codes.index.isin(symbol_list.index)]
        return company_codes

