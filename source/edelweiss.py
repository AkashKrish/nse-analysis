import time

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from helpers import SafeHDFStore, clean_file
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from symbol import Symbol


def create_header_list(soup):

    edelweiss_list = pd.Series()
    symbol_info_soup = soup.find('div', {'id': 'market-overview'})
    edelweiss_list['edelweiss_url'] = soup.find('meta', {'name': 'twitter:url'})['content'].strip()
    edelweiss_list['company_name'] = symbol_info_soup.find('span', {'ng-bind': '::vm.compname'}).text.strip()
    edelweiss_list['sector'] = symbol_info_soup.find('a', {'href': '', 'class': 'ng-binding'}).text.strip()
    try:
        edelweiss_list['bullish'] = soup.find('div', {'class': 'bullish'}).find('span').text.strip()
    except:
        pass

    basic_stats = soup.find('div', {'class': 'row border border-top mobile-chart-info'})
    basic_stats_spans = basic_stats.find_all('span', {'class': 'value ng-binding'})
    edelweiss_list['pe_ratio'] = basic_stats_spans[1].text.strip()
    edelweiss_list['div_yield'] = basic_stats_spans[2].text.strip()
    edelweiss_list['mcap'] = basic_stats_spans[3].text.replace(',', '').strip()
    edelweiss_list['beta'] = basic_stats_spans[4].text.strip()

    identifier = soup.find('div', {'id': 'shareholding-pattern'})
    identifier_list = identifier.find('div', {'class': 'share'}).find_all('li')
    edelweiss_list['bse_identifier'] = identifier_list[0].text[5:].strip()
    edelweiss_list['nse_identifier'] = identifier_list[1].text[5:].strip()
    edelweiss_list['isin_identifier'] = identifier_list[2].text[6:].strip()

    shareholding = soup.find('div', {'id': 'shareholding-pattern'})
    shareholding_span = shareholding.find('div', {'config': 'vm.SHChartData'}).find_all('span')
    edelweiss_list['promoter_holding'] = shareholding_span[0].text[:-1].strip().split()[-1]
    edelweiss_list['fii_holding'] = shareholding_span[1].text[:-1].strip().split()[-1]
    edelweiss_list['dii_mf_holding'] = shareholding_span[2].text[:-1].strip().split()[-1]
    edelweiss_list['others_holding'] = shareholding_span[3].text[:-1].strip().split()[-1]
    
    return edelweiss_list


def load_ratios(soup):
    ratios_df = pd.DataFrame()

    for ratio_type in ['MR', 'RR', 'BR', 'LR', 'TR', 'PSD', 'VR']:
        ratio_soup = soup.find('div', {'id': 'key-ratios'})
        ratio_soup = ratio_soup.find('div', {'ng-show': 'vm.{0}'.format(ratio_type)})
        ratio_df_list = pd.read_html(str(ratio_soup), index_col=0, header=0)
        consolidated_element = soup.find('div', {'id': 'key-ratios'})
        consolidated_element = consolidated_element.find('a', {'ng-click': 'vm.KRConsolidated=true'})

        # Skip processing if data is empty
        if len(ratio_df_list) == 1 and ratio_df_list[0].empty:
            continue
        elif len(ratio_df_list) == 2 and ratio_df_list[0].empty and ratio_df_list[1].empty:
            continue

        # For symbols with only standalone and no consolidated
        if consolidated_element['class'][0] == 'ng-hide':
            consolidated = pd.DataFrame()
            try:
                standalone = ratio_df_list[1].reset_index()
            except:
                continue
            standalone.insert(0, 'ratio_type', ratio_type.lower() + '_standalone')
        else:

            consolidated = ratio_df_list[0].reset_index()
            consolidated.insert(0, 'ratio_type', ratio_type.lower() + '_consolidated')
            standalone = ratio_df_list[1].reset_index()
            standalone.insert(0, 'ratio_type', ratio_type.lower() + '_standalone')
        ratio_df = pd.concat([consolidated, standalone])

        if ratio_df.empty:
            continue
        ratio_df = ratio_df.replace('-', np.nan)
        ratio_df = ratio_df.rename(columns={'index': 'ratio_name'})
        ratios_df = ratios_df.append(ratio_df)
        ratios_df = ratios_df.convert_objects(convert_numeric=True)
    return ratios_df.reset_index(drop=True)


def load_pnl_static(soup):
    pnl_soup = soup.find('div', {'id': 'financial-health'})
    pnl_soup = pnl_soup.find('table', {'ng-show': '!vm.PNLConsolidated'})
    yearly_pnl = pd.read_html(str(pnl_soup), index_col=0, header=0)[0].reset_index()
    yearly_pnl = yearly_pnl.convert_objects(convert_numeric=True)
    return yearly_pnl


def load_balancesheet_static(soup):
    balancesheet_soup = soup.find('div', {'ui-view': 'balancesheet'})
    balancesheet = pd.read_html(str(balancesheet_soup), index_col=0, header=0)[0]
    balancesheet = balancesheet.reset_index()
    balancesheet = balancesheet.convert_objects(convert_numeric=True)
    return balancesheet


def load_cashflow(soup):

    consolidated_element = soup.find('a', {'ng-click': 'vm.CFConsolidated=true'})
    cashflow_soup = soup.find_all('div', {'class': 'key-ratio profit-loss'})[1]
    cashflow_df_list = pd.read_html(str(cashflow_soup), index_col=0, header=0)

    if consolidated_element['class'][0] == 'ng-hide':
        consolidated = pd.DataFrame()
    else:
        consolidated = cashflow_df_list[0].reset_index()
        consolidated.insert(0, 'cf_type', 'consolidated')

    standalone = cashflow_df_list[1].reset_index()
    standalone.insert(0, 'cf_type', 'standalone')
    cashflow = consolidated.append(standalone)
    cashflow = cashflow.replace('-', np.nan)
    cashflow = cashflow.rename(columns={'index': 'cf_activity'})
    cashflow = cashflow.convert_objects(convert_numeric=True)
    return cashflow.reset_index(drop=True)


def load_peers_comparision(soup):
    peer_soup = soup.find('div', {'id': 'peer-performance'})
    peer_comparision = pd.read_html(str(peer_soup), header=1)[0]
    peer_comparision = peer_comparision.convert_objects(convert_numeric=True)
    return peer_comparision


def load_initial_data(key):
    try:
        data = pd.read_hdf('edelweiss.h5', key)
    except:
        if key != 'list':
            data = pd.DataFrame()
        else:
            symbol_list = Symbol()
            symbol_list = symbol_list.symbol_list
            data = pd.DataFrame(
                index=symbol_list.index, columns=[
                    'edelweiss_url', 'company_name', 'sector',
                    'pe_ratio', 'div_yield', 'mcap', 'beta', 'bullish',
                    'bse_identifier', 'nse_identifier', 'isin_identifier',
                    'promoter_holding', 'fii_holding', 'dii_mf_holding', 'others_holding'
                ])
    return data


def load_edelweiss_symbol_list(symbol_list, driver=None):
    edelweiss_list = load_initial_data('list')
    ratio_data = load_initial_data('ratios')
    pnl_annual = load_initial_data('pnl_annual')
    balancesheet = load_initial_data('balancesheet')
    cashflow = load_initial_data('cashflow')
    peers = load_initial_data('peers')

    if driver is None:
        driver = webdriver.Firefox()
        driver.get('https://www.edelweiss.in/quotes/equity/Andhra-Cements-Ltd-19')

    delay = 30

    for symbol in symbol_list:
        try:
            try:
                search_box = driver.find_element_by_id('searchAnything')
                search_box.clear()
                search_box.send_keys(symbol)
                search_box.send_keys(Keys.ENTER)
                WebDriverWait(driver, delay).until(EC.element_to_be_clickable((By.CLASS_NAME, 'highcharts-container')))

            except TimeoutException:
                print("Loading took too much time for {0}".format(symbol))
                continue

            time.sleep(5)

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Update edelweiss_list with data from edelweiss site
            edelweiss_list.loc[symbol, :] = create_header_list(soup).rename(symbol)
            string_columns = [
                'edelweiss_url', 'company_name', 'sector',
                'bse_identifier', 'nse_identifier', 'isin_identifier'
            ]
            numeric_columns = [
                'mcap', 'pe_ratio', 'div_yield', 'beta',
                'promoter_holding', 'fii_holding',
                'dii_mf_holding', 'others_holding'
            ]
            edelweiss_list[string_columns] = edelweiss_list[string_columns].astype(str)
            edelweiss_list[numeric_columns] = edelweiss_list[numeric_columns].replace('NM', np.nan).replace('-', np.nan).replace('', np.nan)
            edelweiss_list[string_columns] = edelweiss_list[string_columns].astype(np.float)
            edelweiss_list.bullish = edelweiss_list.bullish.astype(bool)

            # Append ratio_data with ratio data for the current symbol
            ratio_df = load_ratios(soup)
            ratio_df.insert(0, 'symbol', symbol)
            ratio_data = ratio_data.append(ratio_df).reset_index(drop=True)

            yearly_pnl = load_pnl_static(soup)
            yearly_pnl.insert(0, 'symbol', symbol)
            yearly_pnl.insert(1, 'type', 'standalone')
            pnl_annual = pnl_annual.append(yearly_pnl).reset_index(drop=True)

            balance_sheet = load_balancesheet_static(soup)
            balance_sheet.insert(0, 'symbol', symbol)
            balance_sheet.insert(1, 'type', 'standalone')
            balancesheet = balancesheet.append(balance_sheet).reset_index(drop=True)

            cash_flow = load_cashflow(soup)
            cash_flow.insert(0, 'symbol', symbol)
            cashflow = cashflow.append(cash_flow).reset_index(drop=True)

            peer_comparision = load_peers_comparision(soup)
            peer_comparision.insert(0, 'symbol', symbol)
            peers = peers.append(peer_comparision).reset_index(drop=True)
        except Exception as e:

            print('failed for {0} due to {1}'.format(symbol, e))
            continue

        with SafeHDFStore('edelweiss.h5') as store:

            store.put('list', value=edelweiss_list)
            store.put('ratios', value=ratio_data)
            store.put('pnl_annual', value=pnl_annual)
            store.put('balancesheet', value=balancesheet)
            store.put('cashflow', value=cashflow)
            store.put('peers', value=peers)
            print('Successful for {0}'.format(symbol))
        clean_file('edelweiss.h5')


def adjust_data():
    edelweiss_list = load_initial_data('list')
    ratio_data = load_initial_data('ratios')
    pnl_annual = load_initial_data('pnl_annual')
    balancesheet = load_initial_data('balancesheet')
    cashflow = load_initial_data('cashflow')
    peers = load_initial_data('peers')

    sector = {
        'metals': 'metals',
        'information technology': 'it',
        'cement': 'cement',
        'power': 'power',
        'pharmaceuticals': 'pharma',
        'chemicals': 'chemicals',
        'textiles': 'textiles',
        'energy': 'energy',
        'engineering / capital goods': 'capital_goods',
        'retail': 'retail',
        'shipping': 'shipping',
        'diversified': 'diversified',
        'trading': 'trading',
        'construction': 'construction',
        'agro': 'agro',
        'finance': 'finance',
        'hotels': 'hotels',
        'logistics': 'logistics',
        'telecom': 'telecom',
        'real estate': 'real_estate',
        'paints': 'paints',
        'gems and jewellery': 'jewellery',
        'banks': 'banks',
        'auto ancillaries': 'auto_ancillaries',
        'packaging': 'packaging',
        'rubber': 'rubber',
        'healthcare': 'healthcare',
        'tyres': 'tyres',
        'building materials': 'construction_materials',
        'printing and stationery': 'priting',
        'plastics': 'plastics',
        'infrastructure': 'infra',
        'automobiles': 'auto',
        'beverages': 'beverages',
        'consumer products': 'consumer_products',
        'media': 'media',
        'fmcg': 'fmcg',
        'consumer durables': 'consumer_durables',
        'sugar': 'sugar',
        '': 'unknown',
        'paper': 'paper',
        'petrochemicals': 'petro',
        'hospitality': 'hospitality',
        'airlines': 'airlines'
    }
    edelweiss_list = edelweiss_list.dropna(subset=['edelweiss_url'])
    edelweiss_list.sector = edelweiss_list.sector.fillna('unknown').str.lower().replace(sector)
    edelweiss_list.fii_holding = edelweiss_list.fii_holding.fillna(0).astype(int)
    edelweiss_list.dii_mf_holding = edelweiss_list.dii_mf_holding.fillna(0).astype(int)
    edelweiss_list.others_holding = edelweiss_list.others_holding.fillna(0).astype(int)
    edelweiss_list.promoter_holding = edelweiss_list.promoter_holding.fillna(0).astype(int)
    edelweiss_list.pe_ratio = edelweiss_list.pe_ratio.replace('NM', np.nan).replace('', np.nan).fillna(0)
    edelweiss_list.bullish = edelweiss_list.bullish.str.lower().replace('bullish', True).fillna(False)
    edelweiss_list.nse_identifier = edelweiss_list.nse_identifier.str.lower()
    edelweiss_list = edelweiss_list.convert_objects(convert_numeric=True)
    edelweiss_list = edelweiss_list[[
        'edelweiss_url', 'company_name', 'sector',
        'pe_ratio', 'div_yield', 'mcap', 'beta', 'bullish',
        'bse_identifier', 'nse_identifier', 'isin_identifier',
        'promoter_holding', 'fii_holding', 'dii_mf_holding', 'others_holding'
    ]]

    ratio_data = ratio_data.sort_values(['symbol', 'ratio_type', 'ratio_name'])
    if ratio_data.columns[0] != 'symbol':
        ratio_data = ratio_data[list(reversed(ratio_data.columns))]
    ratio_data.FY11 = ratio_data.FY11.fillna(ratio_data['Unnamed: 5'])
    ratio_data = ratio_data.drop(['Unnamed: 5'], axis=1)
    ratio_data = ratio_data.reset_index(drop=True)

    pnl_annual = pnl_annual.sort_values(['symbol', 'type', 'index'])
    if pnl_annual.columns[0] != 'symbol':
        pnl_annual = pnl_annual[list(reversed(pnl_annual.columns))]
    pnl_annual.FY11 = pnl_annual.FY11.fillna(pnl_annual['Unnamed: 5'])
    pnl_annual = pnl_annual.drop(['Unnamed: 5'], axis=1)
    col_order = np.array(pnl_annual.columns)
    col_order[0], col_order[1] = col_order[1], col_order[0]
    pnl_annual = pnl_annual[col_order]
    pnl_annual = pnl_annual.reset_index(drop=True)

    balancesheet = balancesheet.drop(balancesheet[balancesheet['Share Capital'].notnull()].index)
    balancesheet = balancesheet.drop(['Share Capital'], axis=1)
    balancesheet = balancesheet.sort_values(['symbol', 'type', 'index'])
    if balancesheet.columns[0] != 'symbol':
        balancesheet = balancesheet[list(reversed(balancesheet.columns))]
    balancesheet.FY11 = balancesheet.FY11.fillna(balancesheet['Unnamed: 5'])
    balancesheet = balancesheet.drop(['Unnamed: 5'], axis=1)
    balancesheet = balancesheet.reset_index(drop=True)
    col_order = np.array(balancesheet.columns)
    col_order[0], col_order[1] = col_order[1], col_order[0]
    balancesheet = balancesheet[col_order]

    cashflow.symbol.unique()
    cashflow = cashflow.dropna(subset=['FY02', 'FY03', 'FY04', 'FY05',
                                       'FY06', 'FY07', 'FY08', 'FY09',
                                       'FY10', 'FY11', 'FY12', 'FY13',
                                       'FY14', 'FY15', 'FY16'], how='all')
    if cashflow.columns[0] != 'symbol':
        cashflow = cashflow[list(reversed(cashflow.columns))]
    cashflow = cashflow[['symbol', 'cf_type', 'cf_activity',
                         'FY16', 'FY15', 'FY14', 'FY13',
                         'FY12', 'FY11', 'FY10', 'FY09',
                         'FY08', 'FY07', 'FY06', 'FY05',
                         'FY04', 'FY03', 'FY02']]
    cashflow = cashflow.sort_values(['symbol', 'cf_type', 'cf_activity'])
    cashflow = cashflow.reset_index(drop=True)
