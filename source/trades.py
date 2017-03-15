'''Module for handling ZeroDha trades'''

import os

import pandas as pd
import numpy as np
from nse import NSE

from helpers import get_adjacent_dates, get_store_keys


class Zerodha(NSE):
    '''Module for handling ZeroDha trades'''
    CURRENT_PATH = os.path.dirname(__file__)
    TRADES_PATH = os.path.join(CURRENT_PATH, 'data{0}tradebook.xlsx'.format(os.sep))
    TRADES_DATA_PATH = os.path.join(CURRENT_PATH, 'data{0}zerodha.h5'.format(os.sep))

    TRADES_KEY = 'trades'
    INVESTMENTS_KEY = 'investments'
    PROFITS_KEY = 'profits'
    PROFITS_PER_DAY_KEY = 'profits_per_day'
    HOLDINGS_KEY = 'holdings'
    QUANTITY_KEY = 'quantity'
    RETURNS_KEY = 'returns'

    @classmethod
    def read_trades(cls, path=None):
        '''Read trades data from path and return pivot tables'''
        if path is None:
            path = Zerodha.TRADES_PATH
        trades = pd.read_excel(path, skiprows=11, parse_cols='B:J')

        # Basic renaming and slicing columns
        trades.columns = [
            'date', 'time', 'exchange', 'symbol',
            'type', 'trade_qty', 'trade_rate',
            'order_no', 'trade_no'
        ]
        trades.insert(2, 'datetime', trades.date + ' ' +trades.time.astype(str))
        trades.drop(['time', 'exchange'], axis=1, inplace=True)
        trades = trades[[
            'symbol', 'datetime', 'type',
            'trade_qty', 'trade_rate', 'date', 'trade_no'
        ]]

        # Handling column data and datatypes
        trades.date = pd.to_datetime(trades.date, format='%d-%m-%Y')
        trades.datetime = pd.to_datetime(trades.datetime, format='%d-%m-%Y %H:%M:%S')
        trades.symbol = trades.symbol.str.lower()
        trades.type = trades.type.apply(lambda x: 'buy' if x == 'B' else 'sell')
        trades['trade_type_multiplier'] = trades.type.apply(lambda x: 1 if x == 'buy' else -1)
        trades.trade_qty = trades.trade_qty * trades.trade_type_multiplier
        trades = trades.drop(['trade_type_multiplier'], axis=1)
        trades['cash_flow'] = -1 * trades['trade_rate'] * trades['trade_qty']

        # Handle Intra trades
        # TODO Improve intra section if possible
        intra_trades = trades.groupby(['date', 'symbol']).aggregate({
            'trade_qty': 'sum',
            'cash_flow': 'sum',
            'trade_no': 'unique'
        }).reset_index()
        intra_trades = intra_trades.query('trade_qty == 0')
        intra_trade_nos = (intra_trades.trade_no.apply(pd.Series).values).flatten()

        trades.loc[trades.trade_no.isin(intra_trade_nos), 'trade_type'] = ['intra' for trade in range(len(intra_trade_nos))]
        trades['trade_type'] = trades['trade_type'].fillna('inter')
        trades = trades.sort_values(['symbol', 'datetime']).reset_index(drop=True)

        # Calculate total qty till that trade
        for symbol in trades.symbol.unique():
            symbol_filter = trades.symbol == symbol
            symbol_trades = trades.loc[symbol_filter].copy()
            symbol_trades['total_qty'] = symbol_trades['trade_qty'].cumsum()
            trades.loc[trades.trade_no.isin(symbol_trades.trade_no), 'total_qty'] = symbol_trades['total_qty']
        trades = trades.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        return trades

    def get_trades(self, trades_type='all', path=None):
        '''
        Get trades.
        trades_type: (all, intra, inter)
        path: path of file for trades
        '''
        if path is not None:
            trades = self.read_trades(path=path)
        elif Zerodha.TRADES_KEY in get_store_keys(Zerodha.TRADES_DATA_PATH):
            trades = pd.read_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.TRADES_KEY)
        else:
            raise AttributeError('Data not avaliable')
        if trades_type != 'all':
            trades = trades.query('trade_type == @trades_type')

        return trades

    def get_investments(self, trades=None):
        '''
        Get investments in symbols based on trades.
        If trades is none, Local stored trades is used.
        '''
        if trades is None and Zerodha.INVESTMENTS_KEY in get_store_keys(Zerodha.TRADES_DATA_PATH):
            investments = pd.read_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.INVESTMENTS_KEY)
            return investments
        elif trades is None:
            trades = self.get_trades(trades_type='all')

        investments = pd.DataFrame(
            0, index=self.get_traded_dates(start=trades.date.min()).index,
            columns=trades.symbol.unique()
        )
        for trade in trades.itertuples():
            symbol = trade.symbol
            date = trade.date
            rate = trade.trade_rate
            qty = trade.trade_qty
            if trade.total_qty == 0:
                investments.loc[date:, symbol] = 0
            else:
                investments.loc[date:, symbol] = investments.loc[date:, symbol] + (qty * rate)
        return investments

    def get_quantity(self, trades=None):
        '''
        Get quantity of symbols based on trades.
        If trades is none, Local stored trades is used.
        '''
        if trades is None and Zerodha.QUANTITY_KEY in get_store_keys(Zerodha.TRADES_DATA_PATH):
            quantity = pd.read_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.QUANTITY_KEY)
            return quantity
        elif trades is None:
            trades = self.get_trades(trades_type='all')

        quantity = pd.DataFrame(
            0, index=self.get_traded_dates(start=trades.date.min()).index,
            columns=trades.symbol.unique()
        )
        for trade in trades.itertuples():
            symbol = trade.symbol
            date = trade.date
            total_qty = trade.total_qty
            quantity.loc[date:, symbol] = total_qty
        return quantity

    def get_holdings(self, trades=None):
        '''
        Get holdings of symbols based on trades.
        If trades is none, Local stored trades is used.
        '''
        if trades is None and Zerodha.HOLDINGS_KEY in get_store_keys(Zerodha.TRADES_DATA_PATH):
            holdings = pd.read_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.HOLDINGS_KEY)
            return holdings
        elif trades is None:
            trades = self.get_trades(trades_type='all')

        holdings = pd.DataFrame(
            0, index=self.get_traded_dates(start=trades.date.min()).index,
            columns=trades.symbol.unique()
        )
        close = self.get_symbol_eod_values(data='close')
        for trade in trades.itertuples():
            symbol = trade.symbol
            date = trade.date
            total_qty = trade.total_qty
            if symbol not in close.columns:
                holdings[symbol] = np.nan
                continue
            if trade.total_qty == 0:
                holdings.loc[date:, symbol] = 0
            else:
                holdings.loc[date:, symbol] = total_qty * close[symbol][date:]
        return holdings

    def get_profits(self, trades=None, trades_type='all'):
        '''
        Get overall profits of symbols based on trades.
        If trades is none, Local stored trades is used.
        '''
        if trades is None and Zerodha.PROFITS_KEY in get_store_keys(Zerodha.TRADES_DATA_PATH):
            profits = pd.read_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.PROFITS_KEY)
            return profits
        elif trades is None:
            trades = self.get_trades(trades_type=trades_type)

        profits = pd.DataFrame(
            0, index=self.get_traded_dates(start=trades.date.min()).index,
            columns=trades.symbol.unique()
        )
        investments = self.get_investments(trades=trades)
        holdings = self.get_holdings(trades=trades)

        intra_trades, inter_trades = trades.query('trade_type == "intra"'), trades.query('trade_type == "inter"')
        for trade in inter_trades.itertuples():
            symbol = trade.symbol
            date = trade.date
            profits.loc[date:, symbol] = (
                holdings.loc[date:, symbol] - investments.loc[date:, symbol]
            )

        for trade in intra_trades.itertuples():
            profits.loc[trade.date, trade.symbol] = profits.loc[trade.date, trade.symbol] + trade.cash_flow
        return profits

    def get_profits_per_day(self, trades=None, trades_type='all'):
        '''
        Get overall profits per day of symbols based on trades.
        If trades is none, Local stored trades is used.
        '''
        if trades is None and Zerodha.PROFITS_PER_DAY_KEY in get_store_keys(Zerodha.TRADES_DATA_PATH):
            profits_per_day = pd.read_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.PROFITS_PER_DAY_KEY)
            return profits_per_day
        elif trades is None:
            trades = self.get_trades(trades_type=trades_type)

        profits_per_day = pd.DataFrame(
            0, index=self.get_traded_dates(start=trades.date.min()).index,
            columns=trades.symbol.unique()
        )
        profits = self.get_profits(trades=trades, trades_type=trades_type)

        for trade in trades.itertuples():
            symbol = trade.symbol
            date = trade.date
            profits_per_day.loc[date:, symbol] = (
                profits.loc[date:, symbol] - profits.loc[date:, symbol].shift(1)
            )
            profits_per_day.loc[date, symbol] = profits.loc[date, symbol]

        return profits_per_day

    def get_returns(self, trades=None):
        '''
        Get overall returns per share of symbols based on trades.
        If trades is none, Local stored trades is used.
        '''
        if trades is None and Zerodha.RETURNS_KEY in get_store_keys(Zerodha.TRADES_DATA_PATH):
            returns = pd.read_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.RETURNS_KEY)
            return returns
        elif trades is None:
            trades = self.get_trades(trades_type='inter')

        returns = pd.DataFrame(
            index=self.get_traded_dates(start=trades.date.min()).index,
            columns=trades.symbol.unique()
        )
        investments = self.get_investments(trades=trades)
        holdings = self.get_holdings(trades=trades)
        close = self.get_symbol_eod_values(data='close')

        inter_trades = trades.query('trade_type == "inter"')
        for trade in inter_trades.itertuples():
            symbol = trade.symbol
            date = trade.date
            rate = trade.trade_rate
            total_qty = trade.total_qty
            previous_date, next_date = get_adjacent_dates(
                index=returns.index, date=date
            )

            returns.loc[date:, symbol] = np.log(
                holdings.loc[date:, symbol] / holdings.loc[date:, symbol].shift(1)
            )
            if trade.type == 'buy':
                returns.loc[date, symbol] = np.log(
                    holdings.loc[date, symbol] / investments.loc[date, symbol]
                )
            elif previous_date is not None:
                returns.loc[date, symbol] = np.log(rate / close.loc[previous_date, symbol])

            if total_qty == 0:
                returns.loc[next_date:, symbol] = np.nan
        return returns.astype(np.float)

    def force_load_trades_data(self, path=None):
        '''
        Force loading helper method for saving trades data to local HDFStores
        '''

        trades = self.read_trades(path=path)
        trades.to_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.TRADES_KEY)

        investments = self.get_investments(trades=trades)
        investments.to_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.INVESTMENTS_KEY)

        quantity = self.get_quantity(trades=trades)
        quantity.to_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.QUANTITY_KEY)

        holdings = self.get_holdings(trades=trades)
        holdings.to_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.HOLDINGS_KEY)

        profits = self.get_profits(trades=trades, trades_type='all')
        profits.to_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.PROFITS_KEY)

        profits_per_day = self.get_profits_per_day(trades=trades, trades_type='all')
        profits_per_day.to_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.PROFITS_PER_DAY_KEY)

        returns = self.get_returns(trades=trades)
        returns.to_hdf(Zerodha.TRADES_DATA_PATH, Zerodha.RETURNS_KEY)
