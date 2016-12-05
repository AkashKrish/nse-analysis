import symbol
import pandas as pd


def main():
    symbol_meta = symbol.get_symbol_meta()
    symbol_list = pd.DataFrame(symbol_meta.date_of_listing)
    symbol_data_daily = symbol.get_hist_data(symbol_list.copy())

    print(symbol_data_daily[symbol_data_daily.symbol == '3IINFOTECH'].tail())


if __name__ == '__main__':
    main()
