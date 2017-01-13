import symbol as s
import pandas as pd
import numpy as np


def main():

    # returns = symbol.get_returns()
    # print(returns.head())
    # symbol = s.Symbol('edelweiss')
    # sreturns = symbol.get_symbol_returns(load_null=True)
    # index = s.Index('Nifty 50')

    # index = s.Index(index_list='nifty_50')
    # idata = index.get_index_hist()
    # print(len(idata))
    sym = s.Symbol()
    ret = sym.get_symbol_returns(start='2016', end='2017', null_count=5, volume=1000)
    print(len(ret.columns))
    print(ret.tail())
    # cap = s.Portifolio()
    # beta = cap.capm_symbol_to_index(periods=[1, 3, 5, 10, 15, 20])
    # symbol = s.Symbol(index='nifty_500', start=2010, min_rows=4000, null_count=2)
    # symbol.get_symbol_meta(force_load=True)
    # print(len(symbol.symbol_list))
    # print((symbol.symbol_list).max())
    # symbol.get_symbol_meta(force_load=True)
    # symbol_meta = sym.symbol_meta
    # beta = pd.read_hdf('beta.h5', 'beta_all')
    # sector_indices = s.INDEX_META[s.INDEX_META.category == 'sec_index'].index
    # print(beta.head())
    # print(beta[(beta.r_square >= 0.7) & (beta.p_value <= 0.05)])
    # print(beta[(beta.symbol == 'tcs') & (beta.benchmark == 'nifty_it')])

    # def something(symb):
    #     benchmark = symb.benchmark
    #     symbol = symb.symbol
    #     if symbol in symbol_meta.index[symbol_meta[benchmark]]:
    #         symb.symbol = np.nan
    #     return symb
    # print(beta.apply(something, axis=1).dropna())
    # print(symbol_meta[symbol_meta[beta.benchmark.ix[0]]])
    # print(beta[beta.symbol.isin(symbol_meta.index[symbol_meta[beta.benchmark]])])
    print()


if __name__ == '__main__':
    main()
