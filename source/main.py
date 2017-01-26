from symbol import Symbol
from index import Index
from portifolio import Portifolio
from techno import *


def main():
    symbol_list = ['ashokley', 'motilalofs', 'infy', 'edelweiss', 'federalbnk']
    port = Portifolio(symbol_list=symbol_list, benchmark=['nifty_50', 'nifty_500'], start='2016')

    data = port.get_symbol_hist('infy').set_index('date')
    EMA(data, 5)
    # pret = port.get_portifolio_returns().fillna(0)
    # iret = port.get_index_returns().fillna(0)
    # print(pret.resample('A').sum().head() , port.set_risk_free_rate(pret.resample('M').sum()).head())
    # print((port.beta(pret['2016'], iret['2016'])))
    # print(port.alpha(pret['2016'], iret['2016']))
    # print(port.basic_regression(pret.value['2016'], iret.nifty_500['2016']))
    # print(port.calculate_capm(pret['2016'], iret['2016']).round(4))
    # print(port.calculate_capm_frequency(pret.value['2016'], iret.nifty_500['2016'], frequency='M'))
    # print(port.calcluate_capm_interval(pret['2016'], iret.nifty_500['2016']))
    # print(port.lower_partial_moment(pret['2016']))
    # print(port.treynor_ratio(pret['2016'], iret['2016']))
    # print(port.sharpe_ratio(pret['2016']))
    # print(port.information_ratio(pret['2016'], iret['2016']))
    # print(port.modigliani_ratio(pret['2016'], iret['2016']))
    # print(port.omega_ratio(pret['2016']))
    # print(port.sortino_ratio(pret['2016']))
    # print(port.kappa_three_ratio(pret['2016']))
    # print(port.gain_loss_ratio(pret['2016']))
    # print(port.upside_potential_ratio(pret['2016']))
    # returns = symbol.get_returns()
    # print(returns.head())
    # symbol = s.Symbol('edelweiss')
    # sreturns = symbol.get_symbol_returns(load_null=True)
    # index = s.Index('Nifty 50')

    # index = s.Index(index_list='nifty_50')
    # idata = index.get_index_hist()
    # print(len(idata))
    # print(len(sym.get_symbol_meta()))
    # print(len(sym.get_symbol_list()))
    # print(sym.get_s5ymbol_hist().head())
    # ret = sym.get_symbol_returns(start='2016', end='2017', null_count=5, volume=1000)
    # print(len(ret.columns))
    # print(ret.tail())
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
