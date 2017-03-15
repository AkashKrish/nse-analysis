'''Module main'''

from nse import NSE

nse = NSE()

# nse.force_load_data('symbol_eod_data')
ret = nse.get_symbol_eod_values()
print(len(ret), len(ret.columns))