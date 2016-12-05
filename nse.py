from datetime import datetime

# import pandas as pd
# from dateutil.relativedelta import relativedelta
from nsepy import get_history

import symbol as s

symbol_meta = s.load_symbol_meta()
symbol_list = symbol_meta.symbol.copy()

symbol = symbol_meta.ix[1478, :]
from_date = symbol.date_of_listing
to_date = datetime.today().date()
symbol = symbol.symbol
nse_data = get_history(symbol=symbol, start=from_date, end=to_date)
print(nse_data.head())