'''
Created on April 15, 2012
Last update on July 18, 2015

@author: Bruno Franca
@author: Peter Bakker
@author: Femto Trader
'''
import pandas as pd
import numpy as np


class Columns(object):
    OPEN = 'open'
    high = 'high'
    low = 'low'
    close = 'close'
    VOLUME = 'volume'


def get(data, col):
    return(data[col])


# price = COL.close


indicators = [
    "MA", "EMA", "MOM", "ROC", "ATR", "BBANDS", "PPSR", "STOK", "STO",
    "TRIX", "ADX", "MACD", "MassI", "Vortex", "KST", "RSI", "TSI", "ACCDIST",
    "Chaikin", "MFI", "OBV", "FORCE", "EOM", "CCI", "COPP", "KELCH", "ULTOSC",
    "DONCH", "STDDEV"
]


class Settings(object):
    join = True
    col = Columns()


SETTINGS = Settings()


def out(settings, data, result):
    if not settings.join:
        return result
    else:
        data = data.join(result)
        return data


def MA(data, n, price='close'):
    """
    Moving Average
    """
    MA = data[price].rolling(window=n).mean().rename('MA_{n}'.format(n=n))
    return out(SETTINGS, data, MA)


def MSD(data, n, price='close'):
    """
    Moving Average
    """
    MSD = data[price].rolling(window=n).std().rename('MSD{n}'.format(n=n))
    return out(SETTINGS, data, MSD)


def EMA(data, n, price='close'):
    """
    Exponential Moving Average
    """
    EMA = data[price].ewm(span=n, min_periods=n - 1).mean().rename('EMA_{n}'.format(n=n))
    return out(SETTINGS, data, EMA)


def MOM(data, n, price='close'):
    """
    Momentum
    """
    MOM = data[price].diff(n).rename('MOM_{n}'.format(n=n))
    return out(SETTINGS, data, MOM)


def ROC(data, n, price='close'):
    """
    Rate of Change
    """
    M = data[price].diff(n - 1)
    N = data[price].shift(n - 1)
    result = pd.Series(M / N, name='ROC_{n}'.format(n=n))
    return out(SETTINGS, data, result)


def TR(data):
    tr = pd.DataFrame()
    tr['TR1'] = abs(data['high'] - data['low'])
    tr['TR2'] = abs(data['high'] - data['prev_close'])
    tr['TR3'] = abs(data['low'] - data['prev_close'])
    tr['TrueRange'] = tr.max(axis=1)
    return out(SETTINGS, data, tr)


def ATR(data, period=14):
    tr = TR(data)['TrueRange']
    atr = tr.ewm(span=period, min_periods=period).mean()
    atr = atr.rename('ATR_{0}'.format(period))
    return out(SETTINGS, data, atr)


def BBANDS(data, n, price='close'):
    """
    Bollinger Bands
    """
    MA = data[price].rolling(n).mean()
    MSD = data[price].rolling(n).std()
    B1 = pd.Series(MA + 2 * MSD, name='Bollinger_UB_{n}'.format(n=n), index=data.index)
    B2 = pd.Series(MA - 2 * MSD, name='Bollinger_LB_{n}'.format(n=n), index=data.index)
    result = pd.DataFrame([B1, B2]).transpose()
    return out(SETTINGS, data, result)


def PPSR(data):
    """
    Pivot Points, Supports and Resistances
    """
    PP = pd.Series((data['high'] + data['low'] + data['close']) / 3, name='PP')
    R1 = pd.Series(2 * PP - data['low'], name='R1')
    S1 = pd.Series(2 * PP - data['high'], name='S1')
    R2 = pd.Series(PP + data['high'] - data['low'], name='R2')
    S2 = pd.Series(PP - data['high'] + data['low'], name='S2')
    R3 = pd.Series(data['high'] + 2 * (PP - data['low']), name='R3')
    S3 = pd.Series(data['low'] - 2 * (data['high'] - PP), name='S3')
    result = pd.DataFrame([S3, S2, S1, PP, R1, R2, R3]).transpose()
    return out(SETTINGS, data, result)


def STOK(data, period=14):
    """
    Stochastic oscillator %K
    """
    H = data['high'].rolling(period).max()
    L = data['low'].rolling(period).min()
    k = (
        100 * ((data['close'] - L) / (H - L))
    )
    result = k.rename('STOK_FAST_{0}'.format(period)).round(2)
    return out(SETTINGS, data, result)


def STOD(data, period=14, smoothing=3):
    """
    Stochastic oscillator %D
    """
    SOk = STOK(data, period)['STOK_FAST_{0}'.format(period)]
    result = SOk.rolling(smoothing, min_periods=smoothing - 1).mean()
    result = result.rename('STOD_FAST_{0}'.format(period)).round(2)
    return out(SETTINGS, data, result)


def STOKD(data, period=14, smoothing=3):
    """
    Stochastic oscillator %D
    """
    SOk = STOK(data, period)['STOK_FAST_{0}'.format(period)]
    SOd = SOk.rolling(smoothing, min_periods=smoothing - 1).mean()
    SOd = SOd.rename('STOD_FAST_{0}'.format(period))
    result = pd.DataFrame([SOk, SOd]).transpose().round(2)
    return out(SETTINGS, data, result)


def STOKD_SLOW(data, period=14):
    stokd_fast = STOKD(data, period, 3)
    SOk_slow = stokd_fast['STOK_FAST_{0}'.format(period)]
    SOk_slow = SOk_slow.rolling(3).mean()
    SOk_slow = SOk_slow.rename('STOK_SLOW_{0}'.format(period))
    SOd_slow = SOk_slow.rolling(period, min_periods=period - 1).mean()
    SOd_slow = SOd_slow.rename('STOD_SLOW_{0}'.format(period))
    result = pd.DataFrame([SOk_slow, SOd_slow]).transpose().round(2)
    return out(SETTINGS, data, result)


def STOKD_FULL(data, period=14):
    stokd_fast = STOKD(data, period, 3)
    SOk_full = stokd_fast['STOK_FAST_{0}'.format(period)]
    SOk_full = SOk_full.rolling(period).mean()
    SOk_full = SOk_full.rename('STOK_SLOW_{0}'.format(period))
    SOd_full = SOk_full.rolling(period, min_periods=period - 1).mean()
    SOd_full = SOd_full.rename('STOD_SLOW_{0}'.format(period))
    result = pd.DataFrame([SOk_full, SOd_full]).transpose().round(2)
    return out(SETTINGS, data, result)


def TRIX(data, period=14):
    """
    Trix
    """
    ema1 = data['close'].ewm(span=period, min_periods=period - 1).mean()
    ema1 = ema1.rename('EMA_{0}'.format(period))
    ema2 = ema1.ewm(span=period, min_periods=period - 1).mean()
    ema2 = ema2.rename('EMA2_{0}'.format(period))
    ema3 = ema2.ewm(span=period, min_periods=period - 1).mean()
    ema3 = ema3.rename('EMA2_{0}'.format(period))

    trix = ema3.pct_change() * 100
    trix = trix.rename('TRIX_{0}'.format(period))
    result = pd.DataFrame([ema1, ema2, ema3, trix]).T

    return out(SETTINGS, data, result)


def DM(data, period=14):

    high = data['high']
    low = data['low']

    moveup = high - high.shift(1)
    movedown = low.shift(1) - low

    pdm_condition = (moveup > movedown) & (moveup > 0)
    pdm = pdm_condition.replace(True, np.nan).fillna(moveup)
    pdm = pdm.ewm(span=period, min_periods=period - 1).mean()
    pdm = pdm.rename('PDM_{0}'.format(period))

    ndm_condition = (movedown > moveup) & (movedown > 0)
    ndm = ndm_condition.replace(True, np.nan).fillna(movedown)
    ndm = ndm.ewm(span=period, min_periods=period - 1).mean()
    ndm = ndm.rename('NDM_{0}'.format(period))

    dm = pd.DataFrame([pdm, ndm]).T
    return out(SETTINGS, data, dm)


def DI(data, period=14):
    dm = DM(data, period)[['PDM_{0}'.format(period), 'NDM_{0}'.format(period)]]
    dm.columns = ['pdm', 'ndm']
    atr = ATR(data, period)['ATR_{0}'.format(period)]

    pdi = 100 * (dm.pdm / atr)
    pdi = pdi.rename('PDI_{0}'.format(period))

    ndi = 100 * (dm.ndm / atr)
    ndi = ndi.rename('NDI_{0}'.format(period))

    di = pd.DataFrame([pdi, ndi]).T
    return out(SETTINGS, data, di)


def ADX(data, period=14):
    di = DI(data, period)[['PDI_{0}'.format(period), 'NDI_{0}'.format(period)]]
    di.columns = ['pdi', 'ndi']

    dx = 100 * (
        (di.pdi - di.ndi).abs() / (di.pdi + di.ndi)
    )

    adx = dx.ewm(span=period, min_periods=period - 1).mean()
    adx = pd.DataFrame([di.pdi, di.ndi, adx]).T
    adx.columns = [
        'PDI_{0}'.format(period), 'NDI_{0}'.format(period),
        'ADX_{0}'.format(period)
    ]
    return adx


def MACD(data, n_fast=12, n_slow=26, signal=9, price='close'):
    """
    MACD, MACD Signal and MACD difference
    """
    price = data[price]
    EMAfast = price.ewm(span=n_fast, min_periods=n_fast - 1).mean()
    EMAslow = price.ewm(span=n_slow, min_periods=n_slow - 1).mean()
    macd = EMAfast - EMAslow
    macd = macd.rename('MACD_{0}_{1}'.format(n_fast, n_slow))
    macd_signal = macd.ewm(span=signal, min_periods=signal - 1).mean()
    macd_signal = macd_signal.rename('MACD_signal_{0}_{1}'.format(n_fast, n_slow))
    macd_hist = macd - macd_signal
    macd_hist = macd_hist.rename('MACD_hist_{0}_{1}'.format(n_fast, n_slow))
    result = pd.DataFrame([macd, macd_signal, macd_hist]).transpose()
    return out(SETTINGS, data, result)


def PPO(data, n_fast=12, n_slow=26, signal=9, price='close'):
    """
    Percentage Price Oscillator
    """
    price = data[price]
    EMAfast = price.ewm(span=n_fast, min_periods=n_fast - 1).mean()
    EMAslow = price.ewm(span=n_slow, min_periods=n_slow - 1).mean()
    ppo = (EMAfast - EMAslow) / EMAslow
    ppo = ppo.rename('PPO_{0}_{1}'.format(n_fast, n_slow))
    ppo_signal = ppo.ewm(span=signal, min_periods=signal - 1).mean()
    ppo_signal = ppo_signal.rename('PPO_signal_{0}_{1}'.format(n_fast, n_slow))
    ppo_hist = ppo - ppo_signal
    ppo_hist = ppo_hist.rename('PPO_hist_{0}_{1}'.format(n_fast, n_slow))
    result = pd.DataFrame([ppo, ppo_signal, ppo_hist]).transpose()
    return out(SETTINGS, data, result)


# def MassI(data):
#     """
#     Mass Index
#     """
#     Range = data['high'] - data['low']
#     EX1 = pd.ewma(Range, span=9, min_periods=8)
#     EX2 = pd.ewma(EX1, span=9, min_periods=8)
#     Mass = EX1 / EX2
#     result = pd.Series(pd.rolling_sum(Mass, 25), name='Mass Index')
#     return out(SETTINGS, data, result)


# def Vortex(data, n):
#     """
#     Vortex Indicator
#     """
#     i = 0
#     TR = [0]
#     while i < len(data) - 1:  # data.index[-1]:
#         Range = max(data.get_value(i + 1, 'high'), data.get_value(i, 'close')) - min(data.get_value(i + 1, 'low'), data.get_value(i, 'close'))
#         TR.append(Range)
#         i = i + 1
#     i = 0
#     VM = [0]
#     while i < len(data) - 1:  # data.index[-1]:
#         Range = abs(data.get_value(i + 1, 'high') - data.get_value(i, 'low')) - abs(data.get_value(i + 1, 'low') - data.get_value(i, 'high'))
#         VM.append(Range)
#         i = i + 1
#     result = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name='Vortex_' + str(n))
#     return out(SETTINGS, data, result)


# def KST(data, r1, r2, r3, r4, n1, n2, n3, n4):
#     """
#     KST Oscillator
#     """
#     M = data['close'].diff(r1 - 1)
#     N = data['close'].shift(r1 - 1)
#     ROC1 = M / N
#     M = data['close'].diff(r2 - 1)
#     N = data['close'].shift(r2 - 1)
#     ROC2 = M / N
#     M = data['close'].diff(r3 - 1)
#     N = data['close'].shift(r3 - 1)
#     ROC3 = M / N
#     M = data['close'].diff(r4 - 1)
#     N = data['close'].shift(r4 - 1)
#     ROC4 = M / N
#     result = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name='KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
#     return out(SETTINGS, data, result)


# def RSI(data, n):
#     """
#     Relative Strength Index
#     """
#     i = 0
#     UpI = [0]
#     DoI = [0]
#     while i + 1 <= len(data) - 1:  # data.index[-1]
#         UpMove = data.get_value(i + 1, 'high') - data.get_value(i, 'high')
#         DoMove = data.get_value(i, 'low') - data.get_value(i + 1, 'low')
#         if UpMove > DoMove and UpMove > 0:
#             UpD = UpMove
#         else:
#             UpD = 0
#         UpI.append(UpD)
#         if DoMove > UpMove and DoMove > 0:
#             DoD = DoMove
#         else:
#             DoD = 0
#         DoI.append(DoD)
#         i = i + 1
#     UpI = pd.Series(UpI)
#     DoI = pd.Series(DoI)
#     PosDI = pd.Series(pd.ewma(UpI, span=n, min_periods=n - 1))
#     NegDI = pd.Series(pd.ewma(DoI, span=n, min_periods=n - 1))
#     result = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
#     return out(SETTINGS, data, result)


# def TSI(data, r, s):
#     """
#     True Strength Index
#     """
#     M = pd.Series(data['close'].diff(1))
#     aM = abs(M)
#     EMA1 = pd.Series(pd.ewma(M, span=r, min_periods=r - 1))
#     aEMA1 = pd.Series(pd.ewma(aM, span=r, min_periods=r - 1))
#     EMA2 = pd.Series(pd.ewma(EMA1, span=s, min_periods=s - 1))
#     aEMA2 = pd.Series(pd.ewma(aEMA1, span=s, min_periods=s - 1))
#     result = pd.Series(EMA2 / aEMA2, name='TSI_' + str(r) + '_' + str(s))
#     return out(SETTINGS, data, result)


# def ACCDIST(data, n):
#     """
#     Accumulation/Distribution
#     """
#     ad = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low']) * data['Volume']
#     M = ad.diff(n - 1)
#     N = ad.shift(n - 1)
#     ROC = M / N
#     result = pd.Series(ROC, name='Acc/Dist_ROC_' + str(n))
#     return out(SETTINGS, data, result)


# def Chaikin(data):
#     """
#     Chaikin Oscillator
#     """
#     ad = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low']) * data['Volume']
#     result = pd.Series(pd.ewma(ad, span=3, min_periods=2) - pd.ewma(ad, span=10, min_periods=9), name='Chaikin')
#     return out(SETTINGS, data, result)


# def MFI(data, n):
#     """
#     Money Flow Index and Ratio
#     """
#     PP = (data['high'] + data['low'] + data['close']) / 3
#     i = 0
#     PosMF = [0]
#     while i < len(data) - 1:  # data.index[-1]:
#         if PP[i + 1] > PP[i]:
#             PosMF.append(PP[i + 1] * data.get_value(i + 1, 'Volume'))
#         else:
#             PosMF.append(0)
#         i=i + 1
#     PosMF = pd.Series(PosMF)
#     TotMF = PP * data['Volume']
#     MFR = pd.Series(PosMF / TotMF)
#     result = pd.Series(pd.rolling_mean(MFR, n), name='MFI_' + str(n))
#     return out(SETTINGS, data, result)


# def OBV(data, n):
#     """
#     On-balance Volume
#     """
#     i = 0
#     OBV = [0]
#     while i < len(data) - 1:  # data.index[-1]:
#         if data.get_value(i + 1, 'close') - data.get_value(i, 'close') > 0:
#             OBV.append(data.get_value(i + 1, 'Volume'))
#         if data.get_value(i + 1, 'close') - data.get_value(i, 'close') == 0:
#             OBV.append(0)
#         if data.get_value(i + 1, 'close') - data.get_value(i, 'close') < 0:
#             OBV.append(-data.get_value(i + 1, 'Volume'))
#         i = i + 1
#     OBV = pd.Series(OBV)
#     result = pd.Series(pd.rolling_mean(OBV, n), name='OBV_' + str(n))
#     return out(SETTINGS, data, result)


# def FORCE(data, n):
#     """
#     Force Index
#     """
#     result = pd.Series(data['close'].diff(n) * data['Volume'].diff(n), name='Force_' + str(n))
#     return out(SETTINGS, data, result)


# def EOM(data, n):
#     """
#     Ease of Movement
#     """
#     EoM = (data['high'].diff(1) + data['low'].diff(1)) * (data['high'] - data['low']) / (2 * data['Volume'])
#     result = pd.Series(pd.rolling_mean(EoM, n), name='EoM_' + str(n))
#     return out(SETTINGS, data, result)


# def CCI(data, n):
#     """
#     Commodity Channel Index
#     """
#     PP = (data['high'] + data['low'] + data['close']) / 3
#     result = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name='CCI_' + str(n))
#     return out(SETTINGS, data, result)


# def COPP(data, n):
#     """
#     Coppock Curve
#     """
#     M = data['close'].diff(int(n * 11 / 10) - 1)
#     N = data['close'].shift(int(n * 11 / 10) - 1)
#     ROC1 = M / N
#     M = data['close'].diff(int(n * 14 / 10) - 1)
#     N = data['close'].shift(int(n * 14 / 10) - 1)
#     ROC2 = M / N
#     result = pd.Series(pd.ewma(ROC1 + ROC2, span=n, min_periods=n), name='Copp_' + str(n))
#     return out(SETTINGS, data, result)


# def KELCH(data, n):
#     """
#     Keltner Channel
#     """
#     KelChM = pd.Series(pd.rolling_mean((data['high'] + data['low'] + data['close']) / 3, n), name='KelChM_' + str(n))
#     KelChU = pd.Series(pd.rolling_mean((4 * data['high'] - 2 * data['low'] + data['close']) / 3, n), name='KelChU_' + str(n))
#     KelChD = pd.Series(pd.rolling_mean((-2 * data['high'] + 4 * data['low'] + data['close']) / 3, n), name='KelChD_' + str(n))
#     result = pd.DataFrame([KelChM, KelChU, KelChD]).transpose()
#     return out(SETTINGS, data, result)


# def ULTOSC(data):
#     """
#     Ultimate Oscillator
#     """
#     i = 0
#     TR_l = [0]
#     BP_l = [0]
#     while i < len(data) - 1:  # data.index[-1]:
#         TR = max(data.get_value(i + 1, 'high'), data.get_value(i, 'close')) - min(data.get_value(i + 1, 'low'), data.get_value(i, 'close'))
#         TR_l.append(TR)
#         BP = data.get_value(i + 1, 'close') - min(data.get_value(i + 1, 'low'), data.get_value(i, 'close'))
#         BP_l.append(BP)
#         i = i + 1
#     result = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name='Ultimate_Osc')
#     return out(SETTINGS, data, result)


# def DONCH(data, n):
#     """
#     Donchian Channel
#     """
#     i = 0
#     DC_l = []
#     while i < n - 1:
#         DC_l.append(0)
#         i = i + 1
#     i = 0
#     while i + n - 1 < len(data) - 1:  # data.index[-1]:
#         DC = max(data['high'].ix[i:i + n - 1]) - min(data['low'].ix[i:i + n - 1])
#         DC_l.append(DC)
#         i = i + 1
#     DonCh = pd.Series(DC_l, name='Donchian_' + str(n))
#     result = DonCh.shift(n - 1)
#     return out(SETTINGS, data, result)


# def STDDEV(data, n):
#     """
#     Standard Deviation
#     """
#     result = pd.Series(pd.rolling_std(data['close'], n), name='STD_' + str(n))
#     return out(SETTINGS, data, result)
