import pandas as pd
from dateutil import parser
import talib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
import numpy as np
import pickle
import statsmodels.tsa.ar_model as AR

# raw_data must be at least (for a one-day prediction)
# a 10x6 matrix.
# 10 Rows : at least the last 32 observations
# 5 columns : must be, in this orders, Dates, Open, High, Low, Close, Volume
def create_data_set(raw_data, get_final_row=False, n_classes=4):
    dataset = pd.DataFrame()
    dataset['dates'] = \
        [parser.parse(x, dayfirst=True) for x in raw_data.iloc[:, 0]]

    dataset["month_index"] = \
        [x.month for x in dataset['dates']]

    dataset["day_index"] = \
        [x.day for x in dataset['dates']]

    dataset[['open', 'high', 'low', 'close']] = \
        raw_data.iloc[:, 1:-1]

    for n_shift in range(1, 31):
        dataset[['open_delta_' + str(n_shift), 'high_delta_' + str(n_shift),
                 'low_delta_' + str(n_shift), 'close_delta_' + str(n_shift)]] = \
            raw_data.iloc[:, 1:-1].shift(n_shift)

    dataset['volume_from'] = raw_data.iloc[:, -2]
    dataset['volume_to'] = raw_data.iloc[:, -1]
    dataset['volume'] = raw_data.iloc[:, -1] + raw_data.iloc[:, -2]

    for n_days in range(2, 21, 1):
        dataset[str(n_days) + 'D_close_mean'] = \
            dataset["close"].rolling(n_days).mean()
    for n_days in range(25, 61, 5):
        dataset[str(n_days) + 'D_close_mean'] = \
            dataset["close"].rolling(n_days).mean()

    for n_days in range(2, 21, 1):
        dataset[str(n_days) + 'D_close_vol'] = \
            dataset["close"].rolling(n_days).std()
    for n_days in range(25, 61, 5):
        dataset[str(n_days) + 'D_close_vol'] = \
            dataset["close"].rolling(n_days).std()

    for i_lag in range(2, 30):
        AR_window = i_lag
        AR_predicted_close = list()
        [AR_predicted_close.append(pd.np.NaN) for i in range(0, i_lag)]
        for n_days in range(i_lag, len(dataset)):
            AR_endog = dataset.loc[n_days-AR_window:n_days, "close"]
            AR_close = AR.AR(np.asarray(AR_endog))
            AR_close_opt = AR_close.fit(maxlag=i_lag)
            AR_params = AR_close_opt.params
            AR_predicted_close.append(np.sum(dataset.loc[n_days-AR_window+1:n_days, "close"] * AR_params))
        dataset['AR(' + str(i_lag) + ')_REGRESSION_close'] = AR_predicted_close

    dataset[['open_return', 'high_return', 'low_return', 'close_return']] = \
        ((raw_data.iloc[:, 1:-1] - raw_data.iloc[:, 1:-1].shift(1)) / raw_data.iloc[:, 1:-1].shift(1))

    for n_shift in range(1, 31):
        dataset[['open_return_delta_' + str(n_shift), 'high_return_delta_' + str(n_shift),
                 'low_return_delta_' + str(n_shift), 'close_return_delta_' + str(n_shift)]] = \
            dataset.loc[:, 'open_return':'close_return'].shift(n_shift)

    dataset['diff_volume_from'] = (raw_data.iloc[:, -2]) - (raw_data.iloc[:, -2]).shift(1)
    dataset['diff_volume_to'] = (raw_data.iloc[:, -1]) - (raw_data.iloc[:, -1]).shift(1)
    dataset['diff_volume'] = (raw_data.iloc[:, -2] + raw_data.iloc[:, -1]) - (
            raw_data.iloc[:, -2] + raw_data.iloc[:, -1]).shift(1)

    for n_days in range(2, 21, 1):
        dataset[str(n_days) + 'D_return_mean'] = \
            dataset["close_return"].rolling(n_days).mean()
    for n_days in range(25, 61, 5):
        dataset[str(n_days) + 'D_return_mean'] = \
            dataset["close_return"].rolling(n_days).mean()

    for n_days in range(2, 21, 1):
        dataset[str(n_days) + 'D_return_vol'] = \
            dataset["close_return"].rolling(n_days).std()
    for n_days in range(25, 61, 5):
        dataset[str(n_days) + 'D_return_vol'] = \
            dataset["close_return"].rolling(n_days).std()

    for i_lag in range(2, 30):
        AR_window = i_lag
        AR_predicted_close = list()
        [AR_predicted_close.append(pd.np.NaN) for i in range(0, i_lag + 1)]
        for n_days in range(i_lag + 1, len(dataset)):
            AR_endog = dataset.loc[n_days-AR_window:n_days, "close_return"]
            AR_close = AR.AR(np.asarray(AR_endog))
            AR_close_opt = AR_close.fit(maxlag=i_lag)
            AR_params = AR_close_opt.params
            AR_predicted_close.append(np.sum(dataset.loc[n_days-AR_window+1:n_days, "close_return"] * AR_params))
        dataset['AR(' + str(i_lag) + ')_REGRESSION_close_return'] = AR_predicted_close

    # Overlap Studies
    ####################################################################################################################

    for n_period in range(5, 31, 1):
        dataset[str(n_period) + 'D_BB_UP_close'], dataset[str(n_period) + 'D_BB_MIDDLE_close'],\
        dataset[str(n_period) + 'D_BB_LOW_close'] = talib.BBANDS(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_EMA_close'] = talib.EMA(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_DEMA_close'] = talib.DEMA(dataset["close"], timeperiod=n_period)

    dataset[str(n_period) + 'D_HT_TREND_close'] = talib.HT_TRENDLINE(dataset["close"])

    for n_period in range(5, 31, 5):

        dataset[str(n_period) + 'D_KAMA_close'] = talib.KAMA(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MA_close'] = talib.MA(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MIDPOINT_close'] = talib.MIDPOINT(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MIDPRICE_close'] = talib.MIDPRICE(dataset["high"], dataset["low"],
                                                                     timeperiod=n_period)

    dataset[str(n_period) + 'D_SAR_close'] = talib.SAR(dataset["high"], dataset["low"])

    dataset[str(n_period) + 'D_SAREXT_close'] = talib.SAREXT(dataset["high"], dataset["low"])

    for n_period in range(5, 31, 1):
        dataset[str(n_period) + 'D_SMA_close'] = talib.SMA(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_T3_close'] = talib.T3(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_TEMA_close'] = talib.TEMA(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_TRIMA_close'] = talib.TRIMA(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_WMA_close'] = talib.WMA(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_BB_UP_return'], dataset[str(n_period) + 'D_BB_MIDDLE'],\
        dataset[str(n_period) + 'D_BB_LOW'] = talib.BBANDS(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_EMA_return'] = talib.EMA(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_DEMA_return'] = talib.DEMA(dataset["close_return"], timeperiod=n_period)

    dataset[str(n_period) + 'D_HT_TREND_return'] = talib.HT_TRENDLINE(dataset["close_return"])

    for n_period in range(5, 31, 5):
        dataset[str(n_period) + 'D_KAMA_return'] = talib.KAMA(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MA_return'] = talib.MA(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MIDPOINT_return'] = talib.MIDPOINT(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MIDPRICE_return'] = talib.MIDPRICE(dataset["high_return"], dataset["low_return"],
                                                                      timeperiod=n_period)

    dataset[str(n_period) + 'D_SAR_return'] = talib.SAR(dataset["high_return"], dataset["low_return"])

    dataset[str(n_period) + 'D_SAREXT_return'] = talib.SAREXT(dataset["high_return"], dataset["low_return"])

    for n_period in range(5, 31, 5):
        dataset[str(n_period) + 'D_SMA_return'] = talib.SMA(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_T3_return'] = talib.T3(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_TEMA_return'] = talib.TEMA(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_TRIMA_return'] = talib.TRIMA(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_WMA_return'] = talib.WMA(dataset["close_return"], timeperiod=n_period)

    # Momentum Indicators
    ####################################################################################################################

    for n_period in range(5, 31, 5):
        dataset[str(n_period) + 'D_ADX_close'] = \
            talib.ADX(dataset["high"], dataset["low"], dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ADXR_close'] = talib.ADXR(dataset["high"], dataset["low"], dataset["close"],
                                                             timeperiod=n_period)

        dataset[str(n_period) + 'D_APO_close'] = talib.APO(dataset["close"], fastperiod=n_period,
                                                           slowperiod=2*n_period)

        dataset[str(n_period) + 'D_AROON_DOWN_close'], dataset[str(n_period) + 'D_AROON_UP_close'] = \
            talib.AROON(dataset["high"], dataset["low"], timeperiod=n_period)

        dataset[str(n_period) + 'D_AROONOSC_close'] = \
            talib.AROONOSC(dataset["high"], dataset["low"], timeperiod=n_period)

        dataset[str(n_period) + 'D_BOP_close'] = \
            talib.BOP(dataset["open"], dataset["high"], dataset["low"], dataset["close"])

        dataset[str(n_period) + 'D_CCI_close'] = \
            talib.CCI(dataset["high"], dataset["low"], dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_CMO_close'] = \
            talib.CMO(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_DX_close'] = \
            talib.DX(dataset["high"], dataset["low"], dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MACD_close'], dataset[str(n_period) + 'D_MACD_SIGNAL_close'],\
        dataset[str(n_period) + 'D_MACD_HIST_close'] = talib.MACD(dataset["close"], fastperiod=n_period,
                                                                  slowperiod=2*n_period, signalperiod=int(n_period/2))

    dataset[str(n_period) + 'D_MACDEXT_close'], dataset[str(n_period) + 'D_MACDEXT_SIGNAL_close'],\
    dataset[str(n_period) + 'D_MACDEXT_HIST_close'] = talib.MACDEXT(dataset["close"])

    for n_period in range(5, 31, 5):
        dataset[str(n_period) + 'D_MACDFIX_close'], dataset[str(n_period) + 'D_MACDEXT_SIGNAL_close'],\
        dataset[str(n_period) + 'D_MACDEXT_HIST_close'] = talib.MACDFIX(dataset["close"], signalperiod=n_period)

        dataset[str(n_period) + 'D_MFI_from_close'] = \
            talib.MFI(dataset["open"], dataset["high"], dataset["low"], dataset["volume_from"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MFI_to_close'] = \
            talib.MFI(dataset["open"], dataset["high"], dataset["low"], dataset["volume_to"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MFI_close'] = \
            talib.MFI(dataset["open"], dataset["high"], dataset["low"], dataset["volume"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MINUS_DI_close'] = \
            talib.MINUS_DI(dataset["high"], dataset["low"], dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MINUS_DM_close'] = talib.MINUS_DM(dataset["high"], dataset["low"],
                                                                     timeperiod=n_period)

        dataset[str(n_period) + 'D_MOM_close'] = talib.MOM(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_PLUS_DI_close'] = talib.PLUS_DI(dataset["high"], dataset["low"],
                                                                   dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_PLUS_DM_close'] = talib.PLUS_DM(dataset["high"], dataset["low"], timeperiod=n_period)

        dataset[str(n_period) + 'D_PPO_close'] = talib.PPO(dataset["close"], fastperiod=n_period, slowperiod=2*n_period)

        dataset[str(n_period) + 'D_ROC_close'] = talib.ROC(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ROCP_close'] = talib.ROCP(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ROCR_close'] = talib.ROCR(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ROCR100_close'] = talib.ROCR100(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_RSI_close'] = talib.RSI(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_STOCH_K_close'], dataset[str(n_period) + 'D_STOCH_D_close'] = \
            talib.STOCH(dataset["high"], dataset["low"], dataset["close"], fastk_period=n_period,
                        slowk_period=2*n_period, slowd_period=n_period)

        dataset[str(n_period) + 'D_STOCHF_K_close'], dataset[str(n_period) + 'D_STOCHF_D_close'] = \
            talib.STOCHF(dataset["high"], dataset["low"], dataset["close"], fastk_period=n_period,
                         fastd_period=n_period)

        dataset[str(n_period) + 'D_STOCHRSI_K_close'], dataset[str(n_period) + 'D_STOCHRSI_D_close'] = \
            talib.STOCHRSI(dataset["close"], timeperiod=2*n_period, fastk_period=n_period, fastd_period=n_period)

        dataset[str(n_period) + 'D_TRIX_close'] = talib.TRIX(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ULT_OSC_close'] = \
            talib.ULTOSC(dataset["high"], dataset["low"], dataset["close"], timeperiod1=n_period,
                         timeperiod2=2*n_period, timeperiod3=3*n_period)

        dataset[str(n_period) + 'D_WILLR_close'] = \
            talib.WILLR(dataset["high"], dataset["low"], dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ADX_return'] = \
            talib.ADX(dataset["high_return"], dataset["low_return"], dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ADXR_return'] = \
            talib.ADXR(dataset["high_return"], dataset["low_return"], dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_APO_return'] = talib.APO(dataset["close_return"], fastperiod=n_period,
                                                            slowperiod=2*n_period)

        dataset[str(n_period) + 'D_AROON_DOWN_return'], dataset[str(n_period) + 'D_AROON_UP'] = \
            talib.AROON(dataset["high_return"], dataset["low_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_AROONOSC_return'] = \
            talib.AROONOSC(dataset["high_return"], dataset["low_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_BOP_return'] = \
            talib.BOP(dataset["open_return"], dataset["high_return"], dataset["low_return"], dataset["close_return"])

        dataset[str(n_period) + 'D_CCI_return'] = \
            talib.CCI(dataset["high_return"], dataset["low_return"], dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_CMO_return'] = \
            talib.CMO(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_DX_return'] = \
            talib.DX(dataset["high_return"], dataset["low_return"], dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MACD_return'], dataset[str(n_period) + 'D_MACD_SIGNAL'],\
        dataset[str(n_period) + 'D_MACD_HIST'] = talib.MACD(dataset["close_return"], fastperiod=n_period,
                                                            slowperiod=2*n_period, signalperiod=int(n_period/2))

    dataset[str(n_period) + 'D_MACDEXT_return'], dataset[str(n_period) + 'D_MACDEXT_SIGNAL'],\
    dataset[str(n_period) + 'D_MACDEXT_HIST'] = talib.MACDEXT(dataset["close_return"])

    for n_period in range(5, 31, 5):
        dataset[str(n_period) + 'D_MACDFIX_return'], dataset[str(n_period) + 'D_MACDEXT_SIGNAL'],\
        dataset[str(n_period) + 'D_MACDEXT_HIST'] = talib.MACDFIX(dataset["close_return"], signalperiod=n_period)

        dataset[str(n_period) + 'D_MFI_from_return'] = \
            talib.MFI(dataset["open_return"], dataset["high_return"], dataset["low_return"], dataset["volume_from"],
                      timeperiod=n_period)

        dataset[str(n_period) + 'D_MFI_to_return'] = \
            talib.MFI(dataset["open_return"], dataset["high_return"], dataset["low_return"], dataset["volume_to"],
                      timeperiod=n_period)

        dataset[str(n_period) + 'D_MFI_return'] = \
            talib.MFI(dataset["open_return"], dataset["high_return"], dataset["low_return"], dataset["volume"],
                      timeperiod=n_period)

        dataset[str(n_period) + 'D_MINUS_DI_return'] = \
            talib.MINUS_DI(dataset["high_return"], dataset["low_return"], dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MINUS_DM_return'] = \
            talib.MINUS_DM(dataset["high_return"], dataset["low_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_MOM_return'] = talib.MOM(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_PLUS_DI_return'] = talib.PLUS_DI(dataset["high_return"], dataset["low_return"],
                                                                    dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_PLUS_DM_return'] = talib.PLUS_DM(dataset["high_return"], dataset["low_return"],
                                                                    timeperiod=n_period)

        dataset[str(n_period) + 'D_PPO_return'] = talib.PPO(dataset["close_return"], fastperiod=n_period,
                                                            slowperiod=2*n_period)

        dataset[str(n_period) + 'D_ROC_return'] = talib.ROC(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ROCP_return'] = talib.ROCP(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ROCR_return'] = talib.ROCR(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ROCR100_return'] = talib.ROCR100(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_RSI_return'] = talib.RSI(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_STOCH_K_return'], dataset[str(n_period) + 'D_STOCH_D_return'] = \
            talib.STOCH(dataset["high_return"], dataset["low_return"], dataset["close_return"], fastk_period=n_period,
                        slowk_period=2*n_period, slowd_period=n_period)

        dataset[str(n_period) + 'D_STOCHF_K_return'], dataset[str(n_period) + 'D_STOCHF_D_return'] = \
            talib.STOCHF(dataset["high_return"], dataset["low_return"], dataset["close_return"], fastk_period=n_period,
                         fastd_period=2*n_period)

        dataset[str(n_period) + 'D_STOCHRSI_K_return'], dataset[str(n_period) + 'D_STOCHRSI_D_return'] = \
            talib.STOCHRSI(dataset["close_return"], timeperiod=2*n_period, fastk_period=n_period, fastd_period=n_period)

        dataset[str(n_period) + 'D_TRIX_return'] = talib.TRIX(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_ULT_OSC_return'] = \
            talib.ULTOSC(dataset["high_return"], dataset["low_return"], dataset["close_return"], timeperiod1=n_period,
                         timeperiod2=2*n_period, timeperiod3=3*n_period)

        dataset[str(n_period) + 'D_WILLR_return'] = \
            talib.WILLR(dataset["high_return"], dataset["low_return"], dataset["close_return"], timeperiod=n_period)

    # Volume Indicators
    ####################################################################################################################

    dataset['AD_from_close'] = \
        talib.AD(dataset["high"], dataset["low"], dataset["close"], dataset["volume_from"])

    dataset['AD_to_close'] = \
        talib.AD(dataset["high"], dataset["low"], dataset["close"], dataset["volume_to"])

    dataset['AD_close'] = \
        talib.AD(dataset["high"], dataset["low"], dataset["close"], dataset["volume"])

    for n_period in range(2, 21, 1):
        dataset[str(n_period) + 'D_ADOSC_from_close'] = \
            talib.ADOSC(dataset["high"], dataset["low"], dataset["close"], dataset["volume_from"],
                        fastperiod=n_period, slowperiod=2*n_period)

        dataset[str(n_period) + 'D_ADOSC_to_close'] = \
            talib.ADOSC(dataset["high"], dataset["low"], dataset["close"], dataset["volume_to"],
                        fastperiod=n_period, slowperiod=2 * n_period)

        dataset[str(n_period) + 'D_ADOSC_close'] = \
            talib.ADOSC(dataset["high"], dataset["low"], dataset["close"], dataset["volume"],
                        fastperiod=n_period, slowperiod=2 * n_period)

    dataset['OBV_from_close'] = \
        talib.OBV(dataset["close"], dataset["volume_from"])

    dataset['OBV_to_close'] = \
        talib.OBV(dataset["close"], dataset["volume_to"])

    dataset['OBV_close'] = \
        talib.OBV(dataset["close"], dataset["volume"])

    dataset['AD_from_return'] = \
        talib.AD(dataset["high_return"], dataset["low_return"], dataset["close_return"], dataset["volume_from"])

    dataset['AD_to_return'] = \
        talib.AD(dataset["high_return"], dataset["low_return"], dataset["close_return"], dataset["volume_to"])

    dataset['AD_return'] = \
        talib.AD(dataset["high_return"], dataset["low_return"], dataset["close_return"], dataset["volume"])

    for n_period in range(2, 21, 1):
        dataset[str(n_period) + 'D_ADOSC_from_return'] = \
            talib.ADOSC(dataset["high_return"], dataset["low_return"], dataset["close_return"], dataset["volume_from"],
                        fastperiod=n_period, slowperiod=2 * n_period)

        dataset[str(n_period) + 'D_ADOSC_to_return'] = \
            talib.ADOSC(dataset["high_return"], dataset["low_return"], dataset["close_return"], dataset["volume_to"],
                        fastperiod = n_period, slowperiod = 2 * n_period)

        dataset[str(n_period) + 'D_ADOSC_return'] = \
            talib.ADOSC(dataset["high_return"], dataset["low_return"], dataset["close_return"], dataset["volume"],
                        fastperiod=n_period, slowperiod=2 * n_period)

    dataset['OBV_from_return'] = \
        talib.OBV(dataset["close_return"], dataset["volume_from"])

    dataset['OBV_to_return'] = \
        talib.OBV(dataset["close_return"], dataset["volume_to"])

    dataset['OBV_return'] = \
        talib.OBV(dataset["close_return"], dataset["volume"])

    # Volatility Indicators
    ####################################################################################################################

    for n_period in range(2, 21, 1):
        dataset[str(n_period) + 'D_ATR_close'] = \
            talib.ATR(dataset["high"], dataset["low"], dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_NATR_close'] = \
            talib.NATR(dataset["high"], dataset["low"], dataset["close"], timeperiod=n_period)

    dataset['TRANGE_close'] = \
        talib.TRANGE(dataset["high"], dataset["low"], dataset["close"])

    for n_period in range(2, 21, 1):
        dataset[str(n_period) + 'D_ATR_return'] = \
            talib.ATR(dataset["high_return"], dataset["low_return"], dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_NATR_return'] = \
            talib.NATR(dataset["high_return"], dataset["low_return"], dataset["close_return"], timeperiod=n_period)

    dataset['TRANGE_return'] = \
        talib.TRANGE(dataset["high_return"], dataset["low_return"], dataset["close_return"])

    # Cycle Indicators
    ####################################################################################################################

    dataset['HT_DCPERIOD_close'] = talib.HT_DCPERIOD(dataset["close"])

    dataset['HT_DCPHASE_close'] = talib.HT_DCPHASE(dataset["close"])

    dataset['HT_PHASOR_INPHASE_close'], dataset['HT_PHASOR_QUADRATURE'] = talib.HT_PHASOR(dataset["close"])

    dataset['HT_SINE_close'], dataset['HT_LEADSINE_close'] = talib.HT_SINE(dataset["close"])

    dataset['HT_TRENDMODE_close'] = talib.HT_TRENDMODE(dataset["close"])

    dataset['HT_DCPERIOD_return'] = talib.HT_DCPERIOD(dataset["close_return"])

    dataset['HT_DCPHASE_return'] = talib.HT_DCPHASE(dataset["close_return"])

    dataset['HT_PHASOR_INPHASE_return'], dataset['HT_PHASOR_QUADRATURE_return'] = \
        talib.HT_PHASOR(dataset["close_return"])

    dataset['HT_SINE_return'], dataset['HT_LEADSINE_return'] = talib.HT_SINE(dataset["close_return"])

    dataset['HT_TRENDMODE_return'] = talib.HT_TRENDMODE(dataset["close_return"])

    # Statistic Functions
    ####################################################################################################################

    for n_period in range(2, 21):
        dataset[str(n_period) + 'D_BETA_close'] = talib.BETA(dataset["high"], dataset["low"],
                                                             timeperiod=n_period)

        dataset[str(n_period) + 'D_CORREL_close'] = talib.CORREL(dataset["high"], dataset["low"],
                                                                 timeperiod=n_period)

        dataset[str(n_period) + 'D_LINEAR_REGRESSION_close'] = talib.LINEARREG(dataset["close"],
                                                                               timeperiod=n_period)

        dataset[str(n_period) + 'D_LINEAR_REGRESSION_ANGLE_close'] = talib.LINEARREG_ANGLE(dataset["close"],
                                                                                           timeperiod=n_period)

        dataset[str(n_period) + 'D_LINEAR_REGRESSION_INTERCEPT_close'] = talib.LINEARREG_INTERCEPT(dataset["close"],
                                                                                                   timeperiod=n_period)

        dataset[str(n_period) + 'D_LINEAR_REGRESSION_SLOPE_close'] = talib.LINEARREG_SLOPE(dataset["close"],
                                                                                           timeperiod=n_period)

        dataset[str(n_period) + 'D_STD_DEV_close'] = talib.STDDEV(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_TSF_close'] = talib.TSF(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_VAR_close'] = talib.VAR(dataset["close"], timeperiod=n_period)

        dataset[str(n_period) + 'D_BETA_return'] = talib.BETA(dataset["high_return"], dataset["low_return"],
                                                              timeperiod=n_period)

        dataset[str(n_period) + 'D_CORREL_return'] = talib.CORREL(dataset["high_return"], dataset["low_return"],
                                                                  timeperiod=n_period)

        dataset[str(n_period) + 'D_LINEAR_REGRESSION_return'] = talib.LINEARREG(dataset["close_return"],
                                                                                timeperiod=n_period)

        dataset[str(n_period) + 'D_LINEAR_REGRESSION_ANGLE_return'] = talib.LINEARREG_ANGLE(dataset["close_return"],
                                                                                            timeperiod=n_period)

        dataset[str(n_period) + 'D_LINEAR_REGRESSION_INTERCEPT_return'] = talib.LINEARREG_INTERCEPT(
            dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_LINEAR_REGRESSION_SLOPE_return'] = talib.LINEARREG_SLOPE(dataset["close_return"],
                                                                                            timeperiod=n_period)

        dataset[str(n_period) + 'D_STD_DEV_return'] = talib.STDDEV(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_TSF_return'] = talib.TSF(dataset["close_return"], timeperiod=n_period)

        dataset[str(n_period) + 'D_VAR_return'] = talib.VAR(dataset["close_return"], timeperiod=n_period)

    # Price Transform
    ####################################################################################################################

    dataset['AVGPRICE_close'] = \
        talib.AVGPRICE(dataset["open"], dataset["high"], dataset["low"], dataset["close"])

    dataset['MEDPRICE_close'] = \
        talib.MEDPRICE(dataset["high"], dataset["low"])

    dataset['TYPPRICE_close'] = \
        talib.TYPPRICE(dataset["high"], dataset["low"], dataset["close"])

    dataset['WCLPRICE_close'] = \
        talib.WCLPRICE(dataset["high"], dataset["low"], dataset["close"])

    dataset['AVGPRICE_return'] = \
        talib.AVGPRICE(dataset["open_return"], dataset["high_return"], dataset["low_return"], dataset["close_return"])

    dataset['MEDPRICE_return'] = \
        talib.MEDPRICE(dataset["high_return"], dataset["low_return"])

    dataset['TYPPRICE_return'] = \
        talib.TYPPRICE(dataset["high_return"], dataset["low_return"], dataset["close_return"])

    dataset['WCLPRICE_return'] = \
        talib.WCLPRICE(dataset["high_return"], dataset["low_return"], dataset["close_return"])

    # Target classe of each observation
    ####################################################################################################################

    if n_classes == 2:
        dataset['target'] = \
            pd.DataFrame([1 if float(x) > 0.00 else 0 for x in dataset["close_return"]]).shift(-1)
    else:
        dataset['target'] = \
            pd.DataFrame([3 if float(x) > 0.02 else 2 if float(x) > 0.00 else 0 if float(x) < -0.02 else 1 for x in
                          dataset["close_return"]]).shift(-1)

    last_row = dataset.iloc[-1, :]

    dataset = dataset.dropna(axis=0).reset_index(drop=True)

    if get_final_row:
        dataset = dataset.append(last_row)

    n_features = len(dataset.columns)

    with open('OutputResults/Datasets/MainDataSet-' + str(n_features) + "Features", 'wb') as dataset_file:
        pickle.dump(dataset, dataset_file)

    return dataset


def select_features_RF_regressor(raw_data, n_features):

    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    scores = list()
    train_sample = raw_data[raw_data.index <= 0.75 * len(raw_data)]
    test_sample = raw_data[raw_data.index > 0.75 * len(raw_data)]
    for col in raw_data.columns[:-1]:
        one_score = dict()
        one_feature_train = train_sample.loc[:, col]
        one_feature_test = test_sample.loc[:, col]
        rf.fit(one_feature_train.values.reshape(-1, 1), train_sample["target"])
        score = rf.score(one_feature_test.values.reshape(-1, 1), test_sample["target"])
        one_score.update(feature_name=col)
        one_score.update(score=score)
        scores.append(one_score)

    features_infos = pd.DataFrame(scores)
    features_infos = features_infos.sort_values(by='score')

    if str(n_features) != 'all':
        n_best_scores = features_infos.iloc[-n_features:, :]
    else:
        n_best_scores = features_infos

    return n_best_scores


def scale_rolling_window(data, window=30):

    scaled_data = pd.DataFrame(columns=data.columns)

    for n_periods in range(window, len(data), window):
        rolling_data = data.iloc[n_periods-window:n_periods, :]
        rolling_mean = np.mean(rolling_data)
        rolling_std = np.std(rolling_data)

        rolling_scaled_data = (rolling_data-rolling_mean)/rolling_std

        scaled_data = scaled_data.append(rolling_scaled_data)

    scaled_data = scaled_data.set_index(data.index[29:])

    # Replace features that are not supposed to be scaled (dummy variable, targets)
    dummies = data.loc[:, (data.round(5).isin([-1.00000,0.00000,1.00000])).all(axis=0)]
    for dummy in dummies:
        scaled_data[dummy] = data[dummy]

    # In case of 4 classes
    dummies = data.loc[:, (data.round(5).isin([0.00000, 1.00000, 2.00000, 3.00000, 4.00000])).all(axis=0)]
    for dummy in dummies:
        scaled_data[dummy] = data[dummy]

    dummies = data.loc[:, ((data/100).round(10).isin([-1.0000000000,0.0000000000,1.0000000000])).all(axis=0)]
    for dummy in dummies:
        scaled_data[dummy] = data[dummy]

    # In case we forgot some kind of dummy variable (and make some NaN by the way)
    # We fill NaN by linear interpolation
    scaled_data = scaled_data.interpolate(axis=0)

    return scaled_data


def optimize_model(dataset):
    dataset_size = len(dataset)

    targets = dataset.loc[:, 'target']
    dates = dataset.loc[:, 'dates']
    dataset = dataset.iloc[:, 1:-1]

    RF_summaries = list()
    ADA_summaries = list()

    ####################################################################################################################
    # Procédure Walk-Forward d'optimisation et de test de l'algorithme
    # --> 4-Fold Cross-Validation
    ####################################################################################################################
    for interval in range(25, 101, 25):

        test_interval = interval / 100
        train_interval = 1 - test_interval

        ################################################################################################################
        # Split entre Training Set et Test Set, en fonction de l'itération actuelle de l'algorithme de cross-validationn
        # 1 - Test Set : [0-1112]           Training Set : [1112-4448]
        # 2 - Test Set : [1112-2224]        Training Set : [0-1112] + [2224-4448]
        # 3 - Test Set : [2224-3336]        Training Set : [0-2224] + [3336-4448]
        # 4 - Test Set : [3336-4448]        Training Set : [0-3336]
        ################################################################################################################

        test_lower_index = int((test_interval - 0.25) * dataset_size)
        test_upper_index = int((test_interval) * dataset_size)
        test_set = dataset.loc[list(range(test_lower_index, test_upper_index)), :]
        training_set = dataset.loc[~dataset.index.isin(test_set.index), :]
        training_targets = targets[targets.index.isin(training_set.index)]
        test_targets = targets[targets.index.isin(test_set.index)]

        ################################################################################################################
        # Sélection des features les plus explicatives
        ################################################################################################################

        # With Select K Best
        #######################
        # select_k_best_scores = pd.DataFrame(index=training_scaled_data.columns)
        # select_k_best_scores = pd.DataFrame(index=training_set.columns)
        # select_k_best = SelectKBest(k='all')
        # select_k_best.fit(training_set, training_targets)
        # select_k_best_scores['score'] = select_k_best.scores_
        # select_k_best_scores['feature_name'] = training_set.columns
        # select_k_best_scores = select_k_best_scores.sort_values(by='score')

        # With iterative algorithm testing the predictive power of each features through random forest regression on variable to explain
        ################################################################################################################################
        selected_features_RF_reg = select_features_RF_regressor(training_set.join(training_targets), 'all')

        # Test 23/05
        #training_set = training_set[selected_features_name]
        #test_set = test_set[selected_features_name]

        ################################################################################################################
        # Création d'un Dataset Scalé
        ################################################################################################################

        # With our personnal function that scale with a rolling window
        ##############################################################
        training_scaled_data = scale_rolling_window(training_set, window=30)
        test_scaled_data = scale_rolling_window(test_set, window=30)
        training_targets = targets[targets.index.isin(training_scaled_data.index)]
        test_targets = targets[targets.index.isin(test_scaled_data.index)]

        for n_features in range(10, 120, 5):
            ADA_summary = dict()
            RF_summary = dict()

            selected_features_name = selected_features_RF_reg.iloc[-n_features:, :].loc[:, 'feature_name']

            fitted_training_scaled_data = training_scaled_data[selected_features_name]
            fitted_test_scaled_data = test_scaled_data[selected_features_name]

            print("\n")
            print("Période " + str(test_lower_index) + " -> " + str(test_upper_index) + " ")
            print("Nombre de features : " + str(n_features))

            ############################################################################################################
            # Implémentation de l'algorithme de Random Forest
            ############################################################################################################

            random_forest = RandomForestClassifier(n_estimators=100)
            random_forest.fit(fitted_training_scaled_data, training_targets)
            score = random_forest.score(fitted_test_scaled_data, test_targets)

            RF_summary.update(model=random_forest)
            RF_summary.update(period=str(test_lower_index) + ":" + str(test_upper_index))
            RF_summary.update(num_features=n_features)
            RF_summary.update(score=score)
            RF_summary.update(features_names=selected_features_name)

            with open("OutputResults/Models/RandomForest-" + str(test_lower_index) + ":" + str(test_upper_index) + "-" +
                      str(n_features) + "_Features", 'wb') as one_rf_file:
                pickle.dump(RF_summary, one_rf_file)

            RF_summaries.append(RF_summary)

            print("Score de prédiction du modèle Random Forest : " + str(score))

            ############################################################################################################
            # Implémentation de l'algorithme de Ada Boost
            ############################################################################################################

            ada_boost = AdaBoostClassifier()
            ada_boost.fit(fitted_training_scaled_data, training_targets)
            score = ada_boost.score(fitted_test_scaled_data, test_targets)

            ADA_summary.update(model=ada_boost)
            ADA_summary.update(period=str(test_lower_index) + ":" + str(test_upper_index))
            ADA_summary.update(num_features=n_features)
            ADA_summary.update(score=score)
            ADA_summary.update(features_names=selected_features_name)

            with open("OutputResults/Models/AdaBoost-" + str(test_lower_index) + ":" + str(test_upper_index) + "-" +
                      str(n_features) + "_Features", 'wb') as one_ada_file:
                pickle.dump(ADA_summary, one_ada_file)

            ADA_summaries.append(ADA_summary)

            print("Score de prédiction du modèle Ada Boost : " + str(score))
            print("\n")

    with open("OutputResults/OptimSummaries/RandomForest", 'wb') as rf_file:
        pickle.dump(RF_summaries, rf_file)

    with open("OutputResults/OptimSummaries/AdaBoost", 'wb') as ada_file:
        pickle.dump(ADA_summaries, ada_file)

    return RF_summaries, ADA_summaries


def optimize_model_simple(dataset):
    dataset_size = len(dataset)

    targets = dataset.loc[:, 'target']
    dates = dataset.loc[:, 'dates']
    dataset = dataset.iloc[:, 1:-1]

    RF_summaries = list()
    ADA_summaries = list()

    ####################################################################################################################
    # Simple Cross Validation Procedure :
    #   Training Set = 0.75 * Dataset
    #   Test Set = 0.25 * Dataset
    ####################################################################################################################

    ####################################################################################################################
    # Split entre Training Set et Test Set, en fonction de l'itération actuelle de l'algorithme de cross-validationn
    # 1 - Training Set : [0-4949]           Test Set : [4950-6599]
    ####################################################################################################################

    test_lower_index = int(0.75 * dataset_size)
    test_upper_index = int(dataset_size)
    test_set = dataset.iloc[dataset.index > int(0.75 * dataset_size), :]
    training_set = dataset.iloc[dataset.index <= int(0.75 * dataset_size), :]
    training_targets = targets[targets.index.isin(training_set.index)]
    test_targets = targets[targets.index.isin(test_set.index)]

    ####################################################################################################################
    # Sélection des features les plus explicatives
    ####################################################################################################################

    # With iterative algorithm testing the predictive power of each features through random forest regression on variable to explain
    ################################################################################################################################
    selected_features_RF_reg = select_features_RF_regressor(training_set.join(training_targets), 'all')

    selected_features_name = selected_features_RF_reg.iloc[-120:, :].loc[:, 'feature_name']

    ####################################################################################################################
    # Création d'un Dataset Scalé
    ####################################################################################################################

    # With our personnal function that scale with a rolling window
    ##############################################################
    training_scaled_data = scale_rolling_window(training_set, window=30)
    test_scaled_data = scale_rolling_window(test_set, window=30)
    training_targets = targets[targets.index.isin(training_scaled_data.index)]
    test_targets = targets[targets.index.isin(test_scaled_data.index)]

    for n_features in range(10, 120, 5):
        ADA_summary = dict()
        RF_summary = dict()

        selected_features_name = selected_features_RF_reg.iloc[-n_features:, :].loc[:, 'feature_name']

        fitted_training_scaled_data = training_scaled_data[selected_features_name]
        fitted_test_scaled_data = test_scaled_data[selected_features_name]

        print("\n")
        print("Période " + str(test_lower_index) + " -> " + str(test_upper_index) + " ")
        print("Nombre de features : " + str(n_features))

        ################################################################################################################
        # Implémentation de l'algorithme de Random Forest
        ################################################################################################################

        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(fitted_training_scaled_data, training_targets)
        score = random_forest.score(fitted_test_scaled_data, test_targets)

        RF_summary.update(model=random_forest)
        RF_summary.update(period=str(test_lower_index) + ":" + str(test_upper_index))
        RF_summary.update(num_features=n_features)
        RF_summary.update(score=score)
        RF_summary.update(features_names=selected_features_name)

        with open("OutputResults/Models/RandomForest-Simple-" + str(n_features) + "_Features", 'wb') as one_rf_file:
            pickle.dump(RF_summary, one_rf_file)

        RF_summaries.append(RF_summary)

        print("Score de prédiction du modèle Random Forest : " + str(score))

        ################################################################################################################
        # Implémentation de l'algorithme de Ada Boost
        ################################################################################################################

        ada_boost = AdaBoostClassifier()
        ada_boost.fit(fitted_training_scaled_data, training_targets)
        score = ada_boost.score(fitted_test_scaled_data, test_targets)

        ADA_summary.update(model=ada_boost)
        ADA_summary.update(period=str(test_lower_index) + ":" + str(test_upper_index))
        ADA_summary.update(num_features=n_features)
        ADA_summary.update(score=score)
        ADA_summary.update(features_names=selected_features_name)

        with open("OutputResults/Models/AdaBoost-Simple-" + str(n_features) + "_Features", 'wb') as one_ada_file:
            pickle.dump(ADA_summary, one_ada_file)

        ADA_summaries.append(ADA_summary)

        print("Score de prédiction du modèle Ada Boost : " + str(score))
        print("\n")

    with open("OutputResults/OptimSummaries/RandomForest-Simple", 'wb') as rf_file:
        pickle.dump(RF_summaries, rf_file)

    with open("OutputResults/OptimSummaries/AdaBoost-Simple", 'wb') as ada_file:
        pickle.dump(ADA_summaries, ada_file)

    return RF_summaries, ADA_summaries
