import pandas as pd
import numpy as np
import matplotlib.pyplot as mplot
import sys
import pickle

from MachineLearningTools.MachineLearningTools import create_data_set, select_features_RF_regressor,\
    scale_rolling_window, optimize_model, optimize_model_simple

mplot.style.use("ggplot")
stdout = sys.stdout
stderr = sys.stderr

do_optimize = False
four_fold_cross_val = False
do_create_dataset = False
do_detail_backtest = False

########################################################################################################################
# Import des Datas
########################################################################################################################

raw_data = pd.read_csv("Static/spi_yahoo.csv")

if do_create_dataset:
    dataset = create_data_set(raw_data, n_classes=2)
else:
    with open('OutputResults/Datasets/MainDataSet-1934Features', 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)

dataset_size = len(dataset)

if do_optimize:
    if four_fold_cross_val:
        RF_summaries, ADA_summaries = optimize_model(dataset)
    else:
        RF_summaries, ADA_summaries = optimize_model_simple(dataset)
else:
    with open("OutputResults/OptimSummaries/RandomForest-Simple", 'rb') as rf_file:
        RF_summaries = pickle.load(rf_file)

    with open("OutputResults/OptimSummaries/AdaBoost-Simple", 'rb') as ada_file:
        ADA_summaries = pickle.load(ada_file)

if four_fold_cross_val:
    last_period = RF_summaries['period']
else:
    last_period = 'Simple'

########################################################################################################################
# Backtest des stratégies
########################################################################################################################

# Récupération des informations du modèle optimal
######################################################

RF_score_by_num_features = pd.DataFrame(RF_summaries).groupby(by='num_features').mean()
ADA_score_by_num_features = pd.DataFrame(ADA_summaries).groupby(by='num_features').mean()

RF_opt_num_features = RF_score_by_num_features.sort_values(by='score').index[-1]
ADA_opt_num_features = ADA_score_by_num_features.sort_values(by='score').index[-1]

with open('OutputResults/Models/RandomForest-' + last_period + '-' + str(RF_opt_num_features) + '_Features', 'rb') as rf_file:
    RF_opt_dico = pickle.load(rf_file)

with open('OutputResults/Models/AdaBoost-' + last_period + '-' + str(ADA_opt_num_features) + '_Features', 'rb') as ada_file:
    ADA_opt_dico = pickle.load(ada_file)

RF_opt_model = RF_opt_dico['model']
ADA_opt_model = ADA_opt_dico['model']

RF_opt_features_names = RF_opt_dico['features_names']
ADA_opt_features_names = ADA_opt_dico['features_names']

targets = dataset.loc[:, 'target']
dates = dataset.loc[:, 'dates']
dataset = dataset.iloc[:, 1:-1]

# Backtest de l'algorithme de Random Forest
############################################

# Algo 1 : On déboucle la position en t+1 quoi qu'il arrive
# Nos positions ne durent donc jamais plus d'un jour
###############################################################

training_set = dataset[dataset.index <= int(0.75*dataset_size)]
test_set = dataset[dataset.index > int(0.75*dataset_size)]
training_targets = targets[targets.index <= int(0.75 * dataset_size)]
test_targets = targets[targets.index > int(0.75 * dataset_size)]

means = training_set.rolling(30).mean()
vols = training_set.rolling(30).std()

final_index = test_set.index[-1]
#rolling_data_set = training_set.iloc[-30:, :].reset_index(drop=True)

curr_rolling_mean = means.iloc[-1, :]
curr_rolling_vol = vols.iloc[-1, :]

rolling_data_set = training_set.iloc[-30:, :].loc[:, RF_opt_features_names].reset_index(drop=True)
closes = test_set.loc[:, 'close']
test_set = test_set.loc[:, RF_opt_features_names]

asset = 0
prev_cash = 0
cash = 0
max_drawdown = 0
worst_trade = 0
trade_returns = list()
pendingTrade = False
pendingTradeSide = ""
prev_predicted_class = 0
prediction_score = list()
good_prediction = 0
curr_close = 0
prev_close = 0
for one_set in test_set.iterrows():
    index = one_set[0]
    one_set = one_set[1]
    rolling_data_set = rolling_data_set.append(one_set)
    rolling_data_set = rolling_data_set.drop(0)
    rolling_data_set = rolling_data_set.reset_index(drop=True)
    curr_mean = rolling_data_set.mean()

    curr_vol = rolling_data_set.std()

    one_scaled_set = (one_set - curr_mean) / curr_vol
    dummies = [x for x in one_scaled_set.index if (np.isnan(one_scaled_set[x]))]

    if dummies:
        for dummy in dummies:
            one_scaled_set[dummy] = one_set[dummy]
        dummies.clear()

    predicted_class = RF_opt_model.predict(one_scaled_set.values.reshape(1, -1))
    class_probas = RF_opt_model.predict_proba(one_scaled_set.values.reshape(1, -1))
    predicted_class_proba = class_probas[0][int(predicted_class)]

    curr_close = closes[index]
    if curr_close > prev_close and prev_predicted_class == 1:
        good_prediction = 1
    elif curr_close > prev_close and prev_predicted_class == 0:
        good_prediction = 0
    elif curr_close < prev_close and prev_predicted_class == 0:
        good_prediction = 1
    elif curr_close < prev_close and prev_predicted_class == 1:
        good_prediction = 0

    prediction_score.append(good_prediction)
    prev_predicted_class = predicted_class
    prev_close = curr_close

    if pendingTrade:
        if pendingTradeSide == 'BUY':
            asset = asset - 1
            cash = cash + closes[index]
        elif pendingTradeSide == 'SELL':
            asset = asset + 1
            cash = cash - closes[index]
        pendingTrade = False
        trade_returns.append((cash - prev_cash))
        if cash < max_drawdown:
            max_drawdown = cash
        # if (cash - prev_cash < 0) and ((cash - prev_cash) < worst_trade):
        #     worst_trade = cash - prev_cash
        worst_trade = np.min(trade_returns)
        prev_cash = cash

    if predicted_class == 1 and predicted_class_proba > 0.6:
        asset = asset + 1
        cash = cash - closes[index]
        pendingTrade = True
        pendingTradeSide = "BUY"

    elif predicted_class == 0 and predicted_class_proba > 0.6:
        asset = asset - 1
        cash = cash + closes[index]
        pendingTrade = True
        pendingTradeSide = "SELL"

    if index == final_index:
        if asset == -1:
            asset = asset + 1
            cash = cash - closes[index]
        elif asset == 1:
            asset = asset - 1
            cash = cash + closes[index]
        trade_returns.append((cash - prev_cash))
        if cash < max_drawdown:
            max_drawdown = cash
        # if (cash - prev_cash < 0) and ((cash - prev_cash) < worst_trade):
        #     worst_trade = cash - prev_cash
        worst_trade = np.min(trade_returns)
        prev_cash = cash

    if do_detail_backtest:
        print("\n")
        print("Day n° " + str(index) + " of trading : ")
        print("Asset Price : " + str(closes[index]))
        print("Predicted Class : " + str(predicted_class))
        print("Probability of Predicted Class : " + str(predicted_class_proba))
        print("Asset : " + str(asset))
        print("Cash : " + str(cash))
        print("\n")

backtest_file = open("OutputResults/Backtests/RandomForest-Algo1.txt", 'w')
sys.stdout = backtest_file

print("Random Forest Trading Algorithm")
print("Algo 1 : 1D trades only")
print("Final Cash-Out of the trading strategy : " + str(cash))
print("ROI of the trading strategy : " + str(round(
    cash / closes.mean() * 100, 2)) + "%")
print("Max Draw Down of the trading strategy : " + str(max_drawdown))
print("Worst Trade PnL : " + str(worst_trade))
print("Average Return/Trade : " + str(np.mean(trade_returns)))
print("Trade Returns Volatility : " + str(np.std(trade_returns)))
print("Prediction score of the model : " + str(
    pd.Series(prediction_score).value_counts()[1]/pd.Series(prediction_score).value_counts().sum()))

backtest_file.close()
sys.stdout = stdout

with open("OutputResults/Backtests/RandomForest-Algo1.txt", 'r') as backtest_file:
    print(backtest_file.read() + "\n")

# Algo 2 : On déboucle la position quand on recois une prédiction contraire
# Par exemple, si on possède l'actif (on l'a acheté), alors on attend une prédiction (un signal) de vente, et vice-versa
# Nos positions peuvent donc durer plusieurs jours, voire même une infinité de temps !
########################################################################################################################

training_set = dataset[dataset.index <= int(0.75*dataset_size)]
test_set = dataset[dataset.index > int(0.75*dataset_size)]
training_targets = targets[targets.index <= int(0.75 * dataset_size)]
test_targets = targets[targets.index > int(0.75 * dataset_size)]

means = training_set.rolling(30).mean()
vols = training_set.rolling(30).std()

final_index = test_set.index[-1]
#rolling_data_set = training_set.iloc[-30:, :].reset_index(drop=True)

curr_rolling_mean = means.iloc[-1, :]
curr_rolling_vol = vols.iloc[-1, :]

rolling_data_set = training_set.iloc[-30:, :].loc[:, RF_opt_features_names].reset_index(drop=True)
closes = test_set.loc[:, 'close']
test_set = test_set.loc[:, RF_opt_features_names]

asset = 0
prev_cash = 0
cash = 0
max_drawdown = 0
worst_trade = 0
trade_returns = list()
prev_predicted_class = 0
prediction_score = list()
good_prediction = 0
curr_close = 0
prev_close = 0
for one_set in test_set.iterrows():
    index = one_set[0]
    one_set = one_set[1]
    rolling_data_set = rolling_data_set.append(one_set)
    rolling_data_set = rolling_data_set.drop(0)
    rolling_data_set = rolling_data_set.reset_index(drop=True)
    curr_mean = rolling_data_set.mean()

    curr_vol = rolling_data_set.std()

    one_scaled_set = (one_set - curr_mean) / curr_vol

    predicted_class = RF_opt_model.predict(one_scaled_set.values.reshape(1, -1))
    class_probas = RF_opt_model.predict_proba(one_scaled_set.values.reshape(1, -1))
    predicted_class_proba = class_probas[0][int(predicted_class)]

    curr_close = closes[index]
    if curr_close > prev_close and prev_predicted_class == 1:
        good_prediction = 1
    elif curr_close > prev_close and prev_predicted_class == 0:
        good_prediction = 0
    elif curr_close < prev_close and prev_predicted_class == 0:
        good_prediction = 1
    elif curr_close < prev_close and prev_predicted_class == 1:
        good_prediction = 0

    prediction_score.append(good_prediction)
    prev_predicted_class = predicted_class
    prev_close = curr_close

    if predicted_class == 1 and asset in [-1, 0] and predicted_class_proba > 0.6:
        asset = asset + 1
        cash = cash - closes[index]

        if asset == 0:
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            #if (cash - prev_cash < 0) and (cash - prev_cash < worst_trade):
            #    worst_trade = cash - prev_cash
            worst_trade = np.min(trade_returns)
            prev_cash = cash

    elif predicted_class == 0 and asset in [0, 1] and predicted_class_proba > 0.6:
        asset = asset - 1
        cash = cash + closes[index]
        pendingTrade = True
        pendingTradeSide = "SELL"

        if asset == 0:
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            # if (cash - prev_cash < 0) and (cash - prev_cash < worst_trade):
            #     worst_trade = cash - prev_cash
            worst_trade = np.min(trade_returns)
            prev_cash = cash

    if index == final_index:
        if asset == -1:
            asset = asset + 1
            cash = cash - closes[index]
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            prev_cash = cash
            worst_trade = np.min(trade_returns)
        elif asset == 1:
            asset = asset - 1
            cash = cash + closes[index]
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            prev_cash = cash
            worst_trade = np.min(trade_returns)

    if do_detail_backtest:
        print("\n")
        print("Day n° " + str(index) + " of trading : ")
        print("Asset Price : " + str(closes[index]))
        print("Predicted Class : " + str(predicted_class))
        print("Probability of Predicted Class : " + str(predicted_class_proba))
        print("Asset : " + str(asset))
        print("Cash : " + str(cash))
        print("\n")

backtest_file = open("OutputResults/Backtests/RandomForest-Algo2.txt", 'w')
sys.stdout = backtest_file

print("Random Forest Trading Algorithm")
print("Algo 2 : Signal trades only")
print("Final Cash-Out of the trading strategy : " + str(cash))
print("ROI of the trading strategy : " + str(round(
    cash / closes.mean() * 100, 2)) + "%")
print("Max Draw Down of the trading strategy : " + str(max_drawdown))
print("Worst Trade PnL : " + str(worst_trade))
print("Average Return/Trade : " + str(np.mean(trade_returns)))
print("Trade Returns Volatility : " + str(np.std(trade_returns)))
print("Prediction score of the model : " + str(
    pd.Series(prediction_score).value_counts()[1]/pd.Series(prediction_score).value_counts().sum()))

backtest_file.close()
sys.stdout = stdout

with open("OutputResults/Backtests/RandomForest-Algo2.txt", 'r') as backtest_file:
    print(backtest_file.read() + "\n")

# Backtest de l'algorithme de Ada Boost
##########################################

# Algo 1
###############################

training_set = dataset[dataset.index <= int(0.75*dataset_size)]
test_set = dataset[dataset.index > int(0.75*dataset_size)]
training_targets = targets[targets.index <= int(0.75 * dataset_size)]
test_targets = targets[targets.index > int(0.75 * dataset_size)]

means = training_set.rolling(30).mean()
vols = training_set.rolling(30).std()

final_index = test_set.index[-1]
#rolling_data_set = training_set.iloc[-30:, :].reset_index(drop=True)

curr_rolling_mean = means.iloc[-1, :]
curr_rolling_vol = vols.iloc[-1, :]

rolling_data_set = training_set.iloc[-30:, :].loc[:, ADA_opt_features_names].reset_index(drop=True)
closes = test_set.loc[:, 'close']
test_set = test_set.loc[:, ADA_opt_features_names]

asset = 0
prev_cash = 0
cash = 0
max_drawdown = 0
worst_trade = 0
trade_returns = list()
pendingTrade = False
pendingTradeSide = ""
prev_predicted_class = 0
prediction_score = list()
good_prediction = 0
curr_close = 0
prev_close = 0
for one_set in test_set.iterrows():
    index = one_set[0]
    one_set = one_set[1]
    rolling_data_set = rolling_data_set.append(one_set)
    rolling_data_set = rolling_data_set.drop(0)
    rolling_data_set = rolling_data_set.reset_index(drop=True)
    curr_mean = rolling_data_set.mean()
    curr_vol = rolling_data_set.std()

    one_scaled_set = (one_set - curr_mean) / curr_vol

    predicted_class = ADA_opt_model.predict(one_scaled_set.reshape(1, -1))
    class_probas = ADA_opt_model.predict_proba(one_scaled_set.reshape(1, -1))
    predicted_class_proba = class_probas[0][int(predicted_class)]

    curr_close = closes[index]
    if curr_close > prev_close and prev_predicted_class == 1:
        good_prediction = 1
    elif curr_close > prev_close and prev_predicted_class == 0:
        good_prediction = 0
    elif curr_close < prev_close and prev_predicted_class == 0:
        good_prediction = 1
    elif curr_close < prev_close and prev_predicted_class == 1:
        good_prediction = 0

    prediction_score.append(good_prediction)
    prev_predicted_class = predicted_class
    prev_close = curr_close

    if pendingTrade:
        if pendingTradeSide == 'BUY':
            asset = asset - 1
            cash = cash + closes[index]
        elif pendingTradeSide == 'SELL':
            asset = asset + 1
            cash = cash - closes[index]
        pendingTrade = False
        trade_returns.append((cash - prev_cash))
        if cash < max_drawdown:
            max_drawdown = cash
        # if (cash - prev_cash < 0) and (cash - prev_cash < worst_trade):
        #     worst_trade = cash - prev_cash
        worst_trade = np.min(trade_returns)
        prev_cash = cash

    #if predicted_class == 1 and predicted_class_proba > 0.65:
    if predicted_class == 1:
        asset = asset + 1
        cash = cash - closes[index]
        pendingTrade = True
        pendingTradeSide = "BUY"

    #elif predicted_class == 0 and predicted_class_proba > 0.65:
    elif predicted_class == 0:
        asset = asset - 1
        cash = cash + closes[index]
        pendingTrade = True
        pendingTradeSide = "SELL"

    if index == final_index:
        if asset == -1:
            asset = asset + 1
            cash = cash - closes[index]
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            worst_trade = np.min(trade_returns)
            prev_cash = cash
        elif asset == 1:
            asset = asset - 1
            cash = cash + closes[index]
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            worst_trade = np.min(trade_returns)
            prev_cash = cash

    if do_detail_backtest:
        print("\n")
        print("Day n° " + str(index) + " of trading : ")
        print("Asset Price : " + str(closes[index]))
        print("Predicted Class : " + str(predicted_class))
        print("Probability of Predicted Class : " + str(predicted_class_proba))
        print("Asset : " + str(asset))
        print("Cash : " + str(cash))
        print("\n")

backtest_file = open("OutputResults/Backtests/AdaBoost-Algo1.txt", 'w')
sys.stdout = backtest_file

print("Ada Boost Trading Algorithm")
print("Algo 1 : 1D trades only")
print("Final Cash-Out of the trading strategy : " + str(cash))
print("ROI of the trading strategy : " + str(round(
    cash / closes.mean() * 100, 2)) + "%")
print("Max Draw Down of the trading strategy : " + str(max_drawdown))
print("Worst Trade PnL : " + str(worst_trade))
print("Average Return/Trade : " + str(np.mean(trade_returns)))
print("Trade Returns Volatility : " + str(np.std(trade_returns)))
print("Prediction score of the model : " + str(
    pd.Series(prediction_score).value_counts()[1]/pd.Series(prediction_score).value_counts().sum()))

backtest_file.close()
sys.stdout = stdout

with open("OutputResults/Backtests/AdaBoost-Algo1.txt", 'r') as backtest_file:
    print(backtest_file.read() + "\n")

# Algo 2
###############################

training_set = dataset[dataset.index <= int(0.75*dataset_size)]
test_set = dataset[dataset.index > int(0.75*dataset_size)]
training_targets = targets[targets.index <= int(0.75 * dataset_size)]
test_targets = targets[targets.index > int(0.75 * dataset_size)]

means = training_set.rolling(30).mean()
vols = training_set.rolling(30).std()

final_index = test_set.index[-1]
#rolling_data_set = training_set.iloc[-30:, :].reset_index(drop=True)

curr_rolling_mean = means.iloc[-1, :]
curr_rolling_vol = vols.iloc[-1, :]

rolling_data_set = training_set.iloc[-30:, :].loc[:, ADA_opt_features_names].reset_index(drop=True)
closes = test_set.loc[:, 'close']
test_set = test_set.loc[:, ADA_opt_features_names]

asset = 0
prev_cash = 0
cash = 0
max_drawdown = 0
worst_trade = 0
trade_returns = list()
prev_predicted_class = 0
prediction_score = list()
good_prediction = 0
curr_close = 0
prev_close = 0
for one_set in test_set.iterrows():
    index = one_set[0]
    one_set = one_set[1]
    rolling_data_set = rolling_data_set.append(one_set)
    rolling_data_set = rolling_data_set.drop(0)
    rolling_data_set = rolling_data_set.reset_index(drop=True)
    curr_mean = rolling_data_set.mean()
    curr_vol = rolling_data_set.std()

    one_scaled_set = (one_set - curr_mean) / curr_vol

    predicted_class = ADA_opt_model.predict(one_scaled_set.reshape(1, -1))
    class_probas = ADA_opt_model.predict_proba(one_scaled_set.reshape(1, -1))
    predicted_class_proba = class_probas[0][int(predicted_class)]

    curr_close = closes[index]
    if curr_close > prev_close and prev_predicted_class == 1:
        good_prediction = 1
    elif curr_close > prev_close and prev_predicted_class == 0:
        good_prediction = 0
    elif curr_close < prev_close and prev_predicted_class == 0:
        good_prediction = 1
    elif curr_close < prev_close and prev_predicted_class == 1:
        good_prediction = 0

    prediction_score.append(good_prediction)
    prev_predicted_class = predicted_class
    prev_close = curr_close

    #if predicted_class == 1 and asset in [-1, 0] and predicted_class_proba > 0.65:
    if predicted_class == 1 and asset in [-1, 0]:
        asset = asset + 1
        cash = cash - closes[index]

        if asset == 0:
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            # if (cash - prev_cash < 0) and (cash - prev_cash < worst_trade):
            #     worst_trade = cash - prev_cash
            worst_trade = np.min(trade_returns)
            prev_cash = cash

    # elif predicted_class == 0 and asset in [0, 1] and predicted_class_proba > 0.65:
    elif predicted_class == 0 and asset in [0, 1]:
        asset = asset - 1
        cash = cash + closes[index]

        if asset == 0:
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            # if (cash - prev_cash < 0) and (cash - prev_cash < worst_trade):
            #     worst_trade = cash - prev_cash
            worst_trade = np.min(trade_returns)
            prev_cash = cash

    if index == final_index:
        if asset == -1:
            asset = asset + 1
            cash = cash - closes[index]
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            worst_trade = np.min(trade_returns)
            prev_cash = cash
        elif asset == 1:
            asset = asset - 1
            cash = cash + closes[index]
            trade_returns.append((cash - prev_cash))
            if cash < max_drawdown:
                max_drawdown = cash
            worst_trade = np.min(trade_returns)
            prev_cash = cash

    if do_detail_backtest:
        print("\n")
        print("Day n° " + str(index) + " of trading : ")
        print("Asset Price : " + str(closes[index]))
        print("Predicted Class : " + str(predicted_class))
        print("Probability of Predicted Class : " + str(predicted_class_proba))
        print("Asset : " + str(asset))
        print("Cash : " + str(cash))
        print("\n")

backtest_file = open("OutputResults/Backtests/AdaBoost-Algo2.txt", 'w')
sys.stdout = backtest_file

print("Ada Boost Trading Algorithm")
print("Algo 2 : Signal trades only")
print("Final Cash-Out of the trading strategy : " + str(cash))
print("ROI of the trading strategy : " + str(round(
    cash / closes.mean() * 100, 2)) + "%")
print("Max Draw Down of the trading strategy : " + str(max_drawdown))
print("Worst Trade PnL : " + str(worst_trade))
print("Average Return/Trade : " + str(np.mean(trade_returns)))
print("Trade Returns Volatility : " + str(np.std(trade_returns)))
print("Prediction score of the model : " + str(
    pd.Series(prediction_score).value_counts()[1]/pd.Series(prediction_score).value_counts().sum()))

backtest_file.close()
sys.stdout = stdout

with open("OutputResults/Backtests/AdaBoost-Algo2.txt", 'r') as backtest_file:
    print(backtest_file.read() + "\n")