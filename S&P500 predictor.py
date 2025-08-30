import yfinance as yf
import pandas as pd
from sklearn import ensemble, metrics


sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period = "max")
sp500.index

sp500.plot.line(y = "Close", use_index = True)

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1) #.shift(-1) will give sp500["Tomorrow"] the next days closing value

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

sp500 = sp500.loc["1990-01-01":].copy()

model = ensemble.RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)#random_state allows each random numbers to be generated to be generated in a predictable sequence, meaning we'll get the same results twice

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

preds = model.predict(test[predictors])
preds = pd.Series(preds, index = test.index)


pre_score = metrics.precision_score(test["Target"], preds)
print(f"Precision score: {pre_score:3f}")

combined = pd.concat([test["Target"], preds], axis = 1) #axis = 1 treats each input here as an individual column
combined.plot()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined



def backtest(data, model, predictors, start = 2500, step = 250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
pred_count = predictions["Predictions"].value_counts()
print(pred_count)

pre_score = metrics.precision_score(predictions["Target"], predictions["Predictions"])
print(pre_score)

value_count_updown = predictions["Target"].value_counts() / predictions.shape[0]
print(value_count_updown)

horizons = [2, 5, 60, 250, 1000] #Horizons which represents how we want to look at rolling means
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"]/rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"] #Finds on any given the day the sum of target for the past few days. I.E the sum of the past few days in which the stock price went up

    new_predictors += [ratio_column, trend_column]
    
sp500 = sp500.dropna()

model = ensemble.RandomForestClassifier(n_estimators = 200, min_samples_split = 50, random_state = 1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

predictions = backtest(sp500, model, new_predictors)
pre_value_count = predictions["Predictions"].value_counts()

print(pre_value_count)

pre_score = metrics.precision_score(predictions["Target"], predictions["Predictions"])
print(pre_score)