import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

df = pd.read_csv("Lotto.csv", encoding="ISO-8859-1", skiprows=1, header=None)
print(df.head())

print("\nnew section:")

df = df.reset_index(drop=True)
# Delete the first row

# Delete rows after row 2285 (index 2284 since index starts at 0)
df = df.iloc[:2285]

print(df.head())

columns_to_delete = [0, 1, 9, 10, 11]
df = df.drop(columns=columns_to_delete)
print(df.head())

column_names = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'strong']
df.columns = column_names
print("head of df\n", df.head())

X = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'strong']].shift(1).dropna()
y = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'strong']][1:].dropna()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_models = {}
rf_predictions = {}

for col in y.columns:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train[col])
    rf_models[col] = rf
    rf_predictions[col] = rf.predict(x_test)

print("Random Forest Predictions for next draw:")
for col, preds in rf_predictions.items():
    print(f"{col}: {preds[-1]}")

arima_predictions = {}

# Forecast each number column individually
for col in df.columns:
    series = df[col]
    model = ARIMA(series, order=(5, 1, 0))  # Adjust order parameters as needed
    model_fit = model.fit()

    # Forecast the next value for each column
    forecast = model_fit.forecast(steps=1)
    arima_predictions[col] = int(forecast.iloc[0])

# Print ARIMA predictions
print("ARIMA Predictions for next draw:")
print(arima_predictions)

sequence_length = 10
data = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'strong']].values

x_lstm = []
y_lstm = []

for i in range(sequence_length, len(data)):
    x_lstm.append(data[i - sequence_length:i])
    y_lstm.append(data[i])

x_lstm, y_lstm = np.array(x_lstm), np.array(y_lstm)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(x_lstm.shape[1], x_lstm.shape[2])))
model.add(Dense(7))
model.compile(optimizer='adam', loss='mse')

model.fit(x_lstm, y_lstm, epochs=100, verbose=1)

lstm_predictions = model.predict(x_lstm[-1].reshape(1, sequence_length, 7))
lstm_predictions = lstm_predictions.round().astype(int)
print("LSTM Predictions for next draw:")
print(lstm_predictions)
