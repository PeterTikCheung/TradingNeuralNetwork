import asyncio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as web
import datetime as dt
import yfinance as yfin
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

crypto_currency = 'BTC'
against_currency = 'USD'

start = dt.datetime(2000, 1, 1)
end = dt.datetime.now()
yfin.pdr_override()

#data = web.get_data_yahoo(f'{crypto_currency}-{against_currency}', start = start, end = end)
data = web.get_data_yahoo(f'NVDA', start = start, end = end)

#Prepare Data
scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data = scalar.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60
future_day = 5

x_train, y_train = [], []


for x in range(prediction_days, len(scaled_data) - future_day):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x+future_day, 0])
'''

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
    '''


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#create neural network
#pip install numpy==1.19.5
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=1500, batch_size=32)

#Testing the model

test_start = dt.datetime(2022, 1, 1)
test_end = dt.datetime.now()

yfin.pdr_override()

#test_data = web.get_data_yahoo(f'{crypto_currency}-{against_currency}', start = test_start, end = test_end)
test_data = web.get_data_yahoo(f'NVDA', start = test_start, end = test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scalar.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scalar.inverse_transform(prediction_prices)



current_time = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
folder_name = f'predictions_{current_time}'
os.makedirs(folder_name)

plt.plot(actual_prices, color = 'black', label= 'Actual Prices')
plt.plot(prediction_prices, color = 'green', label= 'Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
y_range = max(actual_prices) - min(actual_prices)

# Set the y-axis tick interval to be 10% of the range
tick_interval = 0.1 * y_range

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(base=tick_interval))
last_price = actual_prices[-1]
last_predict = prediction_prices[-1][0]
plt.text(len(actual_prices)-1, last_price, f'{last_price:.2f}', ha='right', va='center')
plt.text(len(prediction_prices) - 1, last_predict, f'{last_predict:.2f}', ha='right', va='center', color='green')
plt.legend(loc='upper left')
plt.savefig(os.path.join(folder_name, 'prediction_plot.png'))
plt.show()

#Predict Next Day

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scalar.inverse_transform(prediction)

with open(os.path.join(folder_name, 'prediction_value.txt'), 'w') as file:
    file.write(f'Predicted Price: {prediction[0][0]:.2f}')

print(prediction)
