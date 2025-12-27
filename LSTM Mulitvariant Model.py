import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy import split
from numpy import array
import seaborn as sns
import dataframe_image as dfi

split_val = 460

df = pd.read_csv('main_df.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_axis(df['date'], inplace=True)
df.fillna(method='ffill', inplace=True)

df_0 = df.drop(axis=1, columns=['date'])

df = df[:split_val]
df_date = df['date']


from sklearn.preprocessing import MinMaxScaler

df = df.drop(axis=1, columns=['date'])
Y = df['new_cases'].values
Y = Y.reshape((Y.shape[0], 1))
Y_copy = Y

scaler = MinMaxScaler()

scaler.fit(Y)
Y = scaler.transform(Y)

print(df)

import matplotlib.dates as mdates

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

fig = plt.figure (figsize=(12, 6))
plt.grid(color='gray', linewidth=1, alpha=0.3)

formatter = mdates.DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=61))
plt.xticks(rotation=-40)  # Set text labels and properties.
plt.plot(df_date, Y_copy, color="cornflowerblue", label="Nya fall per dag")
plt.plot(df["new_cases_smoothed"], color="blue", label="Antal fall trendlinje")

plt.title('Dagliga Covid-19 Fall Normaliserat')
plt.xlabel('Datum')
plt.ylabel('Antal Fall')
plt.legend()

plt.show()


# Correlations of numerical values
df.corr()['new_cases'].sort_values()


df = df[:split_val]
df_date = df_date[:split_val]

n_input = 14
n_out = 14
n_features = len(df.columns)

train, test = df[:-n_input], df[-n_input * 2:]  # 75% and 25%
train.shape, test.shape



# Flattening the data
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)



def split_sequence(sequence, n_input, n_out):
    X, y = list(), list()
    in_start = 0

    sequence = sequence.reshape((sequence.shape[0], sequence.shape[1]))
    # step over the entire history one time step at a time
    for _ in range(len(sequence) + 1):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance

        if out_end < len(sequence) + 1:
            x_input = sequence[in_start:in_end, :]
            X.append(x_input)
            y.append(sequence[in_end:out_end, 0])
        # move along one time step
        in_start += 1 
    return array(X), array(y)



train_x, train_y = split_sequence(train, n_input, n_out)
test_x, test_y = split_sequence(test, n_input, n_out)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)



train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)



# Building the LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense, PReLU, TimeDistributed, RepeatVector, Dropout
from tensorflow.keras.optimizers import Adam

num_epochs, batch_size, verbose = 10, 64, 1


# Using Early stopping of the LSTM model to avoid overfitting and saving the best value using Checkpoint
model = Sequential()

mc = tf.keras.callbacks.ModelCheckpoint('best_model_multi_v2.h5', monitor='val_loss', mode='min')
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, verbose=1, patience=250, 
                                            mode="min", restore_best_weights=True)

model.add(LSTM(64, activation='PReLU', input_shape=(n_input, train_x.shape[2])))
model.add(Dropout(0.2))
model.add(RepeatVector(n_out))
model.add(LSTM(128, activation='PReLU', return_sequences=True))
model.add(LSTM(64, activation='PReLU', return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(32, activation='PReLU')))
model.add(TimeDistributed(Dense(1)))

print(model.summary())

lr = 1e-3
optimizer = Adam(learning_rate=lr, decay=lr/num_epochs)

model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())

model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, validation_data = (test_x, test_y), verbose=verbose, callbacks=[callback, mc])

# load the saved model
from keras.models import load_model
saved_model = load_model('best_model_multi_v2.h5')



# Plotting the model loss
loss_per_epoch = model.history.history['loss']
val_loss_per_epoch = model.history.history['val_loss']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure (figsize=(12, 6))
plt.grid(color='gray', linewidth=1, alpha=0.3)
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.plot(range(len(val_loss_per_epoch)), val_loss_per_epoch)
plt.title('Undersökning av det optimala värdet för epoch hyperparametern')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend(['Modellförlust', 'Valideringsförlust'])

plt.show()



prediction = saved_model.predict(test_x)

prediction = prediction.reshape((prediction.shape[0] * prediction.shape[1], prediction.shape[2]))

prediction = prediction[:, 0]

for i in range(len(prediction)):
    if prediction[i] <= 0:
        prediction[i] = 0 
        
test_y = test_y.reshape(test_y.shape[0] * test_y.shape[1])



# Comparing what the LSTM model predicted vs the actual data using Mean Squared Error and Mean Absolute Percentage Error
from sklearn.metrics import mean_squared_error

print(f'MSE: {round(mean_squared_error(test_y, prediction), 7)}')
print(f'RMSE: {round(mean_squared_error(test_y, prediction, squared=False), 7)}')



new_cases_normalized = df['new_cases'].values

new_cases_normalized = new_cases_normalized.reshape((new_cases_normalized.shape[0], 1))
new_cases_normalized_copies = np.repeat(new_cases_normalized, train_x.shape[2], axis=-1)
new_cases_normalized = scaler.transform(new_cases_normalized_copies)[:,0]



date_test = df_date[-n_input:]

prediction_date = []
for i in range(n_input):
    prediction_date.append(date_test[i])

fig = plt.figure (figsize=(12, 6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.grid(color='gray', linewidth=1, alpha=0.3)

plt.title('Normaliserad Jämförelse Av Sanningsvärde Och Hypotes')
plt.xlabel('Datum')
plt.ylabel('Antal Fall')
plt.legend(['Dagliga Fall'])

plt.xticks(rotation=-40)
formatter = mdates.DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=61))

plt.plot(df_date, new_cases_normalized, label = "Sanningsvärde")
plt.plot(prediction_date, prediction, label = "Hypotes", color="orange")

plt.show()



fig = plt.figure (figsize=(12, 6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.grid(color='gray', linewidth=1, alpha=0.3)

plt.plot(prediction_date, test_y[-n_input:], label = "Sanningsvärde")
plt.plot(prediction_date, prediction, label = "Hypotes", color="orange")

plt.title('Normaliserad Jämförelse Av Sanningsvärde Och Hypotes')
plt.xlabel('Datum')
plt.ylabel('Antal Fall')

plt.xticks(rotation=-40)
formatter = mdates.DateFormatter("%Y-%m-%d")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

plt.show()

prediction_list = df[-n_input:].values
x = scaler.transform(prediction_list)
x = x.reshape((1, prediction_list.shape[0], prediction_list.shape[1]))

forecast = saved_model.predict(x)
forecast = forecast.reshape((forecast.shape[0] * forecast.shape[1], forecast.shape[2]))

df_0 = scaler.transform(df_0)[:, 0]

print(f'MSE: {round(mean_squared_error(df_0[split_val:split_val + n_out], forecast), 7)}')

forecast_copies = np.repeat(forecast, train.shape[1], axis=-1)
forecast = scaler.inverse_transform(forecast_copies)[:, 0]

def predict_dates():
    last_date = df_date.values[-1]
    prediction_dates = pd.date_range(last_date, periods=n_out + 1).tolist()
    prediction_dates.pop(0)
    return prediction_dates

forecast_dates = predict_dates()

for i in range(len(forecast)):
    forecast[i] = abs(forecast[i])




fig = plt.figure (figsize=(12, 6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.grid(color='gray', linewidth=1, alpha=0.3)

plt.plot(df_date, df['new_cases'], label = "Sanningsvärde")
plt.plot(forecast_dates, forecast, label = "Framtidshypotes", color="red")
plt.plot(forecast_dates, Y_copy[split_val:split_val+n_out], color="gray")

plt.title('Framtida Prediktioner Med LSTM')
plt.xlabel('Datum')
plt.ylabel('Antal Fall')

plt.xticks(rotation=-40)
formatter = mdates.DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=61))

plt.show()


fig = plt.figure (figsize=(12, 6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.grid(color='gray', linewidth=1, alpha=0.3)

plt.plot(forecast_dates, forecast, label = "Framtidshypotes", color="red")
plt.plot(forecast_dates, Y_copy[split_val:split_val+n_out], color="gray")

plt.title('Framtida Prediktioner Med LSTM')
plt.xlabel('Datum')
plt.ylabel('Antal Fall')

plt.xticks(rotation=-40)
formatter = mdates.DateFormatter("%Y-%m-%d")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

plt.show()



