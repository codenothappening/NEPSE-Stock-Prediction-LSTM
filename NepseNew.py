import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import time
import pandas as pd
import numpy as np
from lxml import html


from datetime import datetime, timedelta

# from final_project import options

# from final_project import df

# these are the indices which we provide in dropdown

chrome_service = Service("/path/to/chromedriver")
driver = webdriver.Chrome(service=chrome_service)

# navigate to the webpage
driver.get("https://nepsealpha.com/nepse-data")

# wait for the relevant elements to load
wait = WebDriverWait(driver, 200)
wait.until(
    EC.presence_of_element_located(
        (
            By.XPATH,
            '//*[@id="vue_app_content"]/div[3]/div/div/div/form/div/div/div[2]/input',
        )
    )
)

# calculate start and end dates

end_date = datetime.today().strftime("%m/%d/%Y")
start_date = (datetime.today() - timedelta(days=3650)).strftime("%m/%d/%Y")

# fill in the start date

start_date_input = driver.find_element(
    By.XPATH, '//*[@id="vue_app_content"]/div[3]/div/div/div/form/div/div/div[2]/input'
)
start_date_input.clear()
start_date_input.send_keys(start_date)

# fill in the end date

end_date_input = driver.find_element(
    By.XPATH, '//*[@id="vue_app_content"]/div[3]/div/div/div/form/div/div/div[3]/input'
)
end_date_input.clear()
end_date_input.send_keys(end_date)

symbol_indices_option = driver.find_elements(
    By.XPATH,
    '//*[@id="vue_app_content"]/div[3]/div/div/div/form/div/div/div[4]/select',
)

if symbol_indices_option:
    print(symbol_indices_option[0].tag_name)
    # Output: 'select'
else:
    print("Unable to find the select element")

option_elements = symbol_indices_option[0].find_elements(By.CSS_SELECTOR, "option")

options = [option.text for option in option_elements]


selected_option = st.selectbox("Select an Option", options)

st.title(selected_option)

# select the symbol or indices
symbol_indices = selected_option

# click on the symbol or indices dropdown
symbol_indices_dropdown = driver.find_element(
    By.XPATH,
    '//*[@id="vue_app_content"]/div[3]/div/div/div/form/div/div/div[4]/span/span[1]/span',
)
symbol_indices_dropdown.click()

# enter the search text and wait for results to load
search_box = driver.find_element(By.XPATH, "/html/body/span/span/span[1]/input")
search_box.send_keys(symbol_indices)
wait.until(
    EC.visibility_of_element_located(
        (
            By.CSS_SELECTOR,
            ".select2-results__option.select2-results__option--highlighted",
        )
    )
)


# select the searched item from the dropdown
search_results = driver.find_elements(
    By.CSS_SELECTOR, ".select2-results__option.select2-results__option--highlighted"
)
for result in search_results:
    if symbol_indices in result.text:
        result.click()
        break


# click on the submit button to get the data
filter_button = driver.find_element(
    By.XPATH, '//*[@id="vue_app_content"]/div[3]/div/div/div/form/div/div/div[5]/button'
)
filter_button.click()


# wait for the table to load
wait.until(
    EC.presence_of_element_located(
        (By.XPATH, '//*[@id="result-table_wrapper"]/div[1]/button[4]')
    )
)

# import the table as pandas dataframe

close_array = np.zeros(500, dtype=float)

# scrape the first 100 entries on the first page
for i in range(1, 101):
    index = i
    table_cell = driver.find_element(
        By.XPATH, '//*[@id="result-table"]/tbody/tr[{}]/td[6]'.format(index)
    )
    close_value = float(table_cell.text)
    close_array[i - 1] = close_value


# iterate through the remaining pages and scrape the close values
for j in range(1, 5):
    # click the "Next" button
    next_button = driver.find_element(By.XPATH, '//a[@class="paginate_button next"]')
    next_button.click()
    time.sleep(5)  # wait for the page to load

    # scrape the next 100 entries on the current page
    for i in range(1, 101):
        index = i
        table_cell = driver.find_element(
            By.XPATH, '//*[@id="result-table"]/tbody/tr[{}]/td[6]'.format(index)
        )
        close_value = float(table_cell.text)
        close_array[i - 1 + j * 100] = close_value
# close the web driver
# time.sleep(10)

# reverse the order of the close values
close_array = close_array[::-1]
driver.quit()

# close_array

# close the web driver
df = pd.DataFrame(close_array)

st.write(df)


# df = pd.read_csv(
#     "test.csv"
# )  # Reads the csv file and creates a pandas DataFrame onject df
df.head()  # Head() shows the first 5 rows

st.write(df.head())  # gives the Write(UI) for df.head()

# df1 = df.reset_index()["Close"]  # Load the data into a pandas DataFrame
fig, ax = plt.subplots()  # Create a plot using plt
ax.plot(df)


# Display the plot in Streamlit
st.pyplot(fig)


Ytesting = df[425:499]  # It selects rows with integer indices from 425 to 498
Ytesting

from sklearn.preprocessing import (
    MinMaxScaler,
)  # imports the MinMaxScaler class from the sklearn.preprocessing

scaler = MinMaxScaler(feature_range=(0, 1))  # sets the feature_range parameter to (0,1)
df = scaler.fit_transform(
    np.array(df).reshape(-1, 1)
)  # reshape(-1,1) to convert it from a one-dimensional array

print(df)


# Splitting our data set into train set and test set with the ratio of 70-30.
training_size = int(
    len(df) * 0.70
)  # 70% of the length of the df1 NumPy array and assigns it to the variable training_size
test_size = len(df) - training_size  # number of elements in the testing set
train_data, test_data = (
    df[0:training_size, :],
    df[training_size : len(df), :1],
)  # create two new NumPy arrays, train_data and test_data, by splitting the df
training_size, test_size  # displays the values


# convert an array of values into a dataset matrix
def create_dataset(
    dataset, time_step=1
):  # defines a function create_dataset that takes two arguments
    dataX, dataY = (
        [],
        [],
    )  # initializes two empty lists, will be used to store the input and output sequences for the LSTM model
    for i in range(
        len(dataset) - time_step - 1
    ):  # loop to iterate over each element in the dataset array
        a = dataset[i : (i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(
            dataset[i + time_step, 0]
        )  # corresponding output value is added to the dataY list
    return np.array(dataX), np.array(
        dataY
    )  # Finally the function returns dataX and dataY as NumPy arrays


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 75
X_train, y_train = create_dataset(
    train_data, time_step
)  # input and output sequences for the training set
X_test, ytest = create_dataset(
    test_data, time_step
)  # input and output sequences for the testing set
print(X_train.shape), print(
    y_train.shape
)  # shape(attribute of NumPy) shapes are printed in separate lines
print(X_test.shape), print(ytest.shape)


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], 1
)  # first argument is the number of samples in the array
X_test = X_test.reshape(
    X_test.shape[0], X_test.shape[1], 1
)  # the second argument is the number of time steps in each input sequence
print(
    X_test
)  # third argument is the number of features for each time step(one variable in the time series)


# Create the Stacked LSTM model
from tensorflow.keras.models import (
    Sequential,
)  # necessary modules from the TensorFlow Keras library
from tensorflow.keras.layers import (
    Dense,
)  # for building a sequential model with LSTM layers
from tensorflow.keras.layers import LSTM


# builds an LSTM neural network model using the TensorFlow Keras library
model = Sequential()  # Sequential class is used to create a sequential model
model.add(
    LSTM(50, return_sequences=True, input_shape=(75, 1))
)  # specifies the shape of the input sequences
model.add(
    LSTM(50, return_sequences=True)
)  # return the full sequence of outputs, instead of just the last output
model.add(LSTM(50))  # does not return sequences
model.add(Dense(1))  # single output unit, which is used to make a single prediction
model.compile(
    loss="mean_squared_error", optimizer="adam"
)  # mean_squared_error=common loss function used for regression problems

# adam= popular optimization algorithm used to train neural networks.


model.summary()  # summary of the model architecture((batch_size, time_steps, num_features))


model.fit(
    X_train,
    y_train,
    validation_data=(X_test, ytest),
    epochs=75,
    batch_size=64,
    verbose=1,
)  # trains the LSTM model using the fit method of Keras


# scaling the predicted data from the train set to its actual values
# generate predictions for the training and test datasets


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(
    train_predict
)  # inverse_transform method of the MinMaxScaler
test_predict = scaler.inverse_transform(test_predict)

print(test_predict.shape)
test_predict  # contains the predicted values for the test dataset(74 rows and 1 column)

len(train_predict)  # LSTM model made 273 predictions for the training dataset(273 rows)

# plotting the data predicted, blue denotes the dataset values ,
# orange denotes the train-set and the green denotes the predicted data set of the test data
look_back = 75
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back : len(train_predict) + look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[
    len(train_predict) + (look_back * 2) + 1 : len(df) - 1, :
] = test_predict
# plot baseline and predictions

fig, ax = plt.subplots()  # Create a plot using plt
ax.plot(scaler.inverse_transform(df))
ax.plot(trainPredictPlot)
ax.plot(testPredictPlot)
# Display the plot in Streamlit
st.pyplot(fig)


# Checking the accuracy of the LSTM Model of the test-set on the basis of the train-set
from sklearn.metrics import r2_score

st.title("Accuracy")

st.write(r2_score(Ytesting, test_predict))

len(test_data)

x_input = test_data[75:].reshape(1, -1)
x_input  # 2D numpy array with one row and many columns


temp_input = list(
    x_input
)  # last 75 values from the test_data array to predict the next value in the time series.
temp_input = temp_input[0].tolist()
temp_input


# prediction of the 30 Days
from numpy import array

lst_output = []
n_steps = 75
i = 0
while i < 30:
    if len(temp_input) > 75:
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1


print(lst_output)


day_new = np.arange(
    1, 77
)  # representing the days for which we have actual stock price data
day_pred = np.arange(76, 106)

fig, ax = plt.subplots()  # Create a plot using plt
ax.plot(
    day_new, scaler.inverse_transform(df[424:])
)  # the plot of the original data and the predicted values
ax.plot(day_pred, scaler.inverse_transform(lst_output))

# Display the plot in Streamlit
st.pyplot(fig)

# Get the last value
last_value = scaler.inverse_transform(lst_output)[-1]

st.write(last_value)

# Visualize original data
fig, ax = plt.subplots()

ax.plot(scaler.inverse_transform(df))

# Visualize predicted values
ax.plot(range(len(df), len(df) + len(lst_output)), scaler.inverse_transform(lst_output))

st.pyplot(fig)
