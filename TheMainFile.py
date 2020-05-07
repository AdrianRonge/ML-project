import mplfinance as mpf
from sklearn.linear_model import LinearRegression
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
# ----------------------------------------Preparation and reading from file---------------------------------------------
# Creating an array to support the input to neural network
data_ML = [[None for x in range(10)] for x in range(79)]
# Creating an array of data headers to plot diagrams for ML
data_ML_Headers = [None for x in range(10)]
# Creating an array of data headers to plot diagrams for all the data
data_Plots_Headers = [None for x in range(11)]
# Creating an array to support the input to plot the diagrams
data_Plots = [[None for x in range(11)] for x in range(79)]
with open('MAERSK-A-2020-01-01-2020-04-24.csv', newline='') as csvfile:
    # Skipping two first rows because they didnt contained the specific data
    next(csvfile)
    file = csv.reader(csvfile, delimiter = ';')
    # 0.Date, 1.Bid, 2.Ask, 3.Opening price, 4.Highest price, 5.Low price,
    # 6.Closing price, 7.Average price, 8.Total volume, 9.Turnover, 10.Trades
    data_Plots_Headers = next(file)
    data_Plots_Headers.remove('')
    # 0.Bid, 1.Ask, 2.Opening price, 3.Highest price, 4.Low price, 5.Closing price,
    # 6.Average price, 7.Total volume, 8.Turnover, 9.Trades
    data_ML_Headers = data_Plots_Headers [1:11].copy()
    # Old c++ habit, 78 for the last index of array since the whole array has to be reversed
    iterator = 78
    for line in file:
        # Inner iterator for puting the right data into its place in list
        inner_iterator = 0
        while(inner_iterator < 11):
            # Ensuring all data in list is in string type
            data_Plots[iterator][inner_iterator] = str(line[inner_iterator])
            if(inner_iterator > 0):
                data_Plots[iterator][inner_iterator] = float(data_Plots[iterator][inner_iterator].replace(",", "."))
            inner_iterator = inner_iterator + 1
        iterator = iterator - 1
# Closing file (good habit)
csvfile.close()
# Creating a list for Machine Learning purposes
# (if there would be no .copy() the new list would be dependant frm first one)
for x in range (79):
    data_ML[x] = data_Plots[x][1:11].copy()
# ------------------------------------------Plotting candlesticks diagram-----------------------------------------------
# If the headers in the DataFrame are not named EXACTLY like in this table the diagram will not plot
# (took me 1.5H to find that out)
headers = ['Open', 'High', 'Low', 'Close', 'Volume']
candle = [[]for x in range (79)]
dates = []
# The plot that we choose to use requires only relevant data in specific datatype
for x in range(79):
    for y in range(10):
        if(y == 3 or y == 4 or y == 5 or y == 6 or y == 8):
            candle[x].append(data_Plots[x][y])
        elif(y == 0):
            # The indexes in the DataFrame have to be in the pandas datetime format
            mydate = pd.to_datetime(datetime.strptime(data_Plots[x][y], '%Y-%m-%d').date())
            dates.append(mydate)
# Here we create a DataFrame that will be used to plot graph (it requires VERY specific format)
candle_DF = pd.DataFrame(candle,
                         columns = headers,
                         index = dates)
# Printing info about DataFrame
candle_DF.index.name = 'Date'
print("-------------------")
print(candle_DF)
print("-------------------")
# Plotting the first graph
mpf.plot(candle_DF, type = 'candle',
                    mav=(3),
                    volume = True,
                    show_nontrading = True,
                        title = 'Maersk stock price from 2020-01-01 to 2020-04-24')
# --------------------------------------------Setting up neural network-------------------------------------------------
# Creating the two sets of samples and answers in ratio 70% to 30%
answers = [None for x in range (55)]
samples = [[None for x in range (10)] for x in range (55)]
test = [[None for x in range (10)] for x in range (24)]
test_answers = [None for x in range (24)]
# Filling values
for x in range (79):
    for y in range (10):
        if(x < 55):
            samples[x][y] = data_ML[x][y]
        elif(x >= 55):
            test[x - 55][y] = data_ML[x][y]
# Filling values
another = 0
while(another <= 54):
    samples[another].pop(5)
    answers[another] = data_ML[another][5]
    another = another + 1
# Filling values
anotherone = 55
while (anotherone < 79):
    test[anotherone - 55].pop(5)
    test_answers[anotherone - 55] = data_ML[anotherone][5]
    anotherone = anotherone + 1
# Neural network with use of Linear Regression
classifier = LinearRegression()
# Teaching the neural network
brain = classifier.fit(samples, answers)
print(brain)
print("-------------------")
# Checking the accuracy of training (70%)
results = classifier.predict(samples).round(2)
# Checking the prediction capabilities (30%)
newresults = classifier.predict(test).round(2)
# ----------------------------------Displaying results of training neural network---------------------------------------
# 70% learning set graph
plt.title('70% learning set answers')
plt.xlabel('Another values of the set')
plt.ylabel('Price')
plt.plot(answers)
plt.show()
# 70 learning set results of prediction graph
plt.title('70% learning set predicted answers')
plt.xlabel('Another values of the set')
plt.ylabel('Price')
plt.plot(results)
plt.show()
# 30% test set graph
plt.title('30% test set answers')
plt.xlabel('Another values of the set')
plt.ylabel('Prices')
plt.plot(test_answers)
plt.show()
# 30 test set results of prediction graph
plt.title('30% test set predicted answers')
plt.xlabel('Another values of the set')
plt.ylabel('Prices')
plt.plot(newresults)
plt.show()
