# Import Pandas, Numpy, Scipy, and Matplotlib Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hour = pd.read_csv('./data/hour.csv')
day = pd.read_csv('./data/day.csv')

# -------------------------
# Exploratory Data Analysis
# -------------------------

# Portion of registered to casual riders
prop_registered = float(sum(hour.registered) / sum(hour.cnt))
print('Percentage of registered to casual riders:')
print(prop_registered)

# Plot number of registered vs casual riders
sum_registered = sum(hour.registered)
sum_casual = sum(hour.casual)
df_sum = pd.DataFrame({'labels': ['Registered', 'Casual'], 'data': [sum_registered, sum_casual]})
ax = df_sum.plot.bar(x='labels', y='data', rot=0)
ax.set_ylabel('Total Count')
ax.set_xlabel('Riders')
ax.set_title('Registered vs Casual Riders')
plt.show()

# Scatter plot of casual riders on a working day versus weekend/holiday
hour_working_colored = hour.copy()
hour_working_colored["color"] = hour_working_colored.workingday.apply(lambda x: 'red' if x == 1 else 'blue')

ax = hour_working_colored.plot.scatter(x=['casual'], y=['registered'], c="color")
ax.set_xlabel('Casual Riders')
ax.set_ylabel('Registered Riders')
ax.set_title('Casual to Registered Riders on a Working vs Non-Working Day')
ax.legend(['Holiday or Weekend', 'Working Day'])
plt.show()

# Bar plot of weather to casual and registered riders
weather_1 = hour.loc[hour['weathersit'] == 1, ['casual', 'registered']]
weather_2 = hour.loc[hour['weathersit'] == 2, ['casual', 'registered']]
weather_3 = hour.loc[hour['weathersit'] == 3, ['casual', 'registered']]
weather_4 = hour.loc[hour['weathersit'] == 4, ['casual', 'registered']]

weather_casual = [sum(weather_1.casual), sum(weather_2.casual), sum(weather_3.casual), sum(weather_4.casual)]
weather_registered = [sum(weather_1.registered), sum(weather_2.registered), sum(weather_3.registered), sum(weather_4.registered)]
print('\nPercentages of registered riders depending on weather:')
print(weather_registered[0] / (weather_registered[0] + weather_casual[0]))
print(weather_registered[1] / (weather_registered[1] + weather_casual[1]))
print(weather_registered[2] / (weather_registered[2] + weather_casual[2]))
print(weather_registered[3] / (weather_registered[3] + weather_casual[3]))
weather_label = ['Clear', 'Cloudy', 'Light Precip.', 'Heavy Precip.']
plt.bar(weather_label, weather_casual, color='r')
plt.bar(weather_label, weather_registered, bottom=weather_casual, color='b')
plt.legend(['Casual', 'Registered'])
plt.xlabel('Weather')
plt.ylabel('Count')
plt.title('Count of Riders Given the Weather')
plt.show()



# ------------------
# Data Preprocessing
# ------------------

# Min-Max normalization
minmax = hour.copy()
minmax.casual = (minmax.casual - minmax.casual.min()) / (minmax.casual.max() - minmax.casual.min())
minmax.registered = (minmax.registered - minmax.registered.min()) / (minmax.registered.max() - minmax.registered.min())
minmax.cnt = (minmax.cnt - minmax.cnt.min()) / (minmax.cnt.max() - minmax.cnt.min())

# Z-Score normalization
zscore = hour.copy()
zscore.casual = (zscore.casual - zscore.casual.mean()) / zscore.casual.std()
zscore.registered = (zscore.registered - zscore.registered.mean()) / zscore.registered.std()
zscore.cnt = (zscore.cnt - zscore.cnt.mean()) / zscore.cnt.std()

# Decimal scaling
decimal = hour.copy()
digits_casual = len(str(abs(decimal.casual.max())))
digits_registered = len(str(abs(decimal.registered.max())))
digits_cnt = len(str(abs(decimal.cnt.max())))
decimal.casual = (decimal.casual / 10**digits_casual)
decimal.registered = (decimal.registered / 10**digits_registered)
decimal.cnt = (decimal.cnt / 10**digits_cnt)



# Equal width binning
eq_width_hour = hour.copy()
eq_width_bins = np.linspace(eq_width_hour.registered.min(), eq_width_hour.registered.max(), 4)
labels = ['low', 'medium', 'high']
bins = pd.cut(eq_width_hour.registered, bins=eq_width_bins, labels=labels, include_lowest=True)
_, _, bars = plt.hist(bins, bins=3)
i = 0
for bar in bars:
    bar.set_facecolor('C' + str(i))
    i = i + 1
plt.title('Equal Width Binning of Registered Riders')
plt.show()

# Frequency binning
frequency_hour = hour.copy()
bins = pd.qcut(frequency_hour.registered, q=3, precision=1, labels=labels)
_, _, bars = plt.hist(bins, bins=3)
i = 0
for bar in bars:
    bar.set_facecolor('C' + str(i))
    i = i + 1
plt.title('Frequency Binning of Registered Riders')
plt.show()



# Find skewed normalizations
skew_cnt = (3 * (np.mean(hour.cnt) - np.median(hour.cnt))) / np.std(hour.cnt)
print('\nSkews:')
print(skew_cnt)

# Natural log transformation
natlog = np.log(hour.cnt)
skew_natlog = (3 * (np.mean(natlog) - np.median(natlog))) / np.std(natlog)
print(skew_natlog)

# Square root transformation
sqrt = np.sqrt(hour.cnt)
skew_sqrt = (3 * (np.mean(sqrt) - np.median(sqrt))) / np.std(sqrt)
print(skew_sqrt)

# Inverse square root transformation
inv_sqrt = (1 / np.sqrt(hour.cnt))
skew_inv_sqrt = (3 * (np.mean(inv_sqrt) - np.median(inv_sqrt))) / np.std(inv_sqrt)
print(skew_inv_sqrt)



# -------------------
# Regression Analysis
# -------------------

# Data preprocessing -------------------------------------------------------------------------------------------------------

# Standardize the data
sqrt_cnt = np.sqrt(hour.cnt)
sqrt_registered = np.sqrt(hour.registered)
sqrt_casual = np.sqrt(hour.casual)

sdf = pd.DataFrame({ 'cnt': sqrt_cnt, 'casual': sqrt_casual, 'registered': sqrt_registered, 'workingday': hour.workingday })

# Calculate the  normalized value
sdf.casual = (sdf.casual - sdf.casual.min()) / (sdf.casual.max() - sdf.casual.min())
sdf.registered = (sdf.registered - sdf.registered.min()) / (sdf.registered.max() - sdf.registered.min())
sdf.cnt = (sdf.cnt - sdf.cnt.min()) / (sdf.cnt.max() - sdf.cnt.min())

# Analysis -----------------------------------------------------------------------------------------------------------------

weekday = sdf.loc[sdf.workingday == 0]
weekend = sdf.loc[sdf.workingday == 1]

# 0 - cnt, 1 - casual, 2 - registered, 3 - workingday

# Casual riders on a weekend

X = weekend.iloc[:, 1].values.reshape(-1, 1)
Y = weekend.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
prediction = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, prediction, color='red')
plt.xlabel('Casual Riders')
plt.ylabel('Total Count')
plt.title('Casual Riders on a Weekend')
plt.show()

# Casual riders on a weekday

X = weekday.iloc[:, 1].values.reshape(-1, 1)
Y = weekday.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
prediction = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, prediction, color='red')
plt.xlabel('Casual Riders')
plt.ylabel('Total Count')
plt.title('Casual Riders on a Weekday')
plt.show()

# Registered riders on a weekend

X = weekend.iloc[:, 2].values.reshape(-1, 1)
Y = weekend.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
prediction = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, prediction, color='red')
plt.xlabel('Registered Riders')
plt.ylabel('Total Count')
plt.title('Registered Riders on a Weekend')
plt.show()

# Registered riders on a weekday

X = weekday.iloc[:, 2].values.reshape(-1, 1)
Y = weekday.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
prediction = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, prediction, color='red')
plt.xlabel('Registered Riders')
plt.ylabel('Total Count')
plt.title('Registered Riders on a Weekday')
plt.show()

# Registered riders to casual riders on a weekend

X = weekend.iloc[:, 2].values.reshape(-1, 1)
Y = weekend.iloc[:, 1].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
prediction = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, prediction, color='red')
plt.xlabel('Registered Riders')
plt.ylabel('Casual Riders')
plt.title('Registered Riders to Casual Riders on a Weekend')
plt.show()

# Registered riders to casual riders on a weekday

X = weekday.iloc[:, 2].values.reshape(-1, 1)
Y = weekday.iloc[:, 1].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
prediction = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, prediction, color='red')
plt.xlabel('Registered Riders')
plt.ylabel('Casual Riders')
plt.title('Registered Riders to Casual Riders on a Weekday')
plt.show()