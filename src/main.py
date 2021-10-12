# Import Pandas, Numpy, Scipy, and Matplotlib Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats

hour = pd.read_csv('./data/hour.csv')
day = pd.read_csv('./data/day.csv')

# -------------------------
# Exploratory Data Analysis
# -------------------------

# Portion of registered to casual riders
prop_registered = float(sum(hour.registered) / sum(hour.cnt))
print(prop_registered)

# Plot number of registered vs casual riders
sum_registered = sum(hour.registered)
sum_casual = sum(hour.casual)
ax = plt.figure().add_axes([0, 0, 1, 1])
ax.bar(['Registered', 'Casual'], [sum_registered, sum_casual])
plt.show()

# Scatter plot of casual riders on a working day versus weekend/holiday
hour_working_colored = hour.copy()
hour_working_colored["color"] = hour_working_colored.workingday.apply(lambda x: 'red' if x == 1 else 'blue')

ax = hour_working_colored.plot.scatter(x=['casual'], y=['registered'], c="color")
ax.set_xlabel('Casual Riders')
ax.set_ylabel('Registered Riders')
plt.show()

# Bar plot of weather to casual and registered riders
weather_1 = hour.loc[hour['weathersit'] == 1, ['casual', 'registered']]
weather_2 = hour.loc[hour['weathersit'] == 2, ['casual', 'registered']]
weather_3 = hour.loc[hour['weathersit'] == 3, ['casual', 'registered']]
weather_4 = hour.loc[hour['weathersit'] == 4, ['casual', 'registered']]

weather_casual = [sum(weather_1.casual), sum(weather_2.casual), sum(weather_3.casual), sum(weather_4.casual)]
weather_registered = [sum(weather_1.registered), sum(weather_2.registered), sum(weather_3.registered), sum(weather_4.registered)]
weather_label = ['Clear', 'Cloudy', 'Light Precipitation', 'Heavy Precipitation']
plt.bar(weather_label, weather_casual, color='r')
plt.bar(weather_label, weather_registered, bottom=weather_casual, color='b')
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
plt.hist(bins, bins=3)
plt.show()

# Frequency binning
frequency_hour = hour.copy()
bins = pd.qcut(frequency_hour.registered, q=3, precision=1, labels=labels)
plt.hist(bins, bins=3)
plt.show()



# Find skewed normalizations
skew_cnt = (3 * (np.mean(hour.cnt) - np.median(hour.cnt))) / np.std(hour.cnt)
# print(skew_cnt)

# Natural log transformation
natlog = np.log(hour.cnt)
skew_natlog = (3 * (np.mean(natlog) - np.median(natlog))) / np.std(natlog)
# print(skew_natlog)

# Square root transformation
sqrt = np.sqrt(hour.cnt)
skew_sqrt = (3 * (np.mean(sqrt) - np.median(sqrt))) / np.std(sqrt)
# print(skew_sqrt)

# Inverse square root transformation
inv_sqrt = (1 / np.sqrt(hour.cnt))
skew_inv_sqrt = (3 * (np.mean(inv_sqrt) - np.median(inv_sqrt))) / np.std(inv_sqrt)
# print(skew_inv_sqrt)



# -------------------
# Regression Analysis
# -------------------