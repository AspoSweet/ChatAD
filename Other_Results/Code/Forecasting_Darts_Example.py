"""
Minimal forecasting example with the Darts library.

Used to sanity-check the forecasting environment for the cross-task
generalization experiments (FOREC in LLADBench). Replace the dataset with
your own series as needed.
"""
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ExponentialSmoothing

# Load a univariate series (Month / #Passengers columns)
df = pd.read_csv("AirPassengers.csv", delimiter=",")
series = TimeSeries.from_dataframe(df, "Month", "#Passengers")

# Hold out the last 36 months for validation
train, val = series[:-36], series[-36:]

model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples=1000)

series.plot()
prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()
plt.show()
