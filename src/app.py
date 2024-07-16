import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima import auto_arima
import pickle
import warnings


class Ariba:
    data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/alternative-time-series-project/main/sales.csv')

    def __init__(self):
        self.data['date'] = self.data['date'].apply(self.date_stripper)
        self.data = self.data.set_index('date')
        self.decomp = seasonal_decompose(self.data, period=12)
        self.visualize()
        self.still = self.data.diff().dropna()
        self.train, self.test = TimeSeriesSplit(n_splits=2).split(self.still)
        self.actualize()
        with open('./models/ts.dat', 'wb') as file:
            pickle.dump(self.model, file)

    def date_stripper(self, dstr):
        return dt.datetime.strptime(dstr, '%Y-%m-%d %H:%M:%S.%f').date()

    def visualize(self):
        charts = [self.decomp.trend, self.decomp.seasonal, self.decomp.resid]
        colors = ['red', 'green', 'blue']
        for i in range(len(charts)):
            sns.lineplot(data=self.data)
            sns.lineplot(data=charts[i], color=colors[i])
            plt.show()
        plot_acf(self.data)
        plt.show()
        print("Dickey-Fuller test results:")
        dftest = adfuller(self.data, autolag="AIC")
        dfoutput = pd.Series(dftest[0:4],
                             index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        print(dfoutput)

    def actualize(self):
        self.model = auto_arima(self.data.iloc[self.test[0]], seasonal=True, trace=True, m=12)
        self.predictions = self.model.predict(len(self.test[1]))
        print(
            f'Mean Squared Error of predictions: {mean_squared_error(self.data.iloc[self.test[1]], self.predictions)}')
        sns.lineplot(data=self.data.iloc[self.test[1]])
        sns.lineplot(data=self.predictions, c='g')
        plt.show()


if __name__ == '__main__':
    Ariba()
