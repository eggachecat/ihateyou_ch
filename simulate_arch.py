import os
import sys

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl


def tsplot(y, y_2=None, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    if y_2 is not None:
        if not isinstance(y_2, pd.Series):
            y_2 = pd.Series(y_2)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        if y_2 is not None:
            y_2.plot(ax=ts_ax, color="red")
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.show()
    return


def generate_stock_price_with_GARCH(b_0, b_1, b_2, S_0, h_0, r, c, N):
    var_ts = [h_0 ** 2]
    noise_ts = np.random.normal(0, 1, N)
    stochastic_log_price_ts = [np.log(S_0)]
    static_log_price_ts = [np.log(S_0)]

    for i in range(N):
        stochastic_log_price_ts.append(stochastic_log_price_ts[i] + r - var_ts[i] / 2 + np.sqrt(var_ts[i]) * noise_ts[i])
        static_log_price_ts.append(static_log_price_ts[i] + r)
        var_ts.append(b_0 + b_1 * var_ts[i] + b_2 * var_ts[i] ** 2 * (noise_ts[i] - c) ** 2)

    stochastic_log_price_ts = np.array(np.exp(stochastic_log_price_ts))
    static_log_price_ts = np.array(np.exp(static_log_price_ts))

    tsplot(stochastic_log_price_ts, lags=30, y_2=static_log_price_ts)


def generate_stock_price_with_normal(N, S_0=1):
    noise_ts = 2 * np.random.rand(N) - 1
    stock_price_tsa = [S_0]
    static_log_price_ts = [S_0]
    r = (1 + 1 / N)
    for i in range(N):
        stock_price = stock_price_tsa[-1]
        stock_price_tsa.append(r * stock_price + S_0 / 2 * noise_ts[i])
        static_log_price_ts.append(r * stock_price)

    tsplot(stock_price_tsa, lags=1, y_2=static_log_price_ts)

    return stock_price_tsa, static_log_price_ts


if __name__ == '__main__':
    generate_stock_price_with_normal(1000)
    exit()
    generate_stock_price_with_GARCH(0.000006575, 0.9, 0.04, 100, 0.010469 ** 2, 1 / 1000, 0, 500)
