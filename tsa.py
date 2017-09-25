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


def white_noise():
    np.random.seed(1)

    # plot of discrete white noise
    randser = np.random.normal(size=1000)
    tsplot(randser, lags=30)


def random_walk():
    np.random.seed(1)
    n_samples = 1000

    x = w = np.random.normal(size=n_samples)
    for t in range(n_samples):
        x[t] = x[t - 1] + w[t]

    _ = tsplot(x, lags=30)


def linear_model():
    w = np.random.randn(100)
    y = np.empty_like(w)

    b0 = -50.
    b1 = 25.
    for t in range(len(w)):
        y[t] = b0 + b1 * t + 100 * w[t]

    _ = tsplot(y, lags=30)


def AR_p():
    np.random.seed(1)
    n_samples = int(1000)
    a = 0.6
    x = w = np.random.normal(size=n_samples)

    for t in range(n_samples):
        x[t] = a * x[t - 1] + w[t]

    _ = tsplot(x, lags=30)

    mdl = smt.AR(x).fit(maxlag=30, ic='aic', trend='nc')
    est_order = smt.AR(x).select_order(
        maxlag=30, ic='aic', trend='nc')

    true_order = 1
    print('\nalpha estimate: {:3.5f} | best lag order = {}'
          .format(mdl.params[0], est_order))
    print('\ntrue alpha = {} | true order = {}'
          .format(a, true_order))

    n = int(1000)
    alphas = np.array([.666, -.333])
    betas = np.array([0.])

    # Python requires us to specify the zero-lag value which is 1
    # Also note that the alphas for the AR model must be negated
    # We also set the betas for the MA equal to 0 for an AR(p) model
    # For more information see the examples at statsmodels.org
    ar = np.r_[1, -alphas]
    ma = np.r_[1, betas]

    ar2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
    _ = tsplot(ar2, lags=30)

    max_lag = 10
    mdl = smt.AR(ar2).fit(maxlag=max_lag, ic='aic', trend='nc')
    est_order = smt.AR(ar2).select_order(
        maxlag=max_lag, ic='aic', trend='nc')

    true_order = 2
    print('\ncoef estimate: {:3.4f} {:3.4f} | best lag order = {}'
          .format(mdl.params[0], mdl.params[1], est_order))
    print('\ntrue coefs = {} | true order = {}'
          .format([.666, -.333], true_order))


def MA():
    n = int(1000)

    # set the AR(p) alphas equal to 0
    alphas = np.array([0.])
    betas = np.array([0.6])

    # add zero-lag and negate alphas
    ar = np.r_[1, -alphas]
    ma = np.r_[1, betas]

    ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
    _ = tsplot(ma1, lags=30)

    max_lag = 30
    mdl = smt.ARMA(ma1, order=(0, 1)).fit(
        maxlag=max_lag, method='mle', trend='nc')
    print(mdl.summary())


def ARCH():
    np.random.seed(13)

    a0 = 2
    a1 = .5

    y = w = np.random.normal(size=1000)
    Y = np.empty_like(y)

    for t in range(len(y)):
        Y[t] = w[t] * np.sqrt((a0 + a1 * y[t - 1] ** 2))

    # simulated ARCH(1) series, looks like white noise
    tsplot(Y, lags=30)


def GARCH():
    np.random.seed(2)

    a0 = 0.2
    a1 = 0.5
    b1 = 0.3

    n = 10000
    w = np.random.normal(size=n)
    eps = np.zeros_like(w)
    sigsq = np.zeros_like(w)

    for i in range(1, n):
        sigsq[i] = a0 + a1 * (eps[i - 1] ** 2) + b1 * sigsq[i - 1]
        eps[i] = w[i] * np.sqrt(sigsq[i])

    _ = tsplot(eps, lags=30)


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


if __name__ == '__main__':
    generate_stock_price_with_GARCH(0.000006575, 0.9, 0.04, 100, 0.010469 ** 2, 1 / 1000, 0, 500)
    pass
