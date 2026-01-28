import numpy as np
from scipy.optimize import curve_fit


def gompertz_4p(x, a, b, c, d):
    return a + (b - a) * np.exp(-np.exp((-c) * (x - d)))


def fitting(x, y, f, p0):
    popt, _ = curve_fit(f, x, y, p0=p0)
    return popt


def r_squared(x, y, popt, f):
    residuals = y - f(x, *popt)
    rss = np.sum(residuals**2)
    tss = np.sum((y - np.mean(y)) ** 2)
    return 1 - (rss / tss)
