import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm

warnings.filterwarnings('ignore')

sns.set(style='white', context='notebook', palette='deep')
plt.style.use('fivethirtyeight')

def plot_target_distribution(y, title='Target Variable'):
    # Plot Histogram
    sns.distplot(y, fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(y)
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title(title)

    fig = plt.figure()
    res = stats.probplot(y, plot=plt)
    plt.show()

    print("Skewness: %f" % y.skew())
    print("Kurtosis: %f" % y.kurt())