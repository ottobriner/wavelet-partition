import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

def scatterdate(date, data, xlabel='', ylabel='', title='', figsize=(12,6), c='k', s=4):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(date, data, c=c, s=s)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show
    return

def wavelet(date, c, ylabel = 'wavelet power', xlabel = 'date', title = 'wavelet details', figsize=(12,12)):
    fig, ax = plt.subplots(len(c), 1, figsize=figsize)

    # add a big axis, hide frame
    bigax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    bigax.set_ylabel(ylabel)
    bigax.set_title(title)

    for i in range(len(c)):
        ax[i].plot(date, c[i], 'k-')

    ax[len(c)-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()
    return

def reconst(date, data, c, scales, ylabel='data', xlabel = 'date', filename = None):
    
    fig, ax = plt.subplots(len(c), 1, figsize=(12,10))

    # add a big axis, hide frame
    bigax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    bigax.set_ylabel(ylabel)
    
    for i in range(len(c)):
        ax[len(c)-1-i].plot(date[:scales[i]], data[:scales[i]], 'k.',
                          date[:scales[i]], c[i][:scales[i]] + data[:scales[i]].mean())

    ax[0].set(title="{} and wavelet detail".format(ylabel))
    ax[0].legend([ylabel, 'wavelet detail'])
    ax[-1].set_xlabel(xlabel)
    plt.tight_layout()
    
    
    if filename is not None:
        plt.savefig(filename)
    
    plt.show()
    return

def scattercoef(X, Y, level=None, xlabel='', ylabel='', title='', figsize = (10, 10)):
    '''takes two pandas DataFrames and plots scatter of wavelet coefficients'''
    fig, ax = plt.subplots(figsize=figsize)
    
    if (len(X.shape) == 1) | (len(Y.shape) == 1):
        ax.scatter(X, Y, c='k', s=1)
    elif level is not None:
        ax.scatter(X.iloc[:, level], Y.iloc[:, level], c='k', s=1)
    else:
        for j in range(X.shape[1]):
            ax.scatter(X.iloc[:, j], Y.iloc[:, j], c='k', s=1)
    
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    
    plt.tight_layout()
    plt.show()

    

