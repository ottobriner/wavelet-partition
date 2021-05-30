import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

def scatterdate(date, data, xlabel='', ylabel='', title='', c='k', s=4, 
                figsize=(12,6), filename=None):
    '''Scatters data vs date.
    Parameters
    ----------
    date : array_like
    	date of data
    data : array_like
    	data series to be plotted
    xlabel, ylabel, title, c : str
        xlabel, ylabel, title, marker color, passed to plt
    s : int
        size of marker, passed to plt
    figsize : tuple
        size of plot, passed to plt
    filename : str or None
        passed to plt.savefig
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(date, data, c=c, s=s)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show
    return

def wavelet(date, c, xlabel = 'date', ylabel = 'wavelet power', 
            title = 'wavelet details', figsize=(12,12)):
    '''Plots wavelet details vs date at all
    Parameters
    ----------
    date : array_like
    	date of original data
    c : list of ndarray
    	wavelet coefficients as returned by 'pywt.wavedec' or 'mra'
    xlabel, ylabel, title : str
        passed to plt
    figsize : tuple
        size of plot, passed to plt
    filename : str or None
        passed to plt.savefig
    '''
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

def reconst(data, c, scales, zdate, yp, xlabel = 'date', ylabel='data',
            figsize=(12,10), filename = None):
    '''Plots wavelet details and data at different time scales for specified date range
    Parameters
    ----------
    data : pd.Series with DatetimeIndex
    	original data series
    c : list of ndarray
    	wavelet coefficients as returned by 'pywt.wavedec' or 'mra'
    scales : list of int
        specified timescales in indices
    zdate : str
      	date to center zoom
    yp : ndarray
    	y linspace of fit used for normalization
    xlabel, ylabel, title : str
        passed to plt
    figsize : tuple
        size of plot, passed to plt
    filename : str or None
        passed to plt.savefig
    Notes
    -----
    This sums wavelet detail and yp to align wavelet with data for plotting. 
    I'm not sure if that's physical or meaningful.
    Time scales zoom centered on the same date.
    '''
    fig, ax = plt.subplots(len(c), 1, figsize=figsize)

    # add a big axis, hide frame
    bigax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    bigax.set_ylabel(ylabel)
    
    for i in range(len(c)):
        
        win = int(len(data.index)/((i+1)*4))
        center = data.index.get_loc(pd.to_datetime(zdate))
        
        [start, stop] = [int(center-win/2),
                         int(center+win/2)]
    
        ax[len(c)-1-i].plot(data.iloc[start:stop].index, data.iloc[start:stop], 'k.',
                          data.iloc[start:stop].index, c[i][start:stop])
    
    ax[0].set(title="{} and wavelet detail".format(ylabel))
    ax[0].legend([ylabel, 'wavelet detail'])
    ax[-1].set_xlabel(xlabel)
    plt.tight_layout()
    
    
    if filename is not None:
        plt.savefig(filename)
    
    plt.show()
    return

def scattercoef(X, Y, level=None, xlabel='', ylabel='', title='', 
                figsize = (10, 10), filename=None):
    '''Plots scatter of wavelet coefficients with fit.
    Parameters
    ----------
    X, Y : lists of ndarrays
        wavelet coefficients as returned by 'pywt.wavedec' or 'mra'
    xp, yp : ndarrays
        (x, y) linspace coordinates for fit of X vs Y
    level : int
        decomposition level to plot. if None, plots all levels
    xlabel, ylabel, title : str
        passed to plt
    figsize : tuple
        size of plot, passed to plt
    filename : str or None
        passed to plt.savefig
    '''
    
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
    
    if filename is not None:
        plt.savefig(filename)
    
    plt.show()

    

