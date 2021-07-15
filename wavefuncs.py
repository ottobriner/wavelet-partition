import pandas as pd
import numpy as np
import pywt
from functools import partial, reduce
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def mra(data, wavelet, level=None, axis=-1, transform='swt',
        mode='periodization'):
    """Forward 1D multiresolution analysis.
    It is a projection onto the wavelet subspaces.
    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string
        Wavelet to use
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the `dwt_max_level` function.
    axis: int, optional
        Axis over which to compute the DWT. If not given, the last axis is
        used. Currently only available when ``transform='dwt'``.
    transform : {'dwt', 'swt'}
        Whether to use the DWT or SWT for the transforms.
    mode : str, optional
        Signal extension mode, see `Modes` (default: 'symmetric'). This option
        is only used when transform='dwt'.
    Returns
    -------
    [cAn, {details_level_n}, ... {details_level_1}] : list
        For more information, see the detailed description in `wavedec`
    See Also
    --------
    ``imra``, ``swt``
    Notes
    -----
    This is sometimes referred to as an additive decomposition because the
    inverse transform (``imra``) is just the sum of the coefficient arrays
    [1]_. The decomposition using ``transform='dwt'`` corresponds to section
    2.2 while that using an undecimated transform (``transform='swt'``) is
    described in section 3.2 and appendix A.
    This transform does not share the variance partition property of ``swt``
    with `norm=True`. It does however, result in coefficients that are
    temporally aligned regardless of the symmetry of the wavelet used.
    The redundancy of this transform is ``(level + 1)``.
    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551
    """
    if transform == 'swt':
        if mode != 'periodization':
            raise ValueError(
                "transform swt only supports mode='periodization'")
        kwargs = dict(wavelet=wavelet, norm=True)
        forward = partial(pywt.swt, level=level, trim_approx=True, **kwargs)
        if axis % data.ndim != data.ndim - 1:
            raise ValueError("swt only supports axis=-1")
        inverse = partial(pywt.iswt, **kwargs)
        is_swt = True
    elif transform == 'dwt':
        kwargs = dict(wavelet=wavelet, mode=mode, axis=axis)
        forward = partial(pywt.wavedec, level=level, **kwargs)
        inverse = partial(pywt.waverec, **kwargs)
        is_swt = False
    else:
        raise ValueError("unrecognized transform: {}".format(transform))

    wav_coeffs = forward(data)

    mra_coeffs = []
    nc = len(wav_coeffs)

    if is_swt:
        # replicate same zeros array to save memory
        z = np.zeros_like(wav_coeffs[0])
        tmp = [z, ] * nc
    else:
        # zero arrays have variable size in DWT case
        tmp = [np.zeros_like(c) for c in wav_coeffs]

    for j in range(nc):
        # tmp has arrays of zeros except for the jth entry
        tmp[j] = wav_coeffs[j]

        # reconstruct
        rec = inverse(tmp)
        if rec.shape != data.shape:
            # trim any excess coefficients
            rec = rec[tuple([slice(sz) for sz in data.shape])]
        mra_coeffs.append(rec)

        # restore zeros
        if is_swt:
            tmp[j] = z
        else:
            tmp[j] = np.zeros_like(tmp[j])
    return mra_coeffs


def imra(mra_coeffs):
    """Inverse 1D multiresolution analysis via summation.
    Parameters
    ----------
    mra_coeffs : list of ndarray
        Multiresolution analysis coefficients as returned by `mra`.
    Returns
    -------
    rec : ndarray
        The reconstructed signal.
    See Also
    --------
    ``mra``
    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551
    """
    return reduce(lambda x, y: x + y, mra_coeffs)

def mra8(data, wavelet='sym8', level=None, axis=-1, transform='dwt',
         mode='symmetric'):
    '''wrapper for 'mra' defaults LA8 wavelet and symmetric extension mode
    Parameters
    ----------
    data : array_like
    	data series
    wavelet, level, axis, transform, mode
    	passed to 'mra'    
    Notes
    -----
    Tries to read several series as arrays, or reads single series as array.
    '''
    kwargs = dict(wavelet=wavelet, axis=axis, transform=transform)
    f = partial(mra, level=level, **kwargs)
    
    try:
        c = [f(data[:, i]) for i in range(data.shape[-1])]
    except:
        c = f(data)
    
    return c

def sum_scales(c):
    '''Sums wavelet coefficients from adjacent time scales.
    Parameters
    ----------
    c : list of ndarrays
        wavelet coefficients as returned by 'pywt.wavedec' or 'mra'
    Notes
    -----
    Scales are summed pair-wise. an odd len(c) throws error
    '''
    
    if (len(c) % 2) ==0:
        csum = [c[i] + c[i+1] for i in range(0, len(c), 2)]
    else:
        raise ValueError('Cannot pair wavelet scales, number of scales is not even!')
    
    scales = [int(len(csum[i-1])/(i*4)) for i in range(1, len(csum))]    
    scales.insert(0, len(c[0]))
    
    return csum, scales

def read(site_id='FLX_JP-Swl', method='url'):
    '''Reads csv from file or google drive url into pandas dataframe.
    Parameters
    ----------
    site_id : str
        key to retrieve url from dict site_urls
    method : str
        source for csv
    '''
    if method == 'file':
        if site_id == 'FLX_JP-Swl':
            path = ''
        elif site_id == 'FLX_JP-BBY':
            path = 'data/FLUXNET-CH4/FLX_JP-BBY_FLUXNET-CH4_2015-2018_1-1/FLX_JP-BBY_FLUXNET-CH4_HH_2015-2018_1-1.csv'
        df = pd.read_csv(path)
        return df
    elif method == 'url':
        # TODO use dict above
        if site_id == 'FLX_JP-Swl': 
            url = "https://drive.google.com/file/d/1Pudof9T3_TOxpd5eY2F9ZjyvGxFub4Rg/view?usp=sharing"
        elif site_id == 'FLX_JP-BBY':
            url = "https://drive.google.com/file/d/1bMn9xCFZJ8Z1xVZmJ-Z8Xu0AH8xskyYs/view?usp=sharing"
        else: 
            raise ValueError("not a valid site_id!") # could be KeyError with dict
        
        file_id=url.split('/')[-2]
        dwn_url='https://drive.google.com/uc?id=' + file_id
        df = pd.read_csv(dwn_url)
        return df
    else: 
        raise ValueError('not a valid read method!')

def pd_read_from_drive(site_id='FLX_JP-Swl'):
    '''Reads csv from google drive url into pandas dataframe.
    Parameters
    ----------
    site_id : str
        key to retrieve url from dict site_urls
    '''
    
    site_urls = {
        'site_id': ['FLX_JP-Swl', 'FLX_JP-BBY'],
        'url': ['''https://drive.google.com/file/d/
                1Pudof9T3_TOxpd5eY2F9ZjyvGxFub4Rg/view?usp=sharing
                ''',
                '''https://drive.google.com/file/d/
                1bMn9xCFZJ8Z1xVZmJ-Z8Xu0AH8xskyYs/view?usp=sharing
                ''']
    }
    
    # TODO use dict above
    if site_id == 'FLX_JP-Swl': 
        url = "https://drive.google.com/file/d/1Pudof9T3_TOxpd5eY2F9ZjyvGxFub4Rg/view?usp=sharing"
    elif site_id == 'FLX_JP-BBY':
        url = "https://drive.google.com/file/d/1bMn9xCFZJ8Z1xVZmJ-Z8Xu0AH8xskyYs/view?usp=sharing"
    else: 
        raise ValueError("not a valid site_id!") # could be KeyError with dict
    
    file_id=url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    return pd.read_csv(dwn_url)

def norm(data, method = 'poly1', window=96):
    '''Normalizes array by polyfit or rolling mean.
    Parameters
    ----------
    data : array_like
        data series to be normalized
    method : str
        method for normalization
        poly1: 1st order polynomial
        poly2: 2nd order polynomial
        rolling: rolling mean
    window : int
        window for rolling mean
    Returns
    -------
    norm : array_like
        normalized series
    [xy, yp] : list of array_like
        [x, y] linspace coordinates of fit 
    '''
    xp = range(len(data))
    
    if method == 'linreg':
        print(data.shape)
        data = data.reshape(-1, 1)
        print(data.shape)
        print(data)
        X = np.arange(len(data))
        print(X.shape)
        LS = LinearRegression().fit(X.reshape(-1, 1), data)
        yp = LS.predict(range(len(data)))
        norm = data - yp
    elif method == 'rolling':
        yp = data.rolling(window).mean()
        norm = data - yp
        xp = range(len(yp))
    elif method == 'poly1':
        p = np.polynomial.Polynomial.fit(range(len(data)), data, 1)
        xp, yp = p.linspace(len(data))
        norm = data - yp
    elif method == 'poly2':
        p = np.polynomial.Polynomial.fit(range(len(data)), data, 2)
        xp, yp = p.linspace(len(data))
        norm = data - yp
    else:
        norm = None
    
    return norm, [xp, yp]

def proc(df):
    '''Normalizes and wavelet transforms, adds to new columns in df
    '''
    
    nM, _ = norm(df.loc[:, 'FCH4_F'].to_numpy())
    nLE, _ = norm(df.loc[:, 'LE_F'].to_numpy())
    
    df.loc[:, 'FCH4_Fn'] = nM
    df.loc[:, 'LE_Fn'] = nLE
    
    # df.loc[dfw.index, 'TA_Fn'] = nT
    
    [cM, cLE] = mra8(df.loc[:, ['FCH4_Fn', 'LE_Fn']].to_numpy(), level=7, axis=0)
    
    # sum wavelet scales
    csumM, _ = sum_scales(cM)
    csumLE, _ = sum_scales(cLE)
    
    for j in range(len(csumM)):
        df.loc[:, 'FCH4_w{}'.format(j)] = csumM[j]
        df.loc[:, 'LE_w{}'.format(j)] = csumLE[j]
    
    return df

def get_regr(df, Xcols, Ycols):
    '''Returns RMSD of linear regression between two sets of columns of df'''
    
    Xflat = np.concatenate(df[Xcols].to_numpy()).reshape(-1, 1)
    Yflat = np.concatenate(df[Ycols].to_numpy()).reshape(-1, 1)
    
    regr = LinearRegression().fit(Xflat, Yflat)
    
    pred = regr.predict(Xflat)
    rmsd = np.sqrt(mean_squared_error(Yflat, pred))    
    r2 = r2_score(Yflat, pred)
    
    return pred, [Xflat, Yflat], rmsd, r2

def part(df, pred, rmsd, r2):
    '''Partitions diffusive and ebullitive fluxes by comparing to computed RMSD
    Parameters
    ----------
    df : pandas DataFrame
        date-indexed, with fluxes to be partitioned
    Returns
    -------
    df : pandas DataFrame
        df with partitioned fluxes added
    '''
    
    cols = df.columns[df.columns.str.startswith('FCH4_w')]
    
    df.loc[:, 'pdiff'] = np.ones(len(df))
        
    for j in range(len(cols)):
        predw = pred[j*len(df):j*len(df) + len(df)]
        df.loc[:, 'diff{}'.format(j)] = np.ones(len(df))
    
        wave = df.loc[:, 'FCH4_w{}'.format(j)].to_numpy().reshape(-1, 1)
        maskmore = wave > predw - 3*rmsd
        maskless = wave < predw + 3*rmsd
        df['diff{}'.format(j)] = (maskless & maskmore).astype(int)
        
        for i in range(len(df)):
            if df.loc[df.index[i], 'diff{}'.format(j)] == 1:
                if df.loc[df.index[i], 'pdiff'] == 0.:
                    continue
                else:
                    df.loc[df.index[i], 'pdiff'] = 1.
            else:
                df.loc[df.index[i], 'pdiff'] = 0.
    # colsd = df.columns[df.columns.str.startswith('diff')]
    # df['pdiff'] = df.where(df[colsd].any(), other = 0.)
    
    df.loc[:, 'rmsd'] = np.ones(len(df)) * rmsd
    df.loc[:, 'r2'] = np.ones(len(df)) * r2
    
    return df

def chop(df):
    start = df['FCH4_F'].first_valid_index()
    windows = []

    while start < df.index[-1]:
        start = df.loc[start:, 'FCH4_F'].first_valid_index()
        window = pd.date_range(start, periods = 1920, freq = '30min')

        if window[-1] < df.index[-1]:
            if df.loc[window, 'FCH4_F'].isna().any():
                print('window {} to {} contains missing data'.format(window[0], window[-1]))
            else:
                print('appending window {} to {} to windows'.format(window[0], window[-1]))    
                windows.append(window)

        start = window[-1]
    return windows

def wave(df):
    # normalize and wavelet transform, add back to df
    dfp = proc(df)
    
    # choose columns for partitioning
    Xcols = dfp.columns[dfp.columns.str.startswith('LE_w')]
    Ycols = dfp.columns[dfp.columns.str.startswith('FCH4_w')]
    
    # calc regression
    pred, [Xflat, Yflat], rmsd, r2 = get_regr(dfp, Xcols, Ycols)
    
    # partition
    dfp = part(dfp, pred, rmsd, r2)
    
    # write rmsd back to df
#    dfp.loc[:, 'rmsd'] = dfp.loc[:, 'rmsd']
    
    return dfp
