#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pywt
import wavefuncs as wave
import plotter as p

from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from numpy.polynomial import Polynomial as P
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# global plotting settings
plt.style.use('ggplot')
# text_kwargs = dict(ha='center', va='center', fontsize=28, color='C1') 


# ## Import data and process

df = wave.pd_read_from_drive('FLX_JP-BBY') # read from google drive into pd.DataFrame


df = df.replace(-9999, np.nan) # replace missing with nan
df['date'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M') # parse timestamp to new column 'date'
df = df.set_index(df['date'])


# trim off Dec and sparse regions in Nov 

dfw = df.loc['2018-08-01':'2018-11-22']
dfw = dfw.loc[dfw['FCH4'].first_valid_index():dfw['FCH4'].last_valid_index()]
dfw.tail()

# p.date(dfw.index, dfw['FCH4'])
# p.date(dfw.index, dfw['FCH4_F'])


# gap-filling adding some info after first big spike (~mid Sept), but I'll use it and see what happens

# normalize
nM, [xpM, ypM] = wave.norm(dfw.loc[:, 'FCH4_F'].to_numpy())
nLE, [xpLE, ypLE] = wave.norm(dfw.loc[:, 'LE_F'].to_numpy())
nT, [xpT, ypT] = wave.norm(dfw.loc[:, 'TA_F'].to_numpy())

# add normalized series to df
df.loc[dfw.index, 'FCH4_Fn'] = nM
df.loc[dfw.index, 'LE_Fn'] = nLE
df.loc[dfw.index, 'TA_Fn'] = nT

# compute wavelet coefficients using MODWT
[cM, cT, cLE] = wave.mra8(df.loc[dfw.index, ['FCH4_Fn', 'TA_Fn', 'LE_Fn']].to_numpy(), level=7, axis=0)

# sum wavelet scales and calculate scale windows
csumM, scalesM = wave.sum_scales(cM)
csumT, scalesT = wave.sum_scales(cT)
csumLE, scalesLE = wave.sum_scales(cLE)

# p.wavelet(dfw.index, cM)

# p.wavelet(dfw.index, cLE)

# add wavelet tranform back into df

df.loc[dfw.index, ['FCH4_w{}'.format(j) for j in range(len(csumM))]] = np.array(csumM).T

df.loc[dfw.index, ['LE_w{}'.format(j) for j in range(len(csumLE))]] = np.array(csumLE).T

df.loc[dfw.index, ['TA_w{}'.format(j) for j in range(len(csumT))]] = np.array(csumT).T

# flatten lists of wavelet coefficients
cMflat = np.concatenate(csumM).ravel().reshape(-1, 1)
cLEflat = np.concatenate(csumLE).ravel().reshape(-1, 1)

# Fit FCH4 vs LE
regr = LinearRegression().fit(cLEflat, cMflat)

pred = regr.predict(cLEflat)
#rmsd = np.sqrt(mean_squared_error(cMflat, pred))
#rmsd

# normalize and wavelet transform, add back to df
dfp = wave.proc(df.loc[dfw.index])

# choose columns for partitioning
Xcols = dfp.columns[dfp.columns.str.startswith('LE_w')]
Ycols = dfp.columns[dfp.columns.str.startswith('FCH4_w')]

# calc regression
pred, rmsd = wave.get_regr(dfp, Xcols, Ycols)

# partition
dfp = wave.part(dfp, pred, rmsd)

# Fit WS to diffusive FCH4
fitx_diff = dfp.loc[dfp['pdiff']==1, 'WS_F'].to_numpy().reshape(-1, 1)
fity_diff = dfp.loc[dfp['pdiff']==1, 'FCH4_F'].to_numpy().reshape(-1, 1)
regr_ws_diff = LinearRegression().fit(fitx_diff, fity_diff)
pred_ws_diff = regr_ws_diff.predict(fitx_diff)

# Fit WS to ebullitive FCH4
fitx_eb = dfp.loc[dfp['pdiff']==0, 'WS_F'].to_numpy().reshape(-1, 1)
fity_eb = dfp.loc[dfp['pdiff']==0, 'FCH4_F'].to_numpy().reshape(-1, 1)
regr_ws_eb = LinearRegression().fit(fitx_eb, fity_eb)
pred_ws_eb = regr_ws_eb.predict(fitx_eb)

# plot windspeed vs eb and diff FCH4
fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharey=True)

ax[0].plot(fitx_diff, fity_diff, 'k.',
           fitx_diff, pred_ws_diff, 'r-')
ax[0].set(title = 'Diffusive', ylabel='FCH4', xlabel = 'Wind speed')

ax[1].plot(fitx_eb, fity_eb, 'k.',
           fitx_eb, pred_ws_eb, 'r-')
ax[1].set(title = 'Ebullitive', xlabel = 'Wind speed')

plt.tight_layout()
# plt.savefig('plot/20210701_windspeed_vs_FCH4.jpeg')
plt.show()

# plot coefs with partition RMSD
fig, ax = plt.subplots(1, 1, figsize = (12,12))

ax.plot(cLEflat, cMflat, 'k.',
        cLEflat, regr.predict(cLEflat),
        cLEflat, regr.predict(cLEflat) + 3*rmsd, 'r--',
        cLEflat, regr.predict(cLEflat) - 3*rmsd, 'r--')
plt.tight_layout()
plt.show()

# make pandas df of wavelet transform
cMpd = pd.DataFrame(csumM).transpose()
cLEpd = pd.DataFrame(csumLE).transpose()

# plot coefs with partition RMSD
# p.coef(cLEpd, cMpd, cLEflat, pred, rmsd, xlabel='LE wavelet coeff', ylabel='FCH4 wavelet coeff')

# plot by date, separated by scale, marking eb fluxes
fig, ax = plt.subplots(4, 1, figsize=(14,14))

for j in range(4):
    ax[j].plot(dfw.index, df.loc[dfw.index, 'FCH4_F'], 'k.',
           df.loc[df['FCH4_w{}d'.format(j)]==0].index, df.loc[df.loc[df['FCH4_w{}d'.format(j)]==0].index, 'FCH4_F'], 'r.')
    ax[3-j].set(ylabel = '{0}-{1}h'.format(2**(2*j),2**(2*j)*2))
    
ax[-1].set_xlabel('date')
ax[0].legend(['diff.', 'ebull.'])
#plt.savefig('plot/20210603_part_BBY.jpeg')
plt.tight_layout()
plt.show()

# plot windspeed vs eb and diff fluxes
fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)

for j in range(4):
    ax[0].plot(df.loc[df['FCH4_w{}d'.format(j)]==0, 'WS'],df.loc[df['FCH4_w{}d'.format(j)]==0, 'FCH4_F'], 'k.')
    ax[1].plot(df.loc[df['FCH4_w{}d'.format(j)]==1, 'WS'],df.loc[df['FCH4_w{}d'.format(j)]==1, 'FCH4_F'], 'k.')
#            df.loc[df['FCH4_w{}d'.format(j)]==0].index, df.loc[df.loc[df['FCH4_w{}d'.format(j)]==0].index, 'WS'], 'r.')
#     ax[3-j,0].set(ylabel = '{0}-{1}h'.format(2**(2*j),2**(2*j)*2))
    
#ax[-1].set_xlabel('date')
#ax[0,0].legend(['diff.', 'ebull.'])
#plt.savefig('plot/20210603_part_BBY.jpeg')
plt.tight_layout()
plt.show()

# plot windspeed vs eb and diff flux, separated by scale

fig, ax = plt.subplots(4, 2, figsize=(7,14), sharey=True)

for j in range(4):
    ax[j,0].plot(df.loc[df['FCH4_w{}d'.format(j)]==0, 'WS'],df.loc[df['FCH4_w{}d'.format(j)]==0, 'FCH4_F'], 'k.')
    ax[j,1].plot(df.loc[df['FCH4_w{}d'.format(j)]==1, 'WS'],df.loc[df['FCH4_w{}d'.format(j)]==1, 'FCH4_F'], 'k.')
#            df.loc[df['FCH4_w{}d'.format(j)]==0].index, df.loc[df.loc[df['FCH4_w{}d'.format(j)]==0].index, 'WS'], 'r.')
#     ax[3-j,0].set(ylabel = '{0}-{1}h'.format(2**(2*j),2**(2*j)*2))
    
#ax[-1].set_xlabel('date')
ax[0,0].legend(['diff.', 'ebull.'])
#plt.savefig('plot/20210603_part_BBY.jpeg')
plt.tight_layout()
plt.show()

print("success")
