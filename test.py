#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pywt
import wavefuncs as wave
import plotter as p

from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from numpy.polynomial import Polynomial as P
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# global plotting settings
plt.style.use(['ggplot'])


default_cycler = (plt.rcParams['axes.prop_cycle'][:4]  +
                  cycler(linestyle=['-', '--', ':', '-.']))

plt.rcParams.update({'axes.grid' : True,
                     'axes.facecolor' : 'white',
                     'axes.edgecolor' : '.15',
                     'grid.color' : '.8',
                     'axes.prop_cycle' : default_cycler
                    })

# plt.rc('axes', prop_cycle=default_cycler)
site_id = 'JP-BBY'

df = wave.read('FLX_{}'.format(site_id), method='file')

df = df.replace(-9999, np.nan)  # replace missing with nan
df['date'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')  # parse timestamp to new column 'date'
df = df.set_index(df['date'])

windows = wave.chop(df, samples=239)

# Scales separated

for window in windows:
    dfw = df.loc[window, :]
    dfp = wave.proc(dfw, level=3, sum_scales=False)

    # choose columns for partitioning
    Xcols = dfp.columns[dfp.columns.str.startswith('LE_w')]
    Ycols = dfp.columns[dfp.columns.str.startswith('FCH4_w')]

    plt.close('all')
    fig, ax = plt.subplots(len(Xcols) // 2, 2, figsize=(12, 12), sharex=True, sharey=True)

    bigax = fig.add_subplot(111, frameon=False)
    bigax.grid(False)
    bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                      left=False, right=False)

    bigax.set(xlabel='LE wavelet coeff', ylabel='FCH4 wavelet coeff',
              title='{} {} - {}'.format(site_id, window[0], window[-1]))

    for j in range(len(Xcols)):
        # calc regression
        pred, rmsd, r2 = wave.get_regr(dfp, Xcols[j], Ycols[j])

        df.loc[dfp.index, 'pred_{}'.format(j)] = pred
        df.loc[dfp.index, 'rmsd_{}'.format(j)] = np.ones(len(dfp)) * rmsd
        df.loc[dfp.index, 'r2_{}'.format(j)] = np.ones(len(dfp)) * r2

        ax[j // 2, j % 2].plot(dfp[Xcols[j]], dfp[Ycols[j]], 'k.',
                               dfp[Xcols[j]], pred, 'r-',
                               dfp[Xcols[j]], pred + 3 * rmsd, 'r--',
                               dfp[Xcols[j]], pred - 3 * rmsd, 'r--')

    plt.tight_layout()

    filename = 'plot/20210821/no_sum/20210821_{}_short_iwata7_{}_{}_{}.jpeg'.format(site_id, window[0].year,
                                                                                    window[0].month, window[0].day)
    #     plt.savefig(filename)

    df.loc[window, Xcols] = dfp[Xcols]
    df.loc[window, Ycols] = dfp[Ycols]

    plt.close()

# filename = f'plot/20210821/no_sum/20210821_{site_id}_short_rmsd_vs_date.jpeg'
filename = None
p.date_rmsd(df, title=f'{site_id} RMSD for ~5d periods',
            filename=filename,
            figsize=(14, 6)
            )

df.loc['2018-10':'2018-12-15', [f'rmsd_{j}' for j in range(len(Xcols))]].plot(figsize=(12, 6))

df = wave.partition(df, windows, rmsd=df.loc[windows[-10], 'rmsd_1'])

# Choose window and columns to plot
window = windows[-24]
dfw = df.loc[window]

cols = df.columns[df.columns.str.startswith('FCH4_w')]

# mask for ebullitive fluxes
mask = [dfw[f'ebull_{j}'] == True for j in range(len(cols))]

plt.close('all')
fig, ax = plt.subplots(len(cols), 1, figsize=(14, 14), sharex=True, sharey=False)

bigax = fig.add_subplot(111, frameon=False)
bigax.grid(False)
bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                  left=False, right=False)
bigax.set(ylabel='FCH4 and FCH4 wavelet', title=f'{site_id} {window[0]} - {window[-1]}')

colors = list(plt.rcParams['axes.prop_cycle'])

for j in range(len(cols)):
    ax[j].plot(window, dfw[f'FCH4_F'], '.', color=colors[-1]['color'])
    ax[j].plot(window, dfw[f'FCH4_w{j}'], '.', color=colors[3]['color'], label='diffusive', markersize=1)
    ax[j].plot(dfw.loc[mask[j]].index, dfw.loc[dfw.loc[mask[j]].index, f'FCH4_w{j}'], '.', color=colors[0]['color'])
#     ax[j].plot(dfw.loc[dfw[f'ebull_{j}']==True].index, dfw.loc[dfw.loc[dfw[f'ebull_{j}']==True].index, f'FCH4_w{j}'], 'r.', label='ebullitive')

ax[-1].set(xlabel='date')

plt.tight_layout()

filename = f'plot/20210821/no_sum/20210821_FLX-BBY_{window[0].month}_partition_vs_FCH4.jpeg'
# plt.savefig(filename)

plt.show()

# event durations

fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharex=True, sharey=False)

bigax = fig.add_subplot(111, frameon=False)
bigax.grid(False)
bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                  left=False, right=False)
bigax.set(xlabel='event duration (hrs)', ylabel='count', title=f'{site_id} {window[0]} - {window[-1]}')

colors = list(plt.rcParams['axes.prop_cycle'])

for j in range(len(cols)):
    df.loc[window, f'event_{j}'] = dfw[f'ebull_{j}'].diff().ne(0).cumsum()
    event_sizes = (df.loc[dfw.loc[mask[j]].index]
                   .groupby(f'event_{j}')
                   .size()
                   .apply(lambda x: x / 2.)
                   )
    ax.plot(event_sizes.value_counts(),
            '.', label=f'Scale {j}')
ax.legend()
# ax[-1].set(xlabel = 'date')

plt.tight_layout()

filename = f'plot/20210821/no_sum/20210821_FLX-BBY_{window[0].month}_event_durations.jpeg'
# plt.savefig(filename)

plt.show()

# event fluxes

fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharex=True, sharey=True)

# bigax = fig.add_subplot(111, frameon=False)
# bigax.grid(False)
# bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False,
#               left=False, right=False)

ax.set(xlabel='summed event flux', ylabel='count', title=f'{site_id} {window[0]} - {window[-1]}')

colors = list(plt.rcParams['axes.prop_cycle'])

for j in range(len(cols)):
    events = dfw.loc[mask[j].index]
    df.loc[window, f'event_{j}'] = dfw[f'ebull_{j}'].diff().ne(0).cumsum()
    event_fluxes = (df.loc[dfw.loc[mask[j]].index]
                    .groupby(f'event_{j}')
                    .sum()
                    ['FCH4_F']
                    .round(-2)
                    )
    ax.plot(event_fluxes.value_counts(), '.',
            label=f'Scale {j}')
ax.legend()
# ax[-1].set(xlabel = 'date')

plt.tight_layout()

filename = f'plot/20210821/no_sum/20210821_FLX-BBY_{window[0].month}_event_fluxes.jpeg'
# plt.savefig(filename)

plt.show()

# # Fit WS to diffusive FCH4
# fitx_diff = dfp.loc[dfp['pdiff']==1, 'WS_F'].to_numpy().reshape(-1, 1)
# fity_diff = dfp.loc[dfp['pdiff']==1, 'FCH4_F'].to_numpy().reshape(-1, 1)
# regr_ws_diff = LinearRegression().fit(fitx_diff, fity_diff)
# pred_ws_diff = regr_ws_diff.predict(fitx_diff)
#
# # Fit WS to ebullitive FCH4
# fitx_eb = dfp.loc[dfp['pdiff']==0, 'WS_F'].to_numpy().reshape(-1, 1)
# fity_eb = dfp.loc[dfp['pdiff']==0, 'FCH4_F'].to_numpy().reshape(-1, 1)
# regr_ws_eb = LinearRegression().fit(fitx_eb, fity_eb)
# pred_ws_eb = regr_ws_eb.predict(fitx_eb)
#
# # plot windspeed vs eb and diff FCH4
# fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharey=True)
#
# ax[0].plot(fitx_diff, fity_diff, 'k.',
#            fitx_diff, pred_ws_diff, 'r-')
# ax[0].set(title = 'Diffusive', ylabel='FCH4', xlabel = 'Wind speed')
#
# ax[1].plot(fitx_eb, fity_eb, 'k.',
#            fitx_eb, pred_ws_eb, 'r-')
# ax[1].set(title = 'Ebullitive', xlabel = 'Wind speed')
#
# plt.tight_layout()
# # plt.savefig('plot/20210701_windspeed_vs_FCH4.jpeg')
# plt.show()
#
# # plot coefs with partition RMSD
# fig, ax = plt.subplots(1, 1, figsize = (12,12))
#
# ax.plot(cLEflat, cMflat, 'k.',
#         cLEflat, regr.predict(cLEflat),
#         cLEflat, regr.predict(cLEflat) + 3*rmsd, 'r--',
#         cLEflat, regr.predict(cLEflat) - 3*rmsd, 'r--')
# plt.tight_layout()
# plt.show()
#
# # make pandas df of wavelet transform
# cMpd = pd.DataFrame(csumM).transpose()
# cLEpd = pd.DataFrame(csumLE).transpose()
#
# # plot coefs with partition RMSD
# # p.coef(cLEpd, cMpd, cLEflat, pred, rmsd, xlabel='LE wavelet coeff', ylabel='FCH4 wavelet coeff')
#
# # plot by date, separated by scale, marking eb fluxes
# fig, ax = plt.subplots(4, 1, figsize=(14,14))
#
# for j in range(4):
#     ax[j].plot(dfw.index, df.loc[dfw.index, 'FCH4_F'], 'k.',
#            df.loc[df['FCH4_w{}d'.format(j)]==0].index, df.loc[df.loc[df['FCH4_w{}d'.format(j)]==0].index, 'FCH4_F'], 'r.')
#     ax[3-j].set(ylabel = '{0}-{1}h'.format(2**(2*j),2**(2*j)*2))
#
# ax[-1].set_xlabel('date')
# ax[0].legend(['diff.', 'ebull.'])
# #plt.savefig('plot/20210603_part_BBY.jpeg')
# plt.tight_layout()
# plt.show()
#
# # plot windspeed vs eb and diff fluxes
# fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)
#
# for j in range(4):
#     ax[0].plot(df.loc[df['FCH4_w{}d'.format(j)]==0, 'WS'],df.loc[df['FCH4_w{}d'.format(j)]==0, 'FCH4_F'], 'k.')
#     ax[1].plot(df.loc[df['FCH4_w{}d'.format(j)]==1, 'WS'],df.loc[df['FCH4_w{}d'.format(j)]==1, 'FCH4_F'], 'k.')
# #            df.loc[df['FCH4_w{}d'.format(j)]==0].index, df.loc[df.loc[df['FCH4_w{}d'.format(j)]==0].index, 'WS'], 'r.')
# #     ax[3-j,0].set(ylabel = '{0}-{1}h'.format(2**(2*j),2**(2*j)*2))
#
# #ax[-1].set_xlabel('date')
# #ax[0,0].legend(['diff.', 'ebull.'])
# #plt.savefig('plot/20210603_part_BBY.jpeg')
# plt.tight_layout()
# plt.show()
#
# # plot windspeed vs eb and diff flux, separated by scale
#
# fig, ax = plt.subplots(4, 2, figsize=(7,14), sharey=True)
#
# for j in range(4):
#     ax[j,0].plot(df.loc[df['FCH4_w{}d'.format(j)]==0, 'WS'],df.loc[df['FCH4_w{}d'.format(j)]==0, 'FCH4_F'], 'k.')
#     ax[j,1].plot(df.loc[df['FCH4_w{}d'.format(j)]==1, 'WS'],df.loc[df['FCH4_w{}d'.format(j)]==1, 'FCH4_F'], 'k.')
# #            df.loc[df['FCH4_w{}d'.format(j)]==0].index, df.loc[df.loc[df['FCH4_w{}d'.format(j)]==0].index, 'WS'], 'r.')
# #     ax[3-j,0].set(ylabel = '{0}-{1}h'.format(2**(2*j),2**(2*j)*2))
#
# #ax[-1].set_xlabel('date')
# ax[0,0].legend(['diff.', 'ebull.'])
# #plt.savefig('plot/20210603_part_BBY.jpeg')
# plt.tight_layout()
# plt.show()

print("success")
