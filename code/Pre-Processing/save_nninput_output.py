#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:37:11 2021

@author: emilygordon
"""

# this is specifically designed to make the NN input/output data 
# save ohc and PDO index as x month running mean with ohc also saved as 3x 4months
# apart

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import math
import random

spinuplength = 0
writeout = True

numonths = 3 # num months into NN
monthgap = 4 # gap between months in NN

run = 6 # running mean

# import files needed, these are the deseasoned OHC and the PDO index
ohcstr = "data/ohc-deseasoned_%d-%d.nc" %(spinuplength,24000)
PDOstr = "data/PDO_%d-%d.nc" %(spinuplength,24000)

PDO_dataarray = xr.open_dataset(PDOstr,decode_times=False)
ohc_dataarray = xr.open_dataset(ohcstr,decode_times=False)

lon = ohc_dataarray.lon
lat = ohc_dataarray.lat
time = ohc_dataarray.time

PDO = PDO_dataarray.PDO

ohc = ohc_dataarray.ohc


PDO = np.asarray(PDO)
ohc = np.asarray(ohc)

arrdims = np.shape(ohc)

#%% apply "run" month smoothing 
PDO_run = []
ohc_run = []

for ii in np.arange(arrdims[0]-run):
    PDO_smooth = np.mean(PDO[ii:ii+run]) # grab up to *run* samples, take mean 
    ohc_smooth = np.mean(ohc[ii:ii+run,:,:],axis=0)
    PDO_run.append(PDO_smooth) # and attach to lest
    ohc_run.append(ohc_smooth)

PDO_run = np.asarray(PDO_run)
ohc_run = np.asarray(ohc_run)
arrdims2 = np.shape(ohc_run)

#reshape ohc to vectorized grid
ohc_run = np.reshape(ohc_run,(arrdims2[0],arrdims2[1]*arrdims2[2]))
#and divide by standard deviation at each grid point, required for NN input
ohc_run = np.divide(ohc_run,np.nanstd(ohc_run,axis=0)) 

#%% make 'prediction' arrays 

#ohc pred is 3 grids of OHC, 4 months apart in vector form hence
#ohc_pred[0,:4050] is ohc vector at month 0
#ohc_pred[0,4050:8100] is ohc vector at month 4
#ohc_pred[0,8100:12150] is ohc vector at month 8

#PDO_pred is PDOindex for last month in corresponding index in ohc_pred i.e.
#PDO_pred[0] is PDO index at month 8 (using month designation above)

PDO_pred = []
ohc_pred = []

for ii in np.arange(arrdims[0]-run-(numonths*monthgap-monthgap)):
    inds = np.arange(ii,ii+numonths*monthgap,monthgap) # indexes of months selected
    
    ohcint = np.concatenate(ohc_run[inds,:])
    pdoint = PDO_run[(ii+numonths*monthgap-monthgap)]
    ohc_pred.append(ohcint)
    PDO_pred.append(pdoint)

ohc_pred = np.asarray(ohc_pred)
PDO_pred = np.asarray(PDO_pred)

arrdims3 = np.shape(PDO_pred)[0]

#%% now save as xarrays
# note OHC is 3 maps, 4 months apart (-8,-4,0) with last month (0) corresponding to PDO index
# e.g. last map of OHC[0] is PDO_pred[0] 

time = np.arange(arrdims3) # not super professional but it is self consistent *shrugs*
space = np.arange(12150) 

# save to xarrat datasets
PDO_dataset = xr.Dataset(
    {"PDO": (("time"), PDO_pred)},
    coords={
        "time": time,
    },
)

ohc_dataset = xr.Dataset(
    {"ohc": (("time","space"), ohc_pred)},
    coords={
        "time": time,
        "space": space
    },
)


#%% and save to netCDF

ohcstr = "data/ohc_nninput_3xmaps_4monthsapart_run%d_samedirection.nc" %(run)
PDOstr = "data/PDO_nninput_run%d_samedirection.nc" %(run)

if writeout:
    PDO_dataset.to_netcdf(PDOstr)
    ohc_dataset.to_netcdf(ohcstr)


#%% and just checking PDOindex is correct

for ii in range(20):
    plt.pcolormesh(np.reshape(ohc_pred[ii+23920,8100:],(45,90)),cmap='RdBu_r')
    plt.clim((-3,3))
    plt.title('index=%d' %(PDO_pred[23920+ii]))
    plt.show()

