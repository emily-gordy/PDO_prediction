#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:27:28 2021

@author: emilygordon
"""
#make deseasoned sst and PDO index and ohc for longlong control run 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import glob

writeout = True #boolean whether to actually write the files

# load in files
filestr1 = glob.glob('sst*.nc')[0]
filestr2 = glob.glob('ohc*.nc')[0]

ts_dataarray = xr.open_dataset(filestr1,decode_times=False)
ohc_dataarray = xr.open_dataset(filestr2,decode_times=False)

ts = ts_dataarray.sst # surface temp
lat = ts_dataarray.lat
lon = ts_dataarray.lon
time = ts_dataarray.time

slicelatmin = 20 # slices for Pacific Ocean
slicelatmax = 60
slicelonmin = 110
slicelonmax = 260

sst_PO = np.asarray(ts.sel(lat=slice(slicelatmin,slicelatmax), #st only Pacific
                                    lon=slice(slicelonmin,slicelonmax)))

sst = np.asarray(ts)
ohc = np.asarray(ohc_dataarray.ohc_100)


#%% deseason the data 

arrdims = np.shape(sst)
POshape = np.shape(sst_PO)

months = np.arange(12)

# empty to be filled with deseasoned data
sst_deseasoned = np.empty(arrdims)
PO_deseasoned = np.empty(POshape)
ohc_deseasoned = np.empty(arrdims)

# to be filled with seasonal cycle, useful to check
sstseasonal = np.empty((12,45,90))
ohcseasonal = np.empty((12,45,90))

for ii in months: #index through months
    inds = np.arange(ii,arrdims[0],12) # all indexes corresponding to that month
    
    tsmonth = sst[inds,:,:] #pull ssts of month ii
    monthmean = np.nanmean(tsmonth,axis=0) #take mean of those sst
    nomean = tsmonth-monthmean # subtract monthly mean sst from that month data
    sst_deseasoned[inds,:,:] = nomean #save to sst deseasoned
    sstseasonal[ii,:,:] = monthmean
    
    POmonth = sst_PO[inds,:,:] #pull PO ssts of month ii
    POmean = np.nanmean(POmonth,axis=0) #etc.
    POnomean = POmonth-POmean
    PO_deseasoned[inds,:,:] = POnomean
    
    ohcmonth = ohc[inds,:,:]
    ohcmean = np.nanmean(ohcmonth,axis=0)
    ohcnomean = ohcmonth-ohcmean
    ohc_deseasoned[inds,:,:] = ohcnomean
    ohcseasonal[ii,:,:] = ohcmean

#%% plot a bunch in a row to check things are going correctly **for debugginf purposes**

# for ii in np.arange(20):
#     plt.contourf(ohc_deseasoned[ii,:,:],cmap="RdBu_r")
#     plt.colorbar()
#     plt.show()

#%% now eofs to make PDO index

sstvec = np.reshape(PO_deseasoned,(arrdims[0],POshape[1]*POshape[2])) #reduce dims to time x space
maskboo = ~np.isnan(sstvec[0,:])

sstonly = sstvec[:,maskboo] # pull out the points that are ocean

covmat = np.cov(np.transpose(sstonly)) #compute covariance of deseasoned PO sst
covsparse = sparse.csc_matrix(covmat) #move to sparse array (makes calculation fast)

[evals, eof] = linalg.eigs(covsparse,k=1) #find first eval/evec which corresponds to PDO
eof = np.real(eof) # take real part of evec

if eof[0]>0: # kind of a magic number, but this ensures PDO is the right sign 
    eof = -1*eof #(positive index corresponds to positve E Pacific anomalies etc) 

#%% plot eof to check for correct shape/pattern

# replace eof into grid 
eofvec = np.empty(380)
eofvec[np.argwhere(maskboo)] = eof
eofvec[np.argwhere(~maskboo)] = np.nan
eofmat = np.reshape(eofvec,(10,38))

plt.pcolormesh(eofmat,cmap='RdBu_r')
plt.colorbar()
plt.clim(-.15,.15)
plt.show()

#%% Calculate PDO index and create xarray datasets

PDOindex = np.squeeze(np.matmul(sstonly,eof)) # PDO index is PC of eof

PDO_dataset = xr.Dataset(
    {"PDO": (("time"), PDOindex)},
    coords={
        "time": time,
    },
)

sst_dataset = xr.Dataset(
    {"sst": (("time","lat","lon"), sst_deseasoned)},
    coords={
        "time": time,
        "lat": lat,
        "lon": lon
    },
)

ohcout_dataset = xr.Dataset(
    {"ohc": (("time","lat","lon"), ohc_deseasoned)},
    coords={
        "time": time,
        "lat": lat,
        "lon": lon
    },
)
#%% save to netCDF

sststr = "PDOexperiment_2000years/data/sst-deseasoned_%d-%d.nc" %(0,24000)
PDOstr = "PDOexperiment_2000years/data/PDO_%d-%d.nc" %(0,24000)
ohcstr = "PDOexperiment_2000years/data/ohc-deseasoned_%d-%d.nc" %(0,24000)

if writeout:
    PDO_dataset.to_netcdf(PDOstr)
    sst_dataset.to_netcdf(sststr)
    ohcout_dataset.to_netcdf(ohcstr)




