#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 09:50:37 2021

@author: emilygordon
"""

# LRP top persistence from top acc models

from keras.layers import Dense, Activation, Dropout
from keras import regularizers,optimizers,metrics,initializers
from keras.utils import to_categorical
from keras.models import Sequential

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import numpy.random as random

import innvestigate
import cartopy.crs as ccrs
from scipy.cluster.vq import kmeans,vq    
import seaborn as sns
import cmasher as cmr
from scipy import interpolate
from matplotlib.colors import ListedColormap


topseeds = [44,75,89] # bestest models

run=6 # some useful variables
gap=30

def loadmodel(random_seed): # function for loading model
    gap = 30     
    n_units = 8
    num_classes = 2    
    ridgepen = 12    
    run=6
    
    # load in the two models
    modelstr1 = '../models/PDOtransition_fromOHC_better/SingleLayer_%dunits_seed%d_ridgepen%d_run%d_samedirection_leadmonth%d.h5'  %(
        n_units, random_seed, ridgepen, run, gap)
    
    model1 = Sequential()
    model1.add(Dense(n_units, activation='relu',input_shape=(3*4050,)))
    model1.add(Dense(num_classes,activation='softmax'))
    model1.load_weights(modelstr1)
    
    trainingdatastr = modelstr1[:-3] + '_validation.nc'
    ohcPDOweight_dataset = xr.open_dataset(trainingdatastr)
    return model1, ohcPDOweight_dataset, modelstr1

#%%

# import ALL data
ohcstr = "../data/ohc_nninput_3xmaps_4monthsapart_run%d_samedirection.nc" %(run)
PDOstr = "../data/PDO_nninput_run%d_samedirection.nc" %(run)
ohcstr2 = "../data/ohc-deseasoned_0-24000.nc"

PDO_dataarray = xr.open_dataset(PDOstr,decode_times=False)
ohc_dataarray = xr.open_dataset(ohcstr,decode_times=False)
ohc2_dataarray = xr.open_dataset(ohcstr2,decode_times=False)

lat = np.asarray(ohc2_dataarray.lat)
lon = np.asarray(ohc2_dataarray.lon)

PDO = PDO_dataarray.PDO
ohc = ohc_dataarray.ohc
PDO = np.asarray(PDO)
ohc = np.asarray(ohc)

ohc[np.isnan(ohc)] = 0

#make phase lengths vector

samplesize = np.shape(PDO)[0]
PDO_now = np.copy(PDO)
PDO_now[PDO_now>0] = 1
PDO_now[PDO_now<=0] = 0

phaselength = []
jj=0

while jj<samplesize:
    kk=jj
    check = PDO_now[jj]
    while check == PDO_now[kk]:
        kk+=1
        if kk == samplesize:
            kk = np.nan
            break
    looplength = kk-jj
    phaselength.append(looplength)
    jj+=1

phaselength = np.asarray(phaselength)
phaselength = phaselength[~np.isnan(phaselength)]

samplescut = np.shape(phaselength)[0]
PDO = PDO[:samplescut]
ohc = ohc[:samplescut,:]
PDO_now = PDO_now[:samplescut]

#%% using ohc2, make ohc maps for plotting

ohcmaps = np.asarray(ohc2_dataarray.ohc) # deseasoned, monthly ohc
arrdims = np.shape(ohcmaps)
ohc_run = []

for ii in range(arrdims[0]-run):
    ohcint = np.mean(ohcmaps[ii:ii+run,:,:],axis=0)
    ohc_run.append(ohcint)

ohc_run = np.asarray(ohc_run)
ohc_run = np.divide(ohc_run,np.nanstd(ohc_run,axis=0))
# ohc_run[np.isnan(ohc_run)] = 0
ohc0 = ohc_run[(gap+8):,:,:]

allpredsize = np.shape(ohc0)[0] # gotta chop because 'gap' month leadtime introduced
phaselength = phaselength[:allpredsize]
ohc = ohc[:allpredsize,:]
PDO_now = PDO_now[:allpredsize]

#%% analyze models

PDO_pred = 1*(phaselength<=gap) # make NN output truth

nummodels = np.shape(topseeds)[0]
acctotal = np.empty(nummodels) # to save accuracy
accpersisttotal = np.empty(nummodels)

allPDOtrue = np.copy(PDO_pred)

allLRPmaps = [] # to save LRP and OHC maps
allohcmaps = []
allohc0 = []

for ii, seeds in enumerate(topseeds):
    model1, ohcPDOweight_dataset, modelstr1 = loadmodel(seeds) # load model
    valdata = np.asarray(ohcPDOweight_dataset.ohc) # predict validation data
    valsize = np.shape(valdata)[0] # size of validation data
    
    ohc_val = valdata[:,:12150]     # separate to nn input
    PDO_val = valdata[:,-3]         # output
    inzone_val = valdata[:,-2]      # inzone broken do not use except to check lengths
    phaselength_val = valdata[:,-1] # phaselength
        
    PDOguess = model1.predict(ohc_val) # predict validation data
    argPDOguess = np.argmax(PDOguess,axis=1) 
    argPDOtrue = np.copy(PDO_val) # truth validation data
    
    modelcorr = argPDOtrue == argPDOguess # boo true where model correct
    nummodelcorr = np.shape(PDO_val[modelcorr])[0] # num model correct
    
    accint = nummodelcorr/valsize    
    acctotal[ii] = accint   # save total accuracy
    
    persist = PDO_val==0 # boo True for persistence
    numpersist = np.shape(inzone_val[persist])[0] # num persistence in validation 
    numcorrpersist = np.shape(inzone_val[persist & modelcorr])[0] # num correct persistence
    
    accpersistint = numcorrpersist/numpersist # save persistence accuracy
    accpersisttotal[ii] = accpersistint
    
    # now all data
    PDOguessall = model1.predict(ohc) # predict all data
    argPDOguessall = np.argmax(PDOguessall,axis=1)
    modelcorrall = (argPDOguessall == allPDOtrue) # boo True model correct
    modelconfall = np.max(PDOguessall,axis=1) #absolute confidence
    numtrue = np.shape(PDO_pred[modelcorrall])[0] # num correct predictions
    
    truepersistence = allPDOtrue==0 # boo True for persistence
    truetrans = allPDOtrue==1 # boo True for transition
    
    posnow = PDO_now==1 # boo True for positive PDO at input
    negnow = PDO_now==0 # boo True for negative PDO at input
    
    # find cutoff of confidence for persistence
    confpercent = 50
    persistenceconf = modelconfall[modelcorrall & truepersistence & posnow]
    confcutoff = np.percentile(persistenceconf,confpercent)
    cutoffboo = modelconfall>confcutoff # confidence value for cutting of 50% least confident
    
    # extract intersection of: correct prediction, persistence, confidence at threshold
    # and positive at input
    ohcLRP = ohc[modelcorrall & truepersistence & cutoffboo & posnow,:] # for ohc nn inputs
    ohc0LRP = ohc0[modelcorrall & truepersistence & cutoffboo & posnow,:] #for ohc at output month
    
    # strip softmax and create LRP-Z analyzer
    model_nosoftmax = innvestigate.utils.model_wo_softmax(model1)
    analyzer1 = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZ(model_nosoftmax)
    
    LRPmaps = []
    predcheck1 = []
    
    for ii, mapsamp in enumerate(ohcLRP): # LRP of the above OHC maps
        sample = np.transpose(np.expand_dims(mapsamp,axis=1)) # make OHC sample correct dimensions
        pred = np.max(model1.predict(sample),axis=1) # predict sample
        LRPsample = analyzer1.analyze(sample)        # make LRP maps
        LRPsample = np.divide(LRPsample,pred)        # divide by confidence
        LRPmaps.append(np.squeeze(LRPsample))        # append together
        predcheck1.append(pred)
    

    LRPmaps = np.asarray(LRPmaps)
    predcheck1 = np.asarray(predcheck1)
    
    allLRPmaps.append(LRPmaps) # save maps from each 
    allohcmaps.append(ohcLRP)
    allohc0.append(ohc0LRP)

#%% stack 'em
allLRPmaps = np.asarray(allLRPmaps)
allLRPstacked = allLRPmaps[0]
allohcmaps = np.asarray(allohcmaps)
allohcstacked = allohcmaps[0]
allohc0 = np.asarray(allohc0)
allohc0stacked = allohc0[0]

for stacknum in range(1,2):
    allLRPstacked = np.concatenate((allLRPstacked,allLRPmaps[stacknum]))
    allohcstacked = np.concatenate((allohcstacked,allohcmaps[stacknum]))
    allohc0stacked = np.concatenate((allohc0stacked,allohc0[stacknum]))
                                    


#%% take mean of stacks, and reshape where required

LRPmean = np.mean(allLRPstacked,axis=0)
LRP30 = np.reshape(LRPmean[8100:],(45,90))
LRP30 = np.divide(LRP30,np.max(np.abs(LRP30)))

ohcmean = np.nanmean(allohcstacked,axis=0)
ohc30 = np.reshape(ohcmean[8100:],(45,90))

ohc0mean = np.nanmean(allohc0stacked,axis=0)

lrpcmap = cmr.redshift
lrpcontours = np.arange(-0.8,0.82,0.02)

ohccmap = cmr.fusion_r
ohccontours = np.arange(-1.5,1.6,0.1)

projection_used = ccrs.EqualEarth(central_longitude=180)
transform_used = ccrs.PlateCarree()

#%% edit cmap so blues are not so emphasized 
lrpcmap = cmr.redshift
lrpcmap = cmr.take_cmap_colors(lrpcmap, 400,return_fmt='float')

negcolors = lrpcmap[:201]
poscolors = lrpcmap[200:]

negcols = np.asarray(negcolors[100:])
# negcolsinterp = np.empty(200,3)
xvec = np.arange(101)
interpob = interpolate.interp1d(xvec,negcols,kind='cubic',axis=0)
xvec2 = np.arange(0,100,0.5)
negcolsinterp = interpob(xvec2)

negcolslist = []
for rownum in range(200):
    negcolslistint = (negcolsinterp[rownum,0],negcolsinterp[rownum,1],negcolsinterp[rownum,2])
    negcolslist.append(negcolslistint)

allcols = negcolslist.copy()
for rownum in range(200):
    poscolslist = poscolors[rownum]
    allcols.append(poscolslist)

cmapuse = ListedColormap(allcols)

#%% plot

plt.figure(figsize=(8,8))
a0=plt.subplot(3,1,1,projection=projection_used)
c0=a0.contourf(lon,lat,LRP30,lrpcontours,cmap=cmapuse,transform=transform_used,extend='both')
a0.coastlines(color='gray')
plt.text(-0.07,0.2,r'LRP at $\tau$=-30',rotation='vertical',fontsize=14,transform = a0.transAxes)
plt.text(0.02,0.95,'a)',fontsize=12,transform=a0.transAxes)

c0ax=plt.axes((0.84,0.68,0.03,0.28))
cbar0=plt.colorbar(c0,cax=c0ax,ticks=np.arange(1))
cbar0.ax.set_ylabel('relevance')

a1=plt.subplot(3,1,2,projection=projection_used)
c1=a1.contourf(lon,lat,ohc30,ohccontours,cmap=ohccmap,transform=transform_used,extend='both')
a1.coastlines(color='gray')
plt.text(-0.07,0.2,r'OHC at $\tau$=-30',rotation='vertical',fontsize=14,transform = a1.transAxes)
plt.text(0.02,0.95,'b)',fontsize=12,transform=a1.transAxes)

a2=plt.subplot(3,1,3,projection=projection_used)
a2.contourf(lon,lat,ohc0mean,ohccontours,cmap=ohccmap,transform=transform_used,extend='both')
a2.coastlines(color='gray')
plt.text(-0.07,0.2,r'OHC at $\tau$=0',rotation='vertical',fontsize=14,transform = a2.transAxes)
plt.text(0.02,0.95,'c)',fontsize=12,transform=a2.transAxes)

c1ax = plt.axes((0.84,0.16,0.03,0.36))
cbar1 = plt.colorbar(c1,cax=c1ax,ticks=np.arange(-2,3))
cbar1.ax.set_ylabel(r'OHC anomaly ($\sigma$)')

plt.tight_layout()
# plt.savefig('../paperfigs/LRP_persistencebetterpospos.png',dpi=300)
plt.show()

