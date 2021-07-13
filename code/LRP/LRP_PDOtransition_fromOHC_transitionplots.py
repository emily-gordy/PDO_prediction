#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:42:29 2021

@author: emilygordon
"""

# LRP top transitions from top acc models
# no significance just contours

from keras.layers import Dense, Activation, Dropout
from keras import regularizers,optimizers,metrics,initializers
from keras.utils import to_categorical
from keras.models import Sequential

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import numpy.random as random

from matplotlib.colors import ListedColormap
import tensorflow as tf
import innvestigate
import cartopy.crs as ccrs
from scipy.cluster.vq import kmeans,vq    
import seaborn as sns
import cmasher as cmr
from scipy import interpolate

sampleweight=1.2 # zone marker

topseeds = [44,75,89]# random seeds for best models
run=6 # running mean used
gap=30 # cutoff

month1 = 12 # transition range we focus on
month2 = 27

def loadmodel(random_seed): # function to load in model wanted
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
    return model1, ohcPDOweight_dataset, modelstr1 #output the model, validation data, and string


#%%

# import ALL data, nn input, output, and deseasoned ohc
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

# cut down to phase lengths available
samplescut = np.shape(phaselength)[0]
PDO = PDO[:samplescut]
ohc = ohc[:samplescut,:]
PDO_now = PDO_now[:samplescut]

#%% using ohc2, make ohc maps for plotting

ohcmaps = np.asarray(ohc2_dataarray.ohc) # deseasoned, monthly ohc
arrdims = np.shape(ohcmaps)
ohc_run = []

for ii in range(arrdims[0]-run): # apply 'run' running mean
    ohcint = np.mean(ohcmaps[ii:ii+run,:,:],axis=0)
    ohc_run.append(ohcint)

ohc_run = np.asarray(ohc_run)
ohc_run = np.divide(ohc_run,np.nanstd(ohc_run,axis=0)) # standardize
ohc0 = ohc_run[(gap+8):,:,:] # ohc at month0 output month

ohcnowlong = ohc_run[8:]
ohctrans = []

# ohctrans at any point is the 1st ohc grid after the next transition
for ii in range(samplescut):
    ind=phaselength[ii]
    if ~np.isnan(ind):
        ohctransint = ohcnowlong[ii+int(phaselength[ii]),:,:] 
    else:
        ohctransint = np.empty((45,90))
        ohctransint[:] = np.nan
    ohctrans.append(ohctransint)

ohctrans = np.asarray(ohctrans)
#%% ohc0 is generally too short because it is the first time we account for "gap"
# generally chop everything to the same length

ohc0 = ohc0[:samplescut,:]
ohctrans = ohctrans[:samplescut,:,:]

ohc0size = np.shape(ohc0)[0]
phaselength = phaselength[:ohc0size]
ohc = ohc[:ohc0size,:]
ohctrans=ohctrans[:ohc0size,:,:]
PDO_now = PDO_now[:ohc0size]

#%% analyze model

PDO_pred = 1*(phaselength<=gap)

nummodels = np.shape(topseeds)[0]

acctotal = np.empty(nummodels) # vector of total accuracies
acctranstotal = np.empty(nummodels)
accalltranstotal = np.empty(nummodels)
condactposneg = np.empty(nummodels)
condactnegpos = np.empty(nummodels)

allPDOtrue = np.copy(PDO_pred) #truth output copy (useful)

allLRPmapsposneg = [] #initialise arrays
allohcmapsposneg = []
allohc0posneg = []
allohctransposneg = []

allLRPmapsnegpos = []
allohcmapsnegpos = []
allohc0negpos = []
allohctransnegpos = []

allposneggrabfrom = []
allnegposgrabfrom = []

allposnegsig = np.empty((nummodels,4050))
allnegpossig = np.empty((nummodels,4050))


for loopind, seeds in enumerate(topseeds):
    print(loopind)
    
    model1, ohcPDOweight_dataset, modelstr1 = loadmodel(seeds) #load model
    valdata = np.asarray(ohcPDOweight_dataset.ohc) #load validation data
   
    valsize = np.shape(valdata)[0]
    
    ohc_val = valdata[:,:12150] #separate into input
    PDO_val = valdata[:,-3] #validation truth
    inzone_val = valdata[:,-2] #validation 12-24 zone
    phaselength_val = valdata[:,-1] #phaselength
        
    PDOguess = model1.predict(ohc_val) #make prediction of validation
    argPDOguess = np.argmax(PDOguess,axis=1)
    argPDOtrue = np.copy(PDO_val)
    
    modelcorr = argPDOtrue == argPDOguess #where model correct about validation data
    
    nummodelcorr = np.shape(PDO_val[modelcorr])[0] #num correct of validation
    
    accint = nummodelcorr/valsize    
    acctotal[loopind] = accint # save total accuracy
    
    transinzone = inzone_val==sampleweight #calculate 12-24 month accuracy
    numtranszone = np.shape(inzone_val[transinzone])[0]
    numcorrtranszone = np.shape(inzone_val[transinzone & modelcorr])[0]
    
    acctransint = numcorrtranszone/numtranszone
    acctranstotal[loopind] = acctransint #save 12-24 month accuracy
    
    alltrans = np.shape(inzone_val[PDO_val==1])[0]
    corrtrans = np.shape(inzone_val[(PDO_val==1) & modelcorr])[0]
    accalltranstotal[loopind] = (corrtrans/alltrans)
    
    #now testing on ALLLLL data (training, validation and whatever is left over)
    PDOguessall = model1.predict(ohc) #predict all data
    argPDOguessall = np.argmax(PDOguessall,axis=1) 
    
    modelcorrall = (argPDOguessall == allPDOtrue) # boo where model is correct
    modelconfall = np.max(PDOguessall,axis=1) # absolute confidence 
    numtrue = np.shape(PDO_pred[modelcorrall])[0] # total correct predictions
    
    truepersistence = allPDOtrue==0 # boolean True for persistence
    truetrans = allPDOtrue==1 # boolean True for transition
    
    posnow = PDO_now==1 # boo True where PDO phase is positive at input
    negnow = PDO_now==0 # boo True where PDO phase is negative at input
    
    numposneg = np.shape(posnow[posnow & truetrans])[0]
    numnegpos = np.shape(negnow[negnow & truetrans])[0]
    numtrueposneg = np.shape(posnow[posnow & truetrans & modelcorrall])[0]
    numtruenegpos = np.shape(negnow[negnow & truetrans & modelcorrall])[0]
    condactposneg[loopind] = numtrueposneg/numposneg
    condactnegpos[loopind] = numtruenegpos/numnegpos
    
    inzone = (phaselength>=month1) & (phaselength<=month2) # boo true where phaselength inzone
    
    # strip softmax and make LRP-Z analyzer
    model_nosoftmax = innvestigate.utils.model_wo_softmax(model1)
    analyzer1 = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZ(model_nosoftmax)
    
    # find cutoff of confidence for each transition direction
    confpercent = 50
    
    transconfposneg = modelconfall[modelcorrall & truetrans & inzone & posnow]
    confcutoffposneg = np.percentile(transconfposneg,confpercent)
    cutoffbooposneg = modelconfall>confcutoffposneg # boo for cutting lowest 50% posneg predictions
    
    transconfnegpos = modelconfall[modelcorrall & truetrans & inzone & negnow]
    confcutoff = np.percentile(transconfnegpos,confpercent)
    cutoffboonegpos = modelconfall>confcutoff # boo for cutting lowest 50% negpos predictions
    
    #grab ohc at input, transition and output for the intersection of: model is correct, transition occurs,
    # transition was in 12-24 month zone, confidence above threshold, and pos to neg transitition
    ohcLRPposneg = ohc[modelcorrall & truetrans & inzone & cutoffbooposneg & posnow,:]
    ohc0LRPposneg = ohc0[modelcorrall & truetrans & inzone & cutoffbooposneg & posnow,:,:]
    ohctransposneg = ohctrans[modelcorrall & truetrans & inzone & cutoffbooposneg & posnow,:,:]
    
    numposnegzone = np.shape(inzone[truetrans & inzone & posnow])[0]
    numnegposzone = np.shape(inzone[truetrans & inzone & negnow])[0]
    numallposnegtruezone = np.shape(inzone[truetrans & inzone & posnow & modelcorrall])[0]
    numallnegpostruezone = np.shape(inzone[truetrans & inzone & negnow & modelcorrall])[0]
    #grab ohc at input, transition and output for the intersection of: model is correct, transition occurs,
    # transition was in 12-24 month zone, confidence above threshold, and neg to pos transitition
    ohcLRPnegpos = ohc[modelcorrall & truetrans & inzone & cutoffboonegpos & negnow,:]
    ohc0LRPnegpos = ohc0[modelcorrall & truetrans & inzone & cutoffboonegpos & negnow,:,:]
    ohctransnegpos = ohctrans[modelcorrall & truetrans & inzone & cutoffboonegpos & negnow,:,:]    
    
    #maps for LRP output
    LRPmaps = []
    predcheck1 = []
    
    for ii, mapsamp in enumerate(ohc): # LRP all maps
        sample = np.transpose(np.expand_dims(mapsamp,axis=1)) # sample in correct dimensions
        pred = np.max(model1.predict(sample),axis=1) #predict sample
        LRPsample = analyzer1.analyze(sample)   # calculate LRP map
        LRPsample = np.divide(LRPsample,pred)   # divide by confidence
        LRPmaps.append(np.squeeze(LRPsample))   # append to vector
        predcheck1.append(pred)
    
    
    LRPmaps = np.asarray(LRPmaps)
    predcheck1 = np.asarray(predcheck1)
    
    #grab LRP at input for the intersection of: model is correct, transition occurs,
    # transition was in 12-24 month zone, confidence above threshold, and pos to neg (neg to pos) transitition
    LRPmapsposneg = LRPmaps[modelcorrall & truetrans & inzone & cutoffbooposneg & posnow,8100:]
    LRPmapsnegpos = LRPmaps[modelcorrall & truetrans & inzone & cutoffboonegpos & negnow,8100:]
    
    allLRPmapsposneg.append(LRPmapsposneg) # LRP and OHC maps for each model
    allohcmapsposneg.append(ohcLRPposneg)
    allohc0posneg.append(ohc0LRPposneg)
    allohctransposneg.append(ohctransposneg)
    
    allLRPmapsnegpos.append(LRPmapsnegpos)
    allohcmapsnegpos.append(ohcLRPnegpos)
    allohc0negpos.append(ohc0LRPnegpos)
    allohctransnegpos.append(ohctransnegpos)
    

#%% make map stacks from lists, the best way I could figure to do it probably could be better
allLRPmapsposneg = np.asarray(allLRPmapsposneg)
allLRPstackedposneg = allLRPmapsposneg[0]

allohcmapsposneg = np.asarray(allohcmapsposneg)
allohcstackedposneg = allohcmapsposneg[0]

allohc0posneg = np.asarray(allohc0posneg)
allohc0stackedposneg = allohc0posneg[0]

allohctransposneg = np.asarray(allohctransposneg)
allohctransstackedposneg = allohctransposneg[0]

allLRPmapsnegpos = np.asarray(allLRPmapsnegpos)
allLRPstackednegpos = allLRPmapsnegpos[0]

allohcmapsnegpos = np.asarray(allohcmapsnegpos)
allohcstackednegpos = allohcmapsnegpos[0]

allohc0negpos = np.asarray(allohc0negpos)
allohc0stackednegpos = allohc0negpos[0]

allohctransnegpos = np.asarray(allohctransnegpos)
allohctransstackednegpos = allohctransnegpos[0]


for stacknum in range(1,nummodels):
    allLRPstackedposneg = np.concatenate((allLRPstackedposneg,allLRPmapsposneg[stacknum]))
    allohcstackedposneg = np.concatenate((allohcstackedposneg,allohcmapsposneg[stacknum]))
    allohc0stackedposneg = np.concatenate((allohc0stackedposneg,allohc0posneg[stacknum]))
    allohctransstackedposneg = np.concatenate((allohctransstackedposneg,allohctransposneg[stacknum]))
    
    allLRPstackednegpos = np.concatenate((allLRPstackednegpos,allLRPmapsnegpos[stacknum]))
    allohcstackednegpos = np.concatenate((allohcstackednegpos,allohcmapsnegpos[stacknum]))
    allohc0stackednegpos = np.concatenate((allohc0stackednegpos,allohc0negpos[stacknum]))
    allohctransstackednegpos = np.concatenate((allohctransstackednegpos,allohctransnegpos[stacknum]))
    

#%% now make maps stlye means

# take mean of each stack, and reshape to lat x lon if required
LRPmeanposneg = np.mean(allLRPstackedposneg,axis=0)
LRP30posneg = np.reshape(LRPmeanposneg,(45,90))
LRP30posneg = np.divide(LRP30posneg,np.max(np.abs(LRP30posneg)))

ohcmeanposneg = np.nanmean(allohcstackedposneg,axis=0)
ohc30posneg = np.reshape(ohcmeanposneg[8100:],(45,90))

ohc0meanposneg = np.nanmean(allohc0stackedposneg,axis=0)
ohctransmeanposneg = np.nanmean(allohctransstackedposneg,axis=0)


LRPmeannegpos = np.mean(allLRPstackednegpos,axis=0)
LRP30negpos = np.reshape(LRPmeannegpos,(45,90))
LRP30negpos = np.divide(LRP30negpos,np.max(np.abs(LRP30negpos)))

ohcmeannegpos = np.nanmean(allohcstackednegpos,axis=0)
ohc30negpos = np.reshape(ohcmeannegpos[8100:],(45,90))

ohc0meannegpos = np.nanmean(allohc0stackednegpos,axis=0)
ohctransmeannegpos = np.nanmean(allohctransstackednegpos,axis=0)

#%% and plot

lat1 = 5 # for drawing boxes
lat2 = 30
lon1 = 125
lon2 = 180

splat1 = -30
splat2 = -5
splon1 = 150
splon2 = 200

#%% edit cmap to de-emphasize the blues
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
#%%

N1 = np.shape(allLRPstackedposneg)[0]
N2 = np.shape(allLRPstackednegpos)[0]

lrpcontours = np.arange(-0.7,.72,0.02)
ohccmap = cmr.fusion_r
ohccontours = np.arange(-1.5,1.51,0.1)

projection_used = ccrs.EqualEarth(central_longitude=180)
transform_used = ccrs.PlateCarree()

contourmarkers = ['dashed','solid']

contour1 = [np.percentile(LRP30negpos,5), np.percentile(LRP30posneg,95)]
contour2 = [np.percentile(LRP30negpos,5), np.percentile(LRP30negpos,95)]

plt.figure(figsize=(11.5,10))

a0=plt.subplot(4,2,1,projection=projection_used)
clrp=a0.contourf(lon,lat,LRP30posneg,lrpcontours,cmap=cmapuse,transform=transform_used,extend='both')
a0.plot([lon1,lon1],[lat2,lat1],color='xkcd:hot pink',transform=transform_used)
a0.plot([lon1,lon2],[lat1,lat1],color='xkcd:hot pink',transform=transform_used)
a0.plot([lon2,lon2],[lat2,lat1],color='xkcd:hot pink',transform=transform_used)
a0.plot([lon1,lon2],[lat2,lat2],color='xkcd:hot pink',transform=transform_used)
a0.plot([splon1,splon1],[splat2,splat1],color='xkcd:hot pink',transform=transform_used)
a0.plot([splon1,splon2],[splat1,splat1],color='xkcd:hot pink',transform=transform_used)
a0.plot([splon2,splon2],[splat2,splat1],color='xkcd:hot pink',transform=transform_used)
a0.plot([splon1,splon2],[splat2,splat2],color='xkcd:hot pink',transform=transform_used)
a0.coastlines(color='gray')
plt.title('Positive to Negative, N = %d' %(N1),fontsize=14)
plt.text(-0.07,0.2,r'LRP at $\tau$=-30',fontsize=14,rotation='vertical',transform=a0.transAxes)
plt.text(0.02,0.95,'a)',fontsize=12,transform=a0.transAxes)

a1=plt.subplot(4,2,2,projection=projection_used)
a1.contourf(lon,lat,LRP30negpos,lrpcontours,cmap=cmapuse,transform=transform_used,extend='both')
a1.plot([lon1,lon1],[lat2,lat1],color='xkcd:hot pink',transform=transform_used)
a1.plot([lon1,lon2],[lat1,lat1],color='xkcd:hot pink',transform=transform_used)
a1.plot([lon2,lon2],[lat2,lat1],color='xkcd:hot pink',transform=transform_used)
a1.plot([lon1,lon2],[lat2,lat2],color='xkcd:hot pink',transform=transform_used)
a1.plot([splon1,splon1],[splat2,splat1],color='xkcd:hot pink',transform=transform_used)
a1.plot([splon1,splon2],[splat1,splat1],color='xkcd:hot pink',transform=transform_used)
a1.plot([splon2,splon2],[splat2,splat1],color='xkcd:hot pink',transform=transform_used)
a1.plot([splon1,splon2],[splat2,splat2],color='xkcd:hot pink',transform=transform_used)
a1.coastlines(color='gray')
plt.title('Negative to Positive N = %d' %(N2),fontsize=14)
plt.text(0.02,0.95,'b)',fontsize=12,transform=a1.transAxes)


a2=plt.subplot(4,2,3,projection=projection_used)
cohc=a2.contourf(lon,lat,ohc30posneg,ohccontours,cmap=ohccmap,transform=transform_used,extend='both')
a2.contour(lon,lat,LRP30posneg,contour1,linestyles=contourmarkers,colors='xkcd:dark gray',transform=transform_used)
a2.plot([lon1,lon1],[lat2,lat1],color='xkcd:hot pink',transform=transform_used)
a2.plot([lon1,lon2],[lat1,lat1],color='xkcd:hot pink',transform=transform_used)
a2.plot([lon2,lon2],[lat2,lat1],color='xkcd:hot pink',transform=transform_used)
a2.plot([lon1,lon2],[lat2,lat2],color='xkcd:hot pink',transform=transform_used)
a2.plot([splon1,splon1],[splat2,splat1],color='xkcd:hot pink',transform=transform_used)
a2.plot([splon1,splon2],[splat1,splat1],color='xkcd:hot pink',transform=transform_used)
a2.plot([splon2,splon2],[splat2,splat1],color='xkcd:hot pink',transform=transform_used)
a2.plot([splon1,splon2],[splat2,splat2],color='xkcd:hot pink',transform=transform_used)
a2.coastlines(color='gray')
plt.text(-0.07,0.2,r'OHC at $\tau$=-30',fontsize=14,rotation='vertical',transform=a2.transAxes)
plt.text(0.02,0.95,'c)',fontsize=12,transform=a2.transAxes)


a3=plt.subplot(4,2,4,projection=projection_used)
a3.contourf(lon,lat,ohc30negpos,ohccontours,cmap=ohccmap,transform=transform_used,extend='both')
a3.contour(lon,lat,LRP30negpos,contour2,linestyles=contourmarkers,colors='xkcd:dark gray',transform=transform_used)
a3.plot([lon1,lon1],[lat2,lat1],color='xkcd:hot pink',transform=transform_used)
a3.plot([lon1,lon2],[lat1,lat1],color='xkcd:hot pink',transform=transform_used)
a3.plot([lon2,lon2],[lat2,lat1],color='xkcd:hot pink',transform=transform_used)
a3.plot([lon1,lon2],[lat2,lat2],color='xkcd:hot pink',transform=transform_used)
a3.plot([splon1,splon1],[splat2,splat1],color='xkcd:hot pink',transform=transform_used)
a3.plot([splon1,splon2],[splat1,splat1],color='xkcd:hot pink',transform=transform_used)
a3.plot([splon2,splon2],[splat2,splat1],color='xkcd:hot pink',transform=transform_used)
a3.plot([splon1,splon2],[splat2,splat2],color='xkcd:hot pink',transform=transform_used)
a3.coastlines(color='gray')
plt.text(0.02,0.95,'d)',fontsize=12,transform=a3.transAxes)

a4=plt.subplot(4,2,5,projection=projection_used)
a4.contourf(lon,lat,ohctransmeanposneg,ohccontours,cmap=ohccmap,transform=transform_used,extend='both')
a4.coastlines(color='gray')
plt.text(-0.07,0.1,r'OHC at transition',fontsize=14,rotation='vertical',transform=a4.transAxes)
plt.text(0.02,0.95,'e)',fontsize=12,transform=a4.transAxes)

a5=plt.subplot(4,2,6,projection=projection_used)
a5.contourf(lon,lat,ohctransmeannegpos,ohccontours,cmap=ohccmap,transform=transform_used,extend='both')
a5.coastlines(color='gray')
plt.text(0.02,0.95,'f)',fontsize=12,transform=a5.transAxes)

a6=plt.subplot(4,2,7,projection=projection_used)
a6.contourf(lon,lat,ohc0meanposneg,ohccontours,cmap=ohccmap,transform=transform_used,extend='both')
a6.coastlines(color='gray')
plt.text(-0.07,0.2,r'OHC at $\tau$=0',fontsize=14,rotation='vertical',transform=a6.transAxes)
plt.text(0.02,0.95,'g)',fontsize=12,transform=a6.transAxes)

a7=plt.subplot(4,2,8,projection=projection_used)
a7.contourf(lon,lat,ohc0meannegpos,ohccontours,cmap=ohccmap,transform=transform_used,extend='both')
a7.coastlines(color='gray')
plt.text(0.02,0.95,'h)',fontsize=12,transform=a7.transAxes)

clrpax = plt.axes((0.93,0.75,0.02,0.21))
cbarlrp = plt.colorbar(clrp,cax=clrpax,ticks=np.arange(1))
cbarlrp.ax.set_ylabel('relevance',fontsize=12)

cohcax = plt.axes((0.93,0.25,0.02,0.25))
cbarohc = plt.colorbar(cohc,cax=cohcax,ticks=np.arange(-2,3))
cbarohc.ax.set_ylabel((r'OHC anomaly ($\sigma$)'))

plt.tight_layout()

# plt.savefig('../paperfigs/LRP_transition%d-%d_sigcontours.png' %(month1,month2),dpi=300)
plt.show()



