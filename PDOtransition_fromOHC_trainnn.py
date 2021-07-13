#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:43:50 2021

@author: emilygordon
"""


#Using only ohc to predict PDO transitions, but make it tidy
# artificially balance classes (no class weights)
# NO use sample weights on 12-24
# no autocorrelation in training/validation

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import math
import random
import cmasher as cmr

writemodel = True
run=6 # using 6 month runningmean data
numonths = 3 # num months into NN
monthgap = 4 # gap between months in NN
split = 0.9 # training/testing split


# import files needed, these are NN input/output specially generated in separate script
ohcstr = "data/ohc_nninput_3xmaps_4monthsapart_run%d_samedirection.nc" %(run)
PDOstr = "data/PDO_nninput_run%d_samedirection.nc" %(run)

PDO_dataarray = xr.open_dataset(PDOstr,decode_times=False)
ohc_dataarray = xr.open_dataset(ohcstr,decode_times=False)

PDO = PDO_dataarray.PDO
ohc = ohc_dataarray.ohc

PDO = np.asarray(PDO)
ohc = np.asarray(ohc)

ohc[np.isnan(ohc)] = 0 # can't have nans in NN input. NN learns to ignore zeros

#%%make phase lengths vector, finds how long till a transition at each point
# phaselength[0] = 36 implies 36 months from index 0 till next PDO transition
 
samplesize = np.shape(PDO)[0]

PDO_now = np.copy(PDO) # PDO_now is 1 for positive phase now, or negative phase now
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

# so designed that the last phase (which doesn't end) is chopped off
phaselength = np.asarray(phaselength)
phaselength = phaselength[~np.isnan(phaselength)] 

#%%
samplescut = np.shape(phaselength)[0]
PDO = PDO[:samplescut] # chop so we are only looking at data with phase limits
ohc = ohc[:samplescut,:]

#%%
# create elarning rate scheduler too decrease learning rate throughout traing (borrowed form Ben Toms)
def step_decay(epoch):
  initial_lrate = 0.001 #The initial learning rate
  drop_ratio = 0.5 #The fraction by which the learning rate is decreased
  epochs_drop = 25.0 #Number of epochs before the learning rate is decreased
  lrate = initial_lrate * drop_ratio**(math.floor((1+epoch)/epochs_drop)) #Re-declaring the learning rate
  return lrate

lrate = keras.callbacks.LearningRateScheduler(step_decay) #This creates the learning rate scheduler using the method in the previously declared cell
callbacks_list = [lrate] 


#%% Define and train model
# designed to specifically solve this problem DO NOT POKE

def loadmodel(random_seed,ridgepen,nn_train,nn_test,PDO_train,PDO_test,leadmonth,run,maxweight):
    
    n_units = 8
    num_classes = 2
    n_epochs = 300
    
    modelstrout = 'models/PDOtransition_fromOHC_better/SingleLayer1_%dunits_seed%d_ridgepen%d_run%d_samedirection_leadmonth%d.h5' %(
        n_units, random_seed, ridgepen, run, leadmonth)#, maxweight, month1, month2)
    
    # define the model
    model = keras.models.Sequential()
    
    # First hidden layer
    model.add(layers.Dense(n_units, activation='relu',input_shape=(numonths*4050,),
                    bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                    kernel_regularizer=keras.regularizers.L2(ridgepen)))

    #dropout layer during training
    model.add(layers.Dropout(rate=0.125,seed=random_seed))    
    
    # final layer
    model.add(layers.Dense(num_classes,activation='softmax',
                    bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),))
    
    
    model.compile(optimizer=keras.optimizers.Adam(0.001),  # optimizer
                loss='categorical_crossentropy',   # loss function   
                metrics=[keras.metrics.categorical_accuracy]) 

    history = model.fit(nn_train, PDO_train, epochs=n_epochs, batch_size=128, validation_data=(nn_test, PDO_test), 
                    shuffle=True, verbose=0,callbacks=callbacks_list)
    
    return model, modelstrout, history

#%% Train a bunch of models
random_seeds = np.arange(95,96) # as many as needed

ridgepen = 12

month1 = 12
month2 = 24


for seed in random_seeds:
    
    print(seed)
    
    jj = 30 # gap between input/output
    kk=1 # this is an artifact that I don't want to delete, and honestly it will probably be useful in future
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    PDO_pred = 1*(phaselength<=jj) # PDO transitions within leadmonth
    inzone = kk*((phaselength>=month1) & (phaselength<=month2)) # mark samples in **transitino zone**
    inzone[inzone==0] = 1
    
    PDOindex = np.copy(PDO_pred) #useful to have later
    
    allsampsize = np.shape(PDO_pred)[0] # size all data
    splitind = int(split*allsampsize) # index of separation between training/validation
    
    # put 'em all together
    together = np.concatenate((ohc,np.expand_dims(PDO_pred,axis=1),np.expand_dims(inzone,axis=1),np.expand_dims(phaselength,axis=1)),axis=1)
    togethertraingrab = together[:splitind,:] # to grab training data from
    togethervalgrab = together[splitind:,:] # to grab validation data from
    
    # from training data, grab all persistence and find that number to grab of transition
    togethertrain0 = togethertraingrab[togethertraingrab[:,-3]==0,:] # all persistence (want)
    sizetrain = np.shape(togethertrain0)[0] # number of persistence in training
    
    togethertrain1all = togethertraingrab[togethertraingrab[:,-3]==1,:] # all transition in training
    sizetrainall = np.shape(togethertrain1all)[0] # number of transitions in training
    # randomly grab transition samples equal to number of persistence samples
    togethertrain1 = togethertrain1all[np.random.randint(0,sizetrainall,sizetrain)]
    
    togethertrain = np.concatenate((togethertrain0,togethertrain1))
    
    # for ii in range(3): # for debugging, check order is correct
    #     plt.plot(togethertrain[:,-(ii+1)])
    #     plt.show()
        
    # from validation data, grab all persistence and find that number to grab of transition
    togetherval0 = togethervalgrab[togethervalgrab[:,-3]==0,:] # all persistence (want)
    sizeval = np.shape(togetherval0)[0] # number of persistence in validation
    
    togetherval1all = togethervalgrab[togethervalgrab[:,-3]==1,:]
    sizevalall = np.shape(togetherval1all)[0]
    togetherval1 = togetherval1all[np.random.randint(0,sizevalall,sizeval)]
    
    togetherval = np.concatenate((togetherval0,togetherval1))
    
    # for ii in range(3): # uncomment for debugging
    #     plt.plot(togetherval[:,-(ii+1)])
    #     plt.show()
    
    # plt.subplot(2,1,1) # check distribution of phase lengths is approximately even in training/validation
    # plt.hist(togetherval1[:,-1],np.arange(0.5,24.5,1))
    # plt.subplot(2,1,2)
    # plt.hist(togethertrain1[:,-1],np.arange(0.5,24.5,1))
    # plt.show()
    
    np.random.shuffle(togethertrain) # shuffle both training and validation
    np.random.shuffle(togetherval)
    
    nn_train = togethertrain[:,:12150] # nninput for training data
    nn_test = togetherval[:,:12150] # nninput for validation data
    
    PDO_train = togethertrain[:,-3] #nn output for training data
    PDO_test = togetherval[:,-3] # nn output for validation data
   
    PDO_train=keras.utils.to_categorical(PDO_train) # make categorical arrays   
    PDO_test=keras.utils.to_categorical(PDO_test) 
    
    # and train!
    model, modelstrout, history = loadmodel(seed,ridgepen,nn_train,nn_test,PDO_train,PDO_test,jj,run,kk)
    
    print('done training')

    # check how training progressed
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label = 'training')
    plt.plot(history.history['val_loss'], label = 'testing')
    plt.ylim((0,1))
    plt.title('loss')
    plt.xlabel('epoch')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['categorical_accuracy'],label = 'training')
    plt.plot(history.history['val_categorical_accuracy'], label = 'testing')
    plt.title('Accuracy')
    #plt.ylim(0,1)
    plt.xlabel('epoch')
    plt.legend()
   
    if writemodel:
        plt.savefig(modelstrout[:-3] + '.png',dpi=100)
    
    plt.show()
    
    if writemodel:
        model.save_weights(modelstrout)
        
    #save all training and validation data
    
    valsize = np.shape(nn_test)[0]
    trainsize = np.shape(nn_train)[0]
    timeval = np.arange(valsize)
    timetrain = np.arange(trainsize)
    space = np.arange(12153)
    
    ohcPDOweightval_dataset = xr.Dataset(
    {"ohc": (("time","space"), togetherval)},
    coords={
        "time": timeval,
        "space": space
    },
    )
    ohcPDOweightval_dataset.to_netcdf(modelstrout[:-3]+'_validation.nc')
    
    ohcPDOweighttrain_dataset = xr.Dataset(
    {"ohc": (("time","space"), togethertrain)},
    coords={
        "time": timetrain,
        "space": space
    },
    )
    ohcPDOweighttrain_dataset.to_netcdf(modelstrout[:-3]+'_training.nc')
    
    # check predicion outputs (3 different ways!)
    PDOguess = model.predict(nn_test)
    
    argPDOpred = np.argmax(PDO_test,axis=1) #1 for positive, 0 for negative truth
    argPDOguess = np.argmax(PDOguess,axis=1) #1 for positive, 0 for negative NN guess
    
    modelcorr = (argPDOpred==argPDOguess) # boolean, True where model is correct
    modelconf = np.max(PDOguess,axis=1) #absolute model confidence
    
    numtest = np.shape(nn_test)[0] # size of validation set
    binwidth = 0.01 # width of bin for confidence
    bins = np.arange(0.5,1,binwidth) #make bins
    
    numsamples = np.zeros(int(0.5/binwidth)) # to be filled
    modelacc = np.zeros(int(0.5/binwidth))
    
    for ii, jj in enumerate(bins):
    
        inbin = modelconf[(modelconf>=jj)] # find all confidence at or above bin threshold
        trueinbin = modelconf[(modelconf>=jj) & modelcorr] # find all in bin that are correct
        numinbin = np.shape(inbin)[0] # find the number of each
        numtrueinbin = np.shape(trueinbin)[0]
        
        if numinbin!=0: # stop there from being zero division when bin is empty
            numsamples[ii] = 100*(numinbin/numtest)
            modelacc[ii] = 100*(numtrueinbin/numinbin)
    
    bins[numsamples==0] = np.nan # don't plot bins that are empty
    modelacc[numsamples==0] = np.nan
    numsamples[numsamples==0] = np.nan
    
    xmax = np.nanmax(bins)
    
    #plot the bins
    plt.plot(bins,numsamples,label='% of total samples at or above confidence')
    plt.plot(bins,modelacc,label='% accurate at or above confidence')
    plt.title('Model Accuracy vs Num samples')
    plt.ylabel('%')
    plt.xlabel('model confidence')
    plt.legend(loc='upper left')
    plt.xlim((0.5,xmax))
    # plt.savefig(modelstrout[:-3]+"accuracyplot.png",dpi=100)
    plt.show()
    
    plt.hist(PDOguess[:,1],bins=np.arange(0,1,0.05))
    plt.xlabel('model confidence in transition occurrence')
    plt.title('histogram of model confidence spread')
    # plt.savefig(modelstrout[:-3]+"histogram.png",dpi=100)
    plt.show()

    print(modelacc[0])

    # make confusion matrix
    confusionmat = np.empty((2,2))
    
    for uu in range(2):
        for vv in range(2): #find percentage in each bin
            boo = (argPDOpred==uu) & (argPDOguess==vv) 
            confusionmat[uu,vv] = np.shape(argPDOguess[boo])[0]
    
    confusionmat = 100*confusionmat*2/numtest
    confusionmatplot = np.flipud(confusionmat) # gotta flip when using imshow
    confusionmatplot=np.around(confusionmatplot,decimals=0)
    
    axes = np.arange(0.5,2.5,1)
    
    ax1 = plt.subplot(1,1,1)
    ax1.imshow(confusionmatplot,cmap=cmr.bubblegum)
    ax1.set_xticks(np.arange(2))
    ax1.set_yticks(np.arange(2))
    ax1.set_xticklabels(['no transition', 'transition'])
    ax1.set_yticklabels(['transition', 'no transition'])
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    for i in range(len(axes)):
        for j in range(len(axes)):
            text = ax1.text(j, i, '%d%%' %(confusionmatplot[i, j]),
                           ha="center", va="center", color="xkcd:neon blue",fontsize=16)
    plt.show()

