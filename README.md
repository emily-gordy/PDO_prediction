# PDO prediction
This is the code for [Gordon et. al 2021](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021GL095392) (the one with the word "harbingers" in the title)

This contains python code written in python 3.9 to pre-process data, create the ANNs, and to analyze the ANNs using layer-wise relevance propagation (LRP).

There are specific python packages that need to be installed. Note, the package containing LRP functions only works with a different tensorflow version than that used to generate the models (also uses python 3.7).

## Packages
- [tensorflow](https://www.tensorflow.org/install/pip) 2.4
- [innvestigate](https://github.com/albermax/innvestigate) with tensorflow 1.15 and python 3.7
- [cmasher](https://cmasher.readthedocs.io/user/introduction.html#how-to-install)

## Pre-processing

#### PDO_prediction/code/Pre-Processing/save_deseasonedPDOandOHC.py

This code assumes access to CESM2 OHC and SST on 4x4 lat x lon grid. The deseason file outputs the (unsmoothed) PDO index and individual deseasoned OHC grids. 

### PDO_prediction/code/Pre-Processing/save_nninput_output.py

In savenn_input_output.py applies 6 month running mean smoothing to both OHC and PDO index, then puts the OHC maps into the required format for the ANN input (3x OHC grids, 4 months apart) so each row is one input sample. The output is the smoothed PDO index, and the OHC input grids, both as separate netCDF4 files.

## Training the ANN

#### PDO_prediction/code/PDOtransition_fromOHC_trainnn.py

This is the python file for generating and training the neural networks in this study as well as saving the weights in hdf5 format so they can be loaded in later.

## Analyzing the ANN

#### PDO_prediction/code/LRP/LRP_PDOtransition_fromOHC_persistenceplots.py

This file use the innvestigate package to analyse the ANNs using LRP. This is specifically for looking at samples where the PDO persists.

#### PDO_prediction/code/LRP/LRP_PDOtransition_fromOHCtransitionplots.py

As above but for analyzing samples where a PDO transition occurs 12-27 months from the input.


## Extras for people who scrolled this far

Let me know if you have comments on this! I am open to questions about setting up environments all the way through to making the analyzer work. I love comments*

*except if they are comments about how my code could run better. This code is at its peak as is. I will hear no criticism

