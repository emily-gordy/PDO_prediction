# PDO prediction
This is the code for the PDO prediction project, [link]

This contains python code written in python 3.9 to pre-process data, create the ANNs, and to analyze the ANNs using layer-wise relevance propagation (LRP).

There are specific python packages that need to be installed. Note, the package containing LRP functions only works with a different tensorflow version than that used to generate the models (also uses python 3.7).

## Packages
- [tensorflow](https://www.tensorflow.org/install/pip) 2.5
- [innvestigate](https://github.com/albermax/innvestigate) with tensorflow 1.15 and python 3.7
- [cmasher](https://cmasher.readthedocs.io/user/introduction.html#how-to-install)

## Pre-processing

The code in the pre-processing folder begins with CESM2 OHC and SST on 4x4 lat x lon grid. First the deseason file outputs the (unsmoothed) PDO index and individual deseasoned OHC grids. Secondly, savenn_input_output applies 6 month running mean smoothing to both OHC and PDO index, then puts the OHC maps into the required format for the ANN input (3x OHC grids, 4 months apart) so each row is one input sample. The output is the smoothed PDO index, and the OHC input grids, both as separate netCDF4 files.

## Training the ANN

In the code directory is the python file for generating and training the neural networks in this study as well as saving the weights in hdf5 format so they can be loaded in later.

## Analyzing the ANN

These files use the innvestigate package to analyse the ANNs using LRP. There are two different files, one for looking at samples where the PDO persists, and one for analyzing samples where a PDO transition occurs 12-27 months from the input.

