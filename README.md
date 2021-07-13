## PDO prediction
This is the code for the PDO prediction project, [link]

This contains python code written in python 3.9 to pre-process data, create the ANNs, and to analyze the ANNs using layer-wise relevance propagation (LRP).

There are specific python packages that need to be installed, as well as in different environments need to be created to run this code. Specifically, the package containing LRP functions only works with a different tensorflow version than that used to generate the models (also uses python 3.7).

# Packages
- [tensorflow](https://www.tensorflow.org/install/pip) 2.5
- [innvestigate](https://github.com/albermax/innvestigate) with tensorflow 1.15
- [cmasher](https://cmasher.readthedocs.io/user/introduction.html#how-to-install)
