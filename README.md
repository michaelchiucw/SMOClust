# Synthetic Minority Oversampling based on stream Clustering (SMOClust)

This repository contains the followings:
 - The MOA implementation of SMOClust
 - Supplementary document
 - Bash script to generate artifical data streams (in Datasets folder)
 

## Abstract
Many real-world data stream applications not only suffer from concept drift but also class imbalance. Yet, very few existing studies investigated this joint challenge. Data difficulty factors, which have been shown to be key challenges in class imbalanced data streams, are not taken into account by existing approaches when learning class imbalanced data streams. In this work, we propose a drift adaptable oversampling strategy to synthesise minority class examples based on stream clustering. The motivation is that stream clustering methods continuously update themselves to reflect the characteristics of the current underlying concept, including data difficulty factors. This nature can potentially be used to compress past information without caching data in the memory explicitly. Based on the compressed information, synthetic examples can be created within the region that recently generated new minority class examples. Experiments with artificial and real-world data streams show that the proposed approach can handle concept drift involving different minority class decomposition better than existing approaches, especially when the data stream is severely class imbalanced and presenting high proportions of safe and borderline minority class examples.

#### Author
 - Chun Wai Chiu (Michael): c dot chiu at keele dot ac dot uk
 - Leandro Minku: L dot L dot Minku at bham dot ac dot uk

#### Environment details
 - Java version: 11.0.1
 - MOA version: 2018.6.0
