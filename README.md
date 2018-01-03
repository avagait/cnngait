# CNN Gait Library for Gait-based Biometrics

Francisco M. Castro and Manuel J. Marin-Jimenez

This library contains support Matlab code for [1] and [2].
If you find useful this code, please, cite [1] or [2].

### Prerequisites
1. MatConvNet library: http://www.vlfeat.org/matconvnet/
1. Tukey's loss function (for regression): https://github.com/bazilas/matconvnet-deepReg

This code has been tested on Ubuntu 14.04 and Matlab 2013b.

### Pretrained models
Download from here: [ICIP'2017](https://www.dropbox.com/sh/wsywgfykmb6pdqw/AAA_5xdywUmBRuMNGLR1WA_oa?dl=0)

### Sample test data
Some test sequences of the test subject partition from TUM-GAID, normal scenario: [download](https://www.dropbox.com/s/2zaf2xafqqz0gzp/matimdbtum_gaid_N155-n-05_06-of25_60x60_lite.mat?dl=0)

### Quick start
Let's assume that you have placed _CNNGait_ library in folder `<cgdir>`. 
Start Matlab and type the following commands:

```
cd <cgdir>
startup_cnngait
cg_demo_test
```

### References
[1] [Deep Multi-Task Learning for Gait-based Biometrics](https://www.researchgate.net/publication/319650919_DEEP_MULTI-TASK_LEARNING_FOR_GAIT-BASED_BIOMETRICS)  
MJ Marin-Jimenez, F Castro, N Guil, F de la Torre, R Medina-Carnicer   
International Conference on Image Processing (ICIP), 2017   

[2] [Automatic learning of gait signatures for people identification](https://www.researchgate.net/publication/301841586_Automatic_learning_of_gait_signatures_for_people_identification)   
FM Castro, MJ Marín-Jiménez, N Guil, NP de la Blanca   
International Work-Conference on Artificial Neural Networks (IWANN), 257-270   

