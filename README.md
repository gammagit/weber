This project contains code for creating datasets to test Weber's Law in Deep Learning models. There are two major files:

`create_weber_stim.py`: Run this script to create a dataset that can be used to test Weber's law.
    The stimulus will be created based on parameters in the get_params() function.

`test_weber.ipynb`: This notebook runs tests Weber's law on a ResNet-151. Currently, we only test whether line lengths are encoded on a log scale.

The directory `mind-set` contains code for training decoders and calculating cosine distance on internal representations of a CNN.