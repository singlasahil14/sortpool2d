Code for the Memory Module as described
in "Learning to Remember Rare Events" by
Lukasz Kaiser, Ofir Nachum, Aurko Roy, and Samy Bengio
published as a conference paper at ICLR 2017.

Requirements:
* TensorFlow (see tensorflow.org for how to install)
* Some basic command-line utilities (git, unzip).

Description:

The general memory module is located in memory.py.
Some code is provided to see the memory module in
action on the standard Omniglot dataset.
Download and setup the dataset using data_utils.py
and then run the training script train.py
(see example commands below).

Note that the structure and parameters of the model
are optimized for the data preparation as provided.

Quick Start:

First download and set-up Omniglot data by running

```
python data_utils.py
```

Then run the training script:

```
python train.py --memory_size=8192 --pool_range=1 --save_dir=5-way/pool-1/  
python train.py --memory_size=8192 --pool_range=2 --save_dir=5-way/pool-2/ 
python train.py --memory_size=8192 --pool_range=3 --save_dir=5-way/pool-3/ 
python train.py --memory_size=8192 --pool_range=4 --save_dir=5-way/pool-4/ 
```
