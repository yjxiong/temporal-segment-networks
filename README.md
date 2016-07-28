Temporal Segment Networks (TSN)
-------------------------------

This repository holds the codes and models for the paper
 
**Temporal Segment Networks: Towards Good Practices for Deep Action Recognition** 
Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool,
ECCV 2016, Amsterdam, Netherland.

It can be used for general video-based action recognition tasks. 

Below is the step-by-step guide to reproduce the reported results.


## Prerequisites

There are a few dependencies to run the code. The major libraries we use are

- [Our home-brewed Caffe][caffe]
- [dense_flow][df]

The codebase is written in Python. We recommend the [Anaconda][anaconda] Python distribution. We also provide Matlab scripts for some critical steps like video-level testing.

Besides software, GPU(s) is highly recommended to run the training. Our Caffe modification supports highly efficient parallel training. So throw in as many GPUs as you like and enjoy faster training.

## Code & Data Preparation

#### Get the code
Use git to clone this repository and its submodules
```
git clone --recursive https://github.com/yjxiong/temporal-segment-networks
```

#### Prepare videos
We experimented on two mainstream action recognition dataset: [UCF-101][ucf101] and [HMDB51][hmdb51]. 
Videos can be downloaded directly from their websites. 

## Testing Provided Models

## Training Temporal Segment Networks