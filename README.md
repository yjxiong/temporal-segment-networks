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

The codebase is written in Python. We recommend the [Anaconda][anaconda] Python distribution. Matlab scripts are provided for some critical steps like video-level testing.

Besides software, GPU(s) are highly recommended for training. Our Caffe modification supports highly efficient parallel training. So throw in as many GPUs as you like and enjoy faster training.

## Code & Data Preparation

#### Get the code
Use git to clone this repository and its submodules
```
git clone --recursive https://github.com/yjxiong/temporal-segment-networks
```

Then run the building scripts to build the libraries.

```
bash build-all.sh
```
It will build Caffe and dense_flow. Since we need OpenCV to have Video IO, which is absent in most default installations, it will also download and build a local installation of OpenCV and use its Python interfaces. 

#### Prepare videos
We experimented on two mainstream action recognition dataset: [UCF-101][ucf101] and [HMDB51][hmdb51]. Videos can be downloaded directly from their websites.
After download, please extract the videos from the `rar` archives.

## Testing Provided Models

## Training Temporal Segment Networks



[ucf101]:http://crcv.ucf.edu/data/UCF101.php
[hmdb51]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[caffe]:https://github.com/yjxiong/caffe
[df]:https://github.com/yjxiong/dense_flow
[anaconda]:https://www.continuum.io/downloads

