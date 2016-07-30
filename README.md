#Temporal Segment Networks (TSN)
-------------------------------

This repository holds the codes and models for the paper
 
**Temporal Segment Networks: Towards Good Practices for Deep Action Recognition**,
Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool,
*ECCV 2016*, Amsterdam, Netherland.

It can be used for general video-based action recognition tasks. 

Below is the step-by-step guide to reproduce the reported results.

# Usage Guide

## Prerequisites

There are a few dependencies to run the code. The major libraries we use are

- [Our home-brewed Caffe][caffe]
- [dense_flow][df]

The codebase is written in Python. We recommend the [Anaconda][anaconda] Python distribution. Matlab scripts are provided for some critical steps like video-level testing.

Besides software, GPU(s) are required for optical flow extraction and model training. Our Caffe modification supports highly efficient parallel training. So throw in as many GPUs as you like and enjoy faster training.

## Code & Data Preparation

### Get the code
Use git to clone this repository and its submodules
```
git clone --recursive https://github.com/yjxiong/temporal-segment-networks
```

Then run the building scripts to build the libraries.

```
bash build-all.sh
```
It will build Caffe and dense_flow. Since we need OpenCV to have Video IO, which is absent in most default installations, it will also download and build a local installation of OpenCV and use its Python interfaces. 

### Get the videos
We experimented on two mainstream action recognition dataset: [UCF-101][ucf101] and [HMDB51][hmdb51]. Videos can be downloaded directly from their websites.
After download, please extract the videos from the `rar` archives.
- UCF101: the ucf101 videos are archived in the downloaded file. Please use `unrar x UCF-101.rar` to extract the videos.
- HMDB51: the HMDB51 video archive has two-level of packaging. The following commands illustrate to extract the videos inside.
```
mkdir rars && mkdir videos
unrar x hmdb51-org.rar rars/
for a in $(ls rars) do; unrar x $a videos/; done;
```

## Extract Frames and Optical Flow Images

To run the training and testing, we need to decompose the video into frames. Also the temporal stream networks need optical flow or warped optical flow images for input.
 
These can be achieved with the script `scripts/extract_optical_flow.sh`. The script has three arguments
- `SRC_FOLDER` points to the folder where you put the video dataset
- `OUT_FOLDER` points to the root folder where the extracted frames and optical images will be put in
- `NUM_WORKER` specifies the number of GPU to use in parallel for flow extraction, must ber larger than 1

The command for running optical flow extraction is as follows

```
bash scripts/extraction_optical_flow.sh SRC_FOLDER OUT_FOLDER NUM_WORKER
```

It will take from several hours to several days to extract optical flows for the whole datasets, depending on the number of GPUs.  

## Testing Provided Models

### Get reference models
To help reproduce the results reported in the paper, we provide reference models trained by us for instant testing. Please use the following command to get the reference models.

```
bash scripts/get_reference_model.sh
```

### Video-level testing

We provide a Python framework to run the testing. For the benchmark datasets, we will test average accuracy on the testing splits. We also provide the facility to analyze a single video.

Generally, to test on the benchmark dataset, we can run the command

```
python tools/eval_net.py \
    --rgb_net path_to_rgb_net \
    --rgb_net_weights path_rgb_net_weights \
    --flow_net path_to_flow_net \
    --flow_net_weights path_flow_net_weights \
    --db UCF-101 --frame_path path_to_frames --split 1 
```

To cache the scores on harddisk, one can add the flag `--rgb_score path_to_rgb_score_file` to the command.

One can also use cached score files to evaluate the performance. To do this, issue the following command

```
python tools/eval_net.py \
    --rgb_score path_to_rgb_score_file \
    --flow_score path_to_flow_score_file \
    --db UCF-101 --frame_path path_to_frames --split 1 
```

The flags `--rgb(flow)_net` and `--rgb(flow)_score` can be mixed in a command. For example, one can use cached scores for a RGB model and extract scores using a flow model. 

To view the full help message of `eval_net.py`, run `python eval_net.py -h`. 

## Training Temporal Segment Networks
 
#Other Info

## Citation
Please cite the following paper if you feel this repository useful.

## Related Projects

## Contact
For any question, please contact
```
Yuanjun Xiong: xy012@ie.cuhk.edu.hk
Limin Wang: lmwang.nju@gmail.com
```

[ucf101]:http://crcv.ucf.edu/data/UCF101.php
[hmdb51]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[caffe]:https://github.com/yjxiong/caffe
[df]:https://github.com/yjxiong/dense_flow
[anaconda]:https://www.continuum.io/downloads

