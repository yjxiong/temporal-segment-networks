#Temporal Segment Networks (TSN)

This repository holds the codes and models for the paper
 
> 
**Temporal Segment Networks: Towards Good Practices for Deep Action Recognition**,
Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool,
*ECCV 2016*, Amsterdam, Netherlands.
>
[[Arxiv Preprint](http://arxiv.org/abs/1608.00859)]

## News & Updates

Nov. 5, 2016 - The project page for TSN is online. [website][tsn_site]

Sep. 14, 2016 - We fixed a legacy [bug][bug] in Caffe. Some parameters in TSN training are affected. 
You are advised to update to the latest version.

##[FAQ][faq]

Below is the guidance to reproduce the reported results and explore more.

# Contents
* [Usage Guide](#usage-guide)
  * [Prerequisites](#prerequisites)
  * [Code & Data Preparation](#code--data-preparation)
    * [Get the code](#get-the-code)
    * [Get the videos](#get-the-videos)
    * [Get trained models](#get-trained-models)
  * [Extract Frames and Optical Flow Images](#extract-frames-and-optical-flow-images)
  * [Testing Provided Models](#testing-provided-models)
    * [Get reference models](#get-reference-models)
    * [Video-level testing](#video-level-testing)
  * [Training Temporal Segment Networks](#training-temporal-segment-networks)
    * [Construct file lists for training and validation](#construct-file-lists-for-training-and-validation)
    * [Get initialization models](#get-initialization-models)
    * [Start training](#start-training)
    * [Config the training process](#config-the-training-process)
* [Other Info](#other-info)
  * [Citation](#citation)
  * [Related Projects](#related-projects)
  * [Contact](#contact)

----
# Usage Guide

## Prerequisites
[[back to top](#temporal-segment-networks-tsn)]

There are a few dependencies to run the code. The major libraries we use are

- [Our home-brewed Caffe][caffe]
- [dense_flow][df]

The codebase is written in Python. We recommend the [Anaconda][anaconda] Python distribution. Matlab scripts are provided for some critical steps like video-level testing.

The most straightforward method to install these libraries is to run the `build-all.sh` script.

Besides software, GPU(s) are required for optical flow extraction and model training. Our Caffe modification supports highly efficient parallel training. Just throw in as many GPUs as you like and enjoy.

## Code & Data Preparation

### Get the code
[[back to top](#temporal-segment-networks-tsn)]

Use git to clone this repository and its submodules
```
git clone --recursive https://github.com/yjxiong/temporal-segment-networks
```

Then run the building scripts to build the libraries.

```
bash build_all.sh
```
It will build Caffe and dense_flow. Since we need OpenCV to have Video IO, which is absent in most default installations, it will also download and build a local installation of OpenCV and use its Python interfaces.

Note that to run training with multiple GPUs, one needs to enable MPI support of Caffe. To do this, run

```
MPI_PREFIX=<root path to openmpi installation> bash build_all.sh MPI_ON
```

### Get the videos
[[back to top](#temporal-segment-networks-tsn)]

We experimented on two mainstream action recognition datasets: [UCF-101][ucf101] and [HMDB51][hmdb51]. Videos can be downloaded directly from their websites.
After download, please extract the videos from the `rar` archives.
- UCF101: the ucf101 videos are archived in the downloaded file. Please use `unrar x UCF101.rar` to extract the videos.
- HMDB51: the HMDB51 video archive has two-level of packaging.
The following commands illustrate how to extract the videos.
```
mkdir rars && mkdir videos
unrar x hmdb51-org.rar rars/
for a in $(ls rars); do unrar x "rars/${a}" videos/; done;
```

### Get trained models
[[back to top](#temporal-segment-networks-tsn)]

We provided the trained model weights in Caffe style, consisting of specifications in Protobuf messages, and model weights.
In the codebase we provide the model spec for UCF101 and HMDB51.
The model weights can be downloaded by running the script

```
bash scripts/get_reference_models.sh
```

## Extract Frames and Optical Flow Images
[[back to top](#temporal-segment-networks-tsn)]

To run the training and testing, we need to decompose the video into frames. Also the temporal stream networks need optical flow or warped optical flow images for input.
 
These can be achieved with the script `scripts/extract_optical_flow.sh`. The script has three arguments
- `SRC_FOLDER` points to the folder where you put the video dataset
- `OUT_FOLDER` points to the root folder where the extracted frames and optical images will be put in
- `NUM_WORKER` specifies the number of GPU to use in parallel for flow extraction, must be larger than 1

The command for running optical flow extraction is as follows

```
bash scripts/extract_optical_flow.sh SRC_FOLDER OUT_FOLDER NUM_WORKER
```

It will take from several hours to several days to extract optical flows for the whole datasets, depending on the number of GPUs.  

## Testing Provided Models

### Get reference models
[[back to top](#temporal-segment-networks-tsn)]

To help reproduce the results reported in the paper, we provide reference models trained by us for instant testing. Please use the following command to get the reference models.

```
bash scripts/get_reference_models.sh
```

### Video-level testing
[[back to top](#temporal-segment-networks-tsn)]

We provide a Python framework to run the testing. For the benchmark datasets, we will measure average accuracy on the testing splits. We also provide the facility to analyze a single video.

Generally, to test on the benchmark dataset, we can use the scripts `eval_net.py` and `eval_scores.py`.

For example, to test the reference rgb stream model on split 1 of ucf 101 with 4 GPUs, run
```
python tools/eval_net.py ucf101 1 rgb FRAME_PATH \
 models/ucf101/tsn_bn_inception_rgb_deploy.prototxt models/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel \
 --num_worker 4 --save_scores SCORE_FILE
```
where `FRAME_PATH` is the path you extracted the frames of UCF-101 to and `SCORE_FILE` is the filename to store the extracted scores.

One can also use cached score files to evaluate the performance. To do this, issue the following command

```
python tools/eval_scores.py SCORE_FILE
```

The more important function of `eval_scores.py` is to do modality fusion.
For example, once we got the scores of rgb stream in `RGB_SCORE_FILE` and flow stream in `FLOW_SCORE_FILE`.
The fusion result with weights of `1:1.5` can be achieved with

```
python tools/eval_scores.py RGB_SCORE_FILE FLOW_SCORE_FILE --score_weights 1 1.5
```

To view the full help message of these scripts, run `python eval_net.py -h` or `python eval_scores.py -h`. 

## Training Temporal Segment Networks
[[back to top](#temporal-segment-networks-tsn)]

Training TSN is straightforward. We have provided the necessary model specs, solver configs, and initialization models.
To achieve optimal training speed,
we strongly advise you to turn on the parallel training support in the Caffe toolbox using following build command
```
MPI_PREFIX=<root path to openmpi installation> bash build_all.sh MPI_ON
```

where `root path to openmpi installation` points to the installation of the OpenMPI, for example `/usr/local/openmpi/`.

### Construct file lists for training and validation
[[back to top](#temporal-segment-networks-tsn)]

The data feeding in training relies on `VideoDataLayer` in Caffe.
This layer uses a list file to specify its data sources.
Each line of the list file will contain a tuple of extracted video frame path, video frame number, and video groundtruth class.
A list file looks like
```
video_frame_path 100 10
video_2_frame_path 150 31
...
```
To build the file lists for all 3 splits of the two benchmark dataset, we have provided a script.
Just use the following command
```
bash scripts/build_file_list.sh ucf101 FRAME_PATH
```
and
```
bash scripts/build_file_list.sh hmdb51 FRAME_PATH
```
The generated list files will be put in `data/` with names like `ucf101_flow_val_split_2.txt`.

### Get initialization models
[[back to top](#temporal-segment-networks-tsn)]

We have built the initialization model weights for both rgb and flow input.
The flow initialization models implements the cross-modality training technique in the paper.
To download the model weights, run
```
bash scripts/get_init_models.sh
```

### Start training
[[back to top](#temporal-segment-networks-tsn)]

Once all necessities ready, we can start training TSN.
For this, use the script `scripts/train_tsn.sh`.
For example, the following command runs training on UCF101 with rgb input
```
bash scripts/train_tsn.sh ucf101 rgb
```
the training will run with default settings on 4 GPUs.
Usually, it takes around 1 hours to train the rgb model and 4 hours for flow models, on 4 GTX Titan X GPUs.

The learned model weights will be saved in `models/`.
The aforementioned testing process can be used to evaluate them.

### Config the training process
[[back to top](#temporal-segment-networks-tsn)]

Here we provide some information on customizing the training process
- **Change split**: By default, the training is conducted on split 1 of the datasets.
To change the split, one can modify corresponding model specs and solver files.
For example, to train on split 2 of UCF101 with rgb input, one can modify the file `models/ucf101/tsn_bn_inception_rgb_train_val.prototxt`.
On line 8, change
```
source: "data/ucf101_rgb_train_split_1.txt"`
```
to
```
`source: "data/ucf101_rgb_train_split_2.txt"`
```
On line 34, change
```
source: "data/ucf101_rgb_val_split_1.txt"
```
to
```
source: "data/ucf101_rgb_val_split_2.txt"
```
Also, in the solver file `models/ucf101/tsn_bn_inception_rgb_solver.prototxt`, on line 12 change
```
snapshot_prefix: "models/ucf101_split1_tsn_rgb_bn_inception"
```
to
```
snapshot_prefix: "models/ucf101_split2_tsn_rgb_bn_inception"
```
in order to distiguish the learned weights.
- **Change GPU number**, in general, one can use any number of GPU to do the training.
To use more or less GPU, one can change the `N_GPU` in `scripts/train_tsn.sh`.
**Important notice**: when the GPU number is changed, the effective batchsize is also changed.
It's better to always make sure the effective batchsize, which equals to `batch_size*iter_size*n_gpu`, to be **128**.
Here, `batch_size` is the number in the model's prototxt, for example [line 9](https://github.com/yjxiong/temporal-segment-networks/blob/master/models/ucf101/tsn_bn_inception_rgb_train_val.prototxt#L9)
in `models/ucf101/tsn_bn_inception_rgb_train_val.protoxt`.
 
#Other Info
[[back to top](#temporal-segment-networks-tsn)]

## Citation
Please cite the following paper if you feel this repository useful.
```
@inproceedings{TSN2016ECCV,
  author    = {Limin Wang and
               Yuanjun Xiong and
               Zhe Wang and
               Yu Qiao and
               Dahua Lin and
               Xiaoou Tang and
               Luc {Val Gool}},
  title     = {Temporal Segment Networks: Towards Good Practices for Deep Action Recognition},
  booktitle   = {ECCV},
  year      = {2016},
}
```

## Related Projects

- [CES-STAR@ActivityNet][anet] : we won ActivityNet challenge 2016 based on TSN
- [TDD][tdd]: Trajectory-pooled Deep Descriptors for action recognition.
- [Very Deep Two Stream CNNs][caffe]

## Contact
For any question, please contact
```
Yuanjun Xiong: yjxiong@ie.cuhk.edu.hk
Limin Wang: lmwang.nju@gmail.com
```

[ucf101]:http://crcv.ucf.edu/data/UCF101.php
[hmdb51]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[caffe]:https://github.com/yjxiong/caffe
[df]:https://github.com/yjxiong/dense_flow
[anaconda]:https://www.continuum.io/downloads
[tdd]:https://github.com/wanglimin/TDD
[anet]:https://github.com/yjxiong/anet2016-cuhk
[faq]:https://github.com/yjxiong/temporal-segment-networks/wiki/Frequently-Asked-Questions
[bs_line]:https://github.com/yjxiong/temporal-segment-networks/blob/master/models/ucf101/tsn_bn_inception_flow_train_val.prototxt#L8
[bug]:https://github.com/yjxiong/caffe/commit/c0d200ba0ed004edcfd387163395be7ea309dbc3
[tsn_site]:http://yjxiong.me/others/tsn/
