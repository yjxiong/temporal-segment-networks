#!/usr/bin/env bash

DATASET=$1
MODALITY=$2

TOOLS=lib/caffe-action/build/install/bin
LOG_FILE=activity_net/log/${DATASET}_${MODALITY}_split1.log
N_GPU=4


echo "logging to ${LOG_FILE}"

mpirun -np $N_GPU \
$TOOLS/caffe train --solver=models/${DATASET}/tsn_bn_inception_${MODALITY}_solver.prototxt  \
   --weights=models/bn_inception_${MODALITY}_init.caffemodel 2>&1 | tee ${LOG_FILE}
