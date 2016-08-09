#!/usr/bin/env bash

# ucf101 rgb models, 3 splits
wget -O models/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel.v5
wget -O models/ucf101_split_2_tsn_rgb_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/ucf101_split_2_tsn_rgb_reference_bn_inception.caffemodel.v5
wget -O models/ucf101_split_3_tsn_rgb_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/ucf101_split_3_tsn_rgb_reference_bn_inception.caffemodel.v5

# ucf101 flow models, 3 splits
wget -O models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel.v5
wget -O models/ucf101_split_2_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/ucf101_split_2_tsn_flow_reference_bn_inception.caffemodel.v5
wget -O models/ucf101_split_3_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/ucf101_split_3_tsn_flow_reference_bn_inception.caffemodel.v5

# hmdb51 rgb models, 3 splits
wget -O models/hmdb51_split_1_tsn_rgb_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/hmdb51_split_1_tsn_rgb_reference_bn_inception.caffemodel.v5
wget -O models/hmdb51_split_2_tsn_rgb_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/hmdb51_split_2_tsn_rgb_reference_bn_inception.caffemodel.v5
wget -O models/hmdb51_split_3_tsn_rgb_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/hmdb51_split_3_tsn_rgb_reference_bn_inception.caffemodel.v5

# hmdb51 flow models, 3 splits
wget -O models/hmdb51_split_1_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/hmdb51_split_1_tsn_flow_reference_bn_inception.caffemodel.v5
wget -O models/hmdb51_split_2_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/hmdb51_split_2_tsn_flow_reference_bn_inception.caffemodel.v5
wget -O models/hmdb51_split_3_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/hmdb51_split_3_tsn_flow_reference_bn_inception.caffemodel.v5
