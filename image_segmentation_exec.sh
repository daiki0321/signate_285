#!/bin/bash

input_height=1216
input_width=1920

model_name=$1

checkpoints_path="checkpoints/${model_name}/"
output_path="output/${model_name}"
output_tmp_path="output/${model_name}_tmp"

if [ ${model_name}=="resnet50_segnet" ]; then
    input_height=608
    input_width=960
fi

mkdir -p ${checkpoints_path}
rm -rf ${checkpoints_path}/*

python -m image-segmentation-keras.keras_segmentation train \
 --checkpoints_path=${checkpoints_path} \
 --train_images="seg_train_images/" \
 --train_annotations="output_train/" \
 --epochs=20 \
 --n_classes=20 \
 --input_height=${input_height} \
 --input_width=${input_width} \
 --model_name=${model_name}

mkdir -p ${output_path}
rm -rf ${output_path}/*

python -m image-segmentation-keras.keras_segmentation predict \
 --checkpoints_path=${checkpoints_path} \
 --input_path="seg_test_images" \
 --output_path=${output_path}

python -m image-segmentation-keras.keras_segmentation predict \
 --checkpoints_path=${checkpoints_path} \
 --input_path="seg_train_images" \
 --output_path=${output_tmp_path}

python seg_codes/make_submit.py -p ${output_path}
mv submit.json ${model_name}_submit.json

python seg_codes/make_submit.py -p ${output_tmp_path}
mv submit.json ${model_name}_tmp_submit.json

python IOU.py -g submit_correct.json -p ${model_name}_tmp_submit.json