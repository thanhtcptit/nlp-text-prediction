#!/bin/bash

docker run --rm -it --net=host --gpus=all \
    --mount type=bind,source=/data/thanhtc3,target=/data/thanhtc3 \
    --mount type=bind,source=/data/zminer,target=/data/zminer \
    -t tf-fed:latest /bin/bash
