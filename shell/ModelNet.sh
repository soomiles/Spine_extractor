#!/usr/bin/env bash
sudo NV_GPU=0 nvidia-docker run \
 -it \
 --rm \
 -v "/home/hslee1/spine:/tmp/spine" \
 -v "/home/Alexandrite/hslee:/workspace" \
 --name "model" \
 --shm-size "32G" \
 hslee:geometry \
 /bin/bash -c 'cd /tmp/spine &&
  python -m ModelNet.Train --config_path=/tmp/spine/config/ModelNet-train.yaml'