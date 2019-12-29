#!/usr/bin/env bash
sudo NV_GPU=0 nvidia-docker run \
 -it \
 --rm \
 -v "/home/hslee1/spine:/tmp/spine" \
 -v "/home/Alexandrite/hslee:/workspace" \
 --name "spinecaps" \
 --shm-size "32G" \
 hslee:pytorch-v1.3 \
 /bin/bash -c 'cd /tmp/spine &&
  python -m SpineCaps.Train --config_path=/tmp/spine/config/SpineCaps-prior.yaml'