#!/bin/bash
# Add dependency onto projects
PROJ_DIR=$HOME/WORK/projects/GAN/code
PYTHONPATH=$PYTHONPATH:$PROJ_DIR
export PYTHONPATH
python3 gan_mnist.py
