#!/bin/bash

set -eux

BASE=$PROJECT_fitmulticell/yannik/study_abc_slad

# install conda environment
CONDA_DIR=$BASE/miniconda3
if [ ! -e $CONDA_DIR ]; then
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR
  rm Miniconda3-latest-Linux-x86_64.sh
fi

# create conda environment
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda create -y -n slad python=3.8

# activate environment
conda activate slad

# install redis
conda install -y redis

# install ssa
pip install ssa

# install jupyter
pip install jupyterlab

# install tumor2d
conda install -y swig=3.0
if [ ! -e $BASE/tumor2d ]; then
  git clone https://github.com/icb-dcm/tumor2d
fi
cd tumor2d
./clean.sh
python setup.py build_ext --inplace
pip install .
cd ..

# install pyabc
if [ ! -e $BASE/pyabc ]; then
  git clone https://github.com/icb-dcm/pyabc
fi
cd pyabc
git checkout feature_learn
pip install -e .
# needed on older systems
# pip uninstall -y pyarrow
cd ..

pip install -e .
# install study
#if [ ! -e $BASE/study_abc_slad ]; then
#  git clone https://github.com/yannikschaelte/study_abc_slad
#fi
#cd study_abc_slad
#pip install -e .
#cd ..

