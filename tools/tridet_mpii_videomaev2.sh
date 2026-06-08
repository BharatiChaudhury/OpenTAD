#!/bin/bash

echo "===== ACTIVATE ENV ====="

module load cuda/11.8

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

source /srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD/venv/bin/activate

cd /srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD

echo "Python used:"
which python
pip install -r requirements.txt
echo "===== JOB START ====="

date

nvidia-smi

python tools/train.py configs/tridet/mpii_tridet.py

echo "===== DONE ====="

date
