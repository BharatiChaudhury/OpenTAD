#!/bin/bash

echo "===== ACTIVATE ENV ====="

module load cuda/11.8

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

export TORCH_CUDA_ARCH_LIST="8.0"

source /srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD/venv/bin/activate

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH

cd /srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD/opentad/models/utils/post_processing/nms

#pip uninstall nms-1d-cpu -y 

#find . -name "*.so" -delete
#python setup.py install 

#cd /srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD/opentad/models/roi_heads/roi_extractors/align1d

#python setup.py install 

cd /srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD


echo "Python used:"
which python
#pip install wheel setuptools ninja
#pip install triton==2.0.0
pip install mamba-ssm==1.2.0.post1 --no-build-isolation
#pip install --no-build-isolation -r requirements.txt
echo "===== JOB START ====="

date

nvidia-smi

python -c "import nms_1d_cpu; print('nms ok')"

python -c "import Align1D; print('align ok')"

python -c "from mamba_ssm import Mamba; print('mamba ok')"
#torchrun --standalone --nproc_per_node=1 tools/train.py configs/tridet/mpii_tridet.py
torchrun --standalone --nproc_per_node=1 tools/train.py configs/causaltad/mpii_groupinteraction_videomaev2.py
echo "===== DONE ====="

date
