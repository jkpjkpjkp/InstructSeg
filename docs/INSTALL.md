# Installation
Set up conda environment and install required packages.

```bash
conda create -n InstructSeg python=3.10.13
conda activate InstructSeg
conda install pytorch torchvision torchaudio pytorch-cuda={your-cuda-version} -c pytorch -c conda-forge -y
pip install -r requirements.txt
conda install opencv
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```