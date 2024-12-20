# Installation
Set up conda environment and install required packages. 

```bash
conda create -n InstructSeg python=3.10.13
conda activate InstructSeg
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -c conda-forge -y
pip install -r requirements.txt
```