## Environment
```
conda create -y -n SSCIL python=3.9
conda activate SSCIL
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c anaconda pandas==1.5.2
pip install tqdm==4.65.0 
pip install timm==0.6.12
```
## Train
```
python main.py -i 51 -d cifar224
```

## Acknowledgment
This repo is based on aspects of https://github.com/McDonnell-Research-Lab/RanPAC
