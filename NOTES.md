# Notes

For TensorFlow: Even with updated NVIDIA GPU Driver, CUDA Toolkit, and cuDNN SDK (and adding all relevant locations to system path), when running in a conda environment I still needed to `conda install cudnn`

For Pytorch: Need to make sure that you have the CUDA version of pytorch:
`pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`