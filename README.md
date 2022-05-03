# Quickstart:

table detection in PDF and scaned documents with scattering wavelets

1) OS name/OS Type: `Ubuntu 20.04.3 LTS/64-bit`
2) Processor: AMD® Ryzen threadripper 1900x 8-core processor × 16
3) You need `zstd` for unpack data: `sudo apt install zstd`
4) `pip install -r requirements.txt`
5) `python setup.py install`
6) `python startMeUp.py`
7) if you are asked "calculate SWCs (Y/N) ?" then type "Y"

# Intention of this project

This project uses scattering wavelet networks for image processing and classification which is so far typically done with convolutional neuronal networks (CNNs). The theory is due to S. Mallat, I. Daubechies and many others which set the basic theory for wavelets. Later on Mallat started to investigate on  the very popular deep convolutional neuronal networks in order to bring light in the black box. 

A good introduction to this topic is the paper [papers/] from S. Mallat.
