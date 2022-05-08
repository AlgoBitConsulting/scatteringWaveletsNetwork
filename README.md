# Intention of this project

This project uses scattering wavelet networks for image processing and classification which is so far typically done with convolutional neuronal networks (CNNs) or related models. This technique is used in order to detect tables and extract their content from scaned documents. 

For more information see [DocScatWaveNet](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/wiki)

# Quickstart:

Table detection in PDF and scaned documents with scattering wavelets:

1) OS name/OS Type: `Ubuntu 20.04.3 LTS/64-bit`
2) Processor: AMD® Ryzen threadripper 1900x 8-core processor × 16
3) You need `zstd` for unpack data: `sudo apt install zstd`
4) Install `git lfs` and then add information about the compressed files via `git lfs track "*.zst"` 
5) run `pip install -r requirements.txt`
6) run `python setup.py install`
7) run `python startMeUp.py`
8) if you are asked "calculate SWCs (Y/N) ?" then type "Y"

For more information see [DocScatWaveNet](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/wiki)



