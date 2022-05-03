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

A good introduction to this topic is the paper 

- [Understanding Deep Convolutional Networks](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/understandingDeepConvolutionalNetworks.pdf) 

from S. Mallat. The basics for the theory of wavelets, multiresolution signal decompositions with wavelets and scattering wavelets can be found in 

- [A Theory for Multiresolution Signal Decomposition: The Wavelet Representation](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/multiresolutionApproximationsAndWaveletsOrthonormalBasesOfL2R.pdf),
- [Group Invariant Scattering](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/groupInvariantScattering.pdf)
- [Invariant Scattering Convolutional Networks](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/invariantScatteringConvolutionNetworks.pdf) (Together with Joan Bruna)
- [Classification with Wavelet Operators](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/classificationWithWaveletOperators.pdf) (Together with Joan Bruna)

The Algorithm presented here works with Morlet Wavelets because in the frequency area they are just two dimensional mltivariate normal distribution. By scaling and rotation the wavelets in the frequency area we get a covering of the frequency area which is in combination with the absolute value function and several repititions nothing else than the conventional convolutional neuronal network.


