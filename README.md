# Intention of this project

This project uses scattering wavelet networks for image processing and classification which is so far typically done with convolutional neuronal networks (CNNs) or related models. This technique is used in order to detect tables and extract their content from scaned documents. 

## Calibration or Training

For given jpg-file `img.jpg` which should use for the training the algorithm makes the following steps (this discription is very crude description, details will follow later in this readme file):

- generate from `img.jpg` a new jpg `img_bb.jpg` which consist only of black filled bounding boxes and remove eventually lines
- annotate `img_bb.jpg` with bounding boxes which contain eventually tables 
- for a given window size cut this jpg in horizontally stripes `H` and vertically stripes `V` 
- by a given rule annotate this stripes to `0`: contains no table and `1`: contains table
- use now the scattering wavelet algorithm in order to calculate scattering wavelets coefficients `SWC_H` and `SWC_V` for all stripes
- generate a random forest `R_H` for the horizontally and `R_V` vertically stripes using the scattering wavelet coefficients `SWC_H` and `SWC_V`

The random forest `R_H` and `R_V` are later used for the prediction (detection) of tables.

## Detection of tables

For given jpg-file `img.jpg` for which we want detect tables the algorithm makes the following steps (this discription is very crude description, details will follow later in this readme file):

- generate from `img.jpg` a new jpg `img_bb.jpg` which consist only of black filled bounding boxes and remove eventually lines
- for the same given window size as used in the random forest `R_H` and `R_V` cut `img_bb.jpg` in horizontally stripes `H` and vertically stripes `V` 
- calculate for all stripes in `H` and `V` scattering wavelet coefficients `SWC_H` and `SWC_V`
- use the calculated scattering wavelet coefficiens `SWC_C` and `SWC_V` in order to detect tables 

# Quickstart:

Table detection in PDF and scaned documents with scattering wavelets

1) OS name/OS Type: `Ubuntu 20.04.3 LTS/64-bit`
2) Processor: AMD® Ryzen threadripper 1900x 8-core processor × 16
3) You need `zstd` for unpack data: `sudo apt install zstd`
4) `pip install -r requirements.txt`
5) `python setup.py install`
6) `python startMeUp.py`
7) if you are asked "calculate SWCs (Y/N) ?" then type "Y"

# Introduction

This project uses scattering wavelet networks for image processing and classification which is so far typically done with convolutional neuronal networks (CNNs). The theory is due to S. Mallat, I. Daubechies and many others which set the basic theory for wavelets. Later on Mallat started to investigate on  the very popular deep convolutional neuronal networks in order to bring light in the black box. 

A good introduction to this topic is the paper 

- [Understanding Deep Convolutional Networks](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/understandingDeepConvolutionalNetworks.pdf) 

from S. Mallat. The basics for the theory of wavelets, multiresolution signal decompositions with wavelets and scattering convolutional networks can be found in 

- [A Theory for Multiresolution Signal Decomposition: The Wavelet Representation](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/multiresolutionApproximationsAndWaveletsOrthonormalBasesOfL2R.pdf),
- [Group Invariant Scattering](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/groupInvariantScattering.pdf)
- [Invariant Scattering Convolutional Networks](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/invariantScatteringConvolutionNetworks.pdf) (Together with Joan Bruna)
- [Classification with Wavelet Operators](https://github.com/AlgoBitConsulting/scatteringWaveletsNetwork/blob/PDF-Table-Extractor/papers/classificationWithWaveletOperators.pdf) (Together with Joan Bruna)

The Algorithm presented here uses Morlet Wavelets because in the frequency area they are just two dimensional multivariate normal distributions. By scaling and rotation the wavelets in the frequency area we get a covering of the frequency area which is in combination with the absolute value function and several repititions nothing else than the conventional convolutional neuronal network operation. Of course this is not easy to see and we will check the single steps needed in order to understand the big picture.

# 


