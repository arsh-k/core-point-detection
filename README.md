## <p align=center>`CP-Net: Multi-Scale Core Point Localization in Fingerprints Using Hourglass Network`</p> 

This repository contains code for CP-Net, a deep learning model that combines a U-Net based architecture MLN (Macro-Localization Network) and a CNN based architecture MRN (Micro-Regression Network) for fingerprint core-point detection. The pre-trained models (single hourglass) can be downloaded from this [link](https://drive.google.com/drive/folders/1x4F7uxXCDTe2Y6WiMkIeJQsQsVPm7ROJ?usp=share_link).

## File Description

NOTE: To use the following .py files one must add all folders in their directory. The folder names have been appropriately mentioned throughout the code files.

**train_mln.py** - Contains code necessary to train the MLN network. It also helps generate masked images for fingerprint images.

**train_mrn.py** - Contains code necessary to train the MRN network. It also helps predict core point coordinates for fingerprint images.

## Citation
