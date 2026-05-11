# TFABS-framework
TFABS is a deep-learning framework for identifying prophage-derived tail fiber proteins and decoding serotype-associated features in Salmonella. It combines a TFP identification module (TFPfinder) and a multimodal serotype association model (SerotypeLinker) integrating Transformer and graph neural networks.

# Overview of project 
- scripts: Contains code related to model training, graph construction, and ablation experiments
- example: Sample data including testing TFPfinder and SerotypeLinker models
- other: Contains relevant code for comparative model training, testing comparative model training, testing, and data graph drawing

# Installation
Install the environment required for training and testing the model, and run:
```bash
conda env create -f environment.yml
conda activate TFABS
```
# Test
## TFPfinder
A model for predicting tail fiber proteins from protein sequences. Prepare protein sequences in FASTA (.faa) format following the provided example, then run:
```bash
python predict_tfpfinder.py --input example/tfpfinder_example.faa
```
## SerotypeLinker
A model for predicting host serotype recognition of tail fiber proteins. Prepare a CSV file following the provided example and generate the corresponding graph representations. Then run:
```bash
python predict_serotypelinker.py --csv example/serotypelinker_example.csv --graph example/graphs
```
# Reproduce TFPfinder and SerotypeLinker from scratch
- Follow the Installation steps to configure the environment, which includes all the software packages required for training and testing the model
- Download this repository, which contains all the code needed for training and testing
- All the data required for training and testing the model has been processed and can be obtained from this Zenodo repository（

# System Requirements
- Linux (Ubuntu 20.04 recommended)
- Python 3.10
- CUDA 11.8
- NVIDIA Driver >= 520
- GPU memory >= 16GB
