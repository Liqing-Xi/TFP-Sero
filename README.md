# TFABS-framework
TFABS is a deep-learning framework for identifying prophage-derived tail fiber proteins and decoding serotype-associated features in Salmonella. It combines a TFP identification module (TFPfinder) and a multimodal serotype association model (SerotypeLinker) integrating Transformer and graph neural networks.
# System Requirements
- Linux (Ubuntu 20.04 recommended)
- Python 3.10
- CUDA 11.8
- NVIDIA Driver >= 520
- GPU memory >= 16GB
# Installation
- conda env create -f environment.yml
- conda activate TFABS
# Test
## TFPfinder
A model for predicting tail fiber proteins from protein sequences.
```bash
python predict_tfpfinder.py --input example/tfpfinder_example.faa

## SerotypeLinker
A model for predicting host serotype recognition of tail fiber proteins.
```bash
python predict_serotypelinker.py --csv example/serotypelinker_example.csv --graph example/graphs
