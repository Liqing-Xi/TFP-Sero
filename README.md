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
- If you want to test your dataset with TFPfinder, please prepare protein sequences in the faa file format consistent with the example and use the following command line for prediction:
python predict_tfpfinder.py --input example/tfpfinder_example.faa
- If you want to test whether your tail fiber protein recognizes Salmonella typhimurium through Serotype linker, please prepare a CSV file like the example and build the corresponding graphs. Run the following command:
python predict_serotypelinker.py --csv example/serotypelinker_example.csv --graph example/graphs


