# Multi-Scale Geographic Location Encoding on the Sphere using Slepian Functions
Spherical Slepian–based positional encoding for multi-scale geographic location encoding, including hybrid encoders with spherical harmonics and experiments on global and local geospatial tasks.
## Usage
```
# Create virtual environment
python -m venv venv
# Activate it (Windows)
.\venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```
Python 3.11.9 or higher is required.
## Experiments
### Land-ocean classification
This experiment assesses the encoder's ability to learn representations across scales. This is reproduced from Rußwurm et al., arXiv:2310.06743.
The data is automatically generated in `src/data/landoceandataset.py`.
To run the Slepian-based encoder with Slepian bandwidth L = 40 and create plots:
```
python train.py --pe slepian \
  --matplotlib \
  --legendre-polys 40 \
  --num-samples 100000
```
### High-resolution land-ocean classification
This version of land-ocean classification tests the location encoder's ability to resolve high resolution details of coastlines and islands.
The data is automatically generated in `src/data/high_res_landoceandataset.py`.
To run all experiments:
```
python run_experiments.py
```
To run specific configurations separately and create plots:
```
python train.py --dataset highreslandoceandataset \
  --pe slepian \
  --nn mlp \
  --legendre-polys 30 \
  --num-samples 100000 \
  --matplotlib
```
### Arctic MSS reconstruction
This experiment compares the Slepian mask and Slepian cap implementations through sea surface height regression.
The preprocessed data from Chen et al., arXiv:2412.11350, is available through [Google Drive](https://drive.google.com/drive/folders/17rwMtEc5vwRKEjNolreUBL2Yk4OSTvr4?usp=sharing).
To run the experiments from the `mss/` directory:
```
python run_mss_experiments.py
python run_mss_experiments_cap.py
```
