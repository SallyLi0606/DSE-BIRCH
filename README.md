# DSE-BIRCH
**Dynamic and Static Enhanced BIRCH for Functional Data Clustering** \
A novel clustering method that characterizes the proximity between functional samples not only based
on static distances but also by considering dynamic discriminability based on derivative functions.

## Directory Structure
```bash
├── UCRArchive_2018/      # Dataset dir
├── fpca.R                # FPCA
├── dse_birch.py          # Python implementation for proposed DSE-BIRCH algorithm
├── clustering.py         # A wrapped Python class for clustering with DSE-BIRCH
├── utils.py              # Collection of utility functions
```

## Requirements
[![python >3.9.15](https://img.shields.io/badge/python-3.9.15-brightgreen)](https://www.python.org/)
* Build R environment \
R version `4.2.0` is required
* Build dependencies with pip
```bash
conda create -n dse_birch python==3.10.11
conda activate dse_birch
cd DSE-BIRCH
pip install -r requirements.txt
```

## Dataset
Download publicly available dataset from: `https://www.cs.ucr.edu/~eamonn/time_series_data_2018/`.

## Clustering with DSE-BIRCH
An example is demonstrated in  `example.ipynb`. \
For more details about the usage please refer to  `clustering.py`.

## Acknowledgement
The SOFTWARE will be used for teaching or not-for-profit research purposes only. Permission is required for any commercial use of the Software.
