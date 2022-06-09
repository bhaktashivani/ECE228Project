# Vessel Classification and Trajectory Prediction

This repository contains the base code for the ECE 228 project by Matthew Aguilar, Shivani Bhakta, and Parsa Assadi. You can find out more about project structure and reproducibility of results below.

## Requirements  

The project source code for training deep learning models uses Python. Shell scripts are used for some data clean up purposes. You can use the following command to install all Python requirements:

```
pip install requirements.txt
```

Python Packages |
------------- |
Numpy |
Pandas  |
Matplotlib|
PyTorch|
Jupyter|

## Project Structure
```
.
├── docs                    # Documentation files about datasets
│   ├── data-dictionary.pdf # Metadata explaining columns of datasets, some statistics about the dataset, etc
│   └── ...                 # etc
├── notebooks               # Jupyter notebooks for generating plots for using in milestone report and final report
    └── ... 
├── scripts                 # Shell scripts for cleaning and preprocessing data using python scripts under src/clean
    ├── clean_data.sh       # Use src/clean/clean_data.py to filter invalid vessel types, and get vessels around San Diego
    ├── collect_ais.sh      # Downloads a portion of the dataset between an specific start date and end date from the dataset source
    ├── get_unique.sh       # Extract unique MMSI entries from the dataset. We need them for the classification task
└── ...
```

## Reproducing Results

- step 1 (TO DO)
- step 2 (TO DO)
- ...


