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
    ├── get_unique_SanDiego.sh       # Extract unique MMSI entries from the dataset around San Diego area
├── src                     # Source code for training, evaluation, and data preprocessing for two designated tasks
    ├── clean               # Contains source code for cleaning up datasets and preprocessing them
        ├── clean_data.py       # filters invalid vessel types, and get vessels around San Diego
        ├── collectByType.py    # categorize vessels by type in different directories
        ├── createVoyages.py    # filters good voyages for LSTM
        ├── seperateMMSI.py     # get all time series data for a specific MMSI
        ├── unique_id.py        # finds all unique MMSI ID entries in the input file
        ├── unique_id_SanDiego.py        # finds all unique MMSI ID entries in the input file around San Diego area
    ├── mlp                # Contains source code for vessel classification task
        ├── AISDataset.py       # setting up classification dataset and balancing the dataset
        ├── evaluate.py         # utilities for evaluating models
        ├── models.py           # deep learning models
        ├── train.py            # training utilities
    ├── lstm                # Contains source code for trajectory prediction task
        ├── LSTMDataset.py      # setting up lstm dataset (reading into memory not all at once)
        ├── LSTMDataset2.py     # setting up lstm dataset (reading into memory all at once)
        ├── evaluate.py         # utilities for evaluating models
        ├── models.py           # deep learning models
        ├── train.py            # training utilities
└── ...
```

## Reproducing Results

#### Downloading the Dataset:
1. Configure data download directory in "collect_ais.sh" bash script under scripts/ folder 
2. Running collect_ais.sh will download all years between those defined in the "years" variable in file

#### MLP
1. Data Manipulations
    *  configure the paths in "get_unique.sh" to direct where the raw data set exists and where the fixed data needs to go.
    *  this creates an output file titled "uniqueMMSI_withDraft.csv" and places it in the clean_dir directory defined in the file.
    *  bash script calls "unique_id.py" that lives under src_dir directory defined in the file
    *  unique_id.py currently lives under the src/clean/ folder
2. MLP File Explanations
    *  AISDataset.py
        * creates a pandas dataframe from unique_id file
        * Sets a "num_class" variable to use when creating the model
        * currently limits all classes to be limited between 1000-2000 samples
        * expects "Length","Width","Draft", and "VesselType" to be valid entries in the data frame
    *  models.py
        *  define the number of layers, hidden neurons, activations, and normalizations in this file
        *  We started with TwoLayerReLU, but added a ThreeLayerSigmoid for more testing
    *  train.py
        *  configure csv file and directory for file
    *  evaluate.py
        *  calculates accuracy by counting how many times correct label was chosen
3. Running MLP
    * Run "train.py" once you have configured all files

#### LSTM
1.  Data Manipulations
    * configure the paths in "clean_data.sh" to direct where the raw data set exists and where the fixed data needs to go.
        * creates one cleaned output file per input file and outputs to defined directory
        * bash script calls "clean_data.py" that lives under src_dir
            * clean_data.py currently lives under the src/clean/ folder
    * unzip all 2021 data into a new folder
    * run get_unique_SanDiego.sh under scripts/ folder
        *  configure paths to unzipped data directory and python sorce directory
        *  calls "unique_id_SanDiego.py" for each cleaned file
        *  creates a "uniqueMMSI_SanDiego.csv" file as output in data_dir
    * run separateMMSI.py under src/clean folder with arguments for input and output directory
        * separates all unique MMSI IDs in 2021 into their own files to prep for LSTM labelling
    * run collectByType.py under src/clean with arguments for input directory, type number (we used 37), and output directory
        * this only brings a specific VesselType of data separated into files by MMSI into a new directory
    * run createVoyages.py under src/clean with arguments for input and output directories
        * for each MMSI through the year, split into multiple voyages and number the voyages.
        * this finalizes the requiremtns for labelling the data for our LSTM
2. LSTM File Explanations
    * LSTMDataset.py
        * Creates an index look-up for file, vessel, and voyage id
        * uses first 20 minutes of voyage as input, and the 25th minute [LAT, LON] as label
        * uses internal index to only load csv file when provided an index from outside
        * Is fast to load data, but slow when training due to the number of reads and searches
     * LSTMDataset2.py
        * Loads all data into a single dataframe 
        * Is slow at start up for large datasets, but faster during training
        * Is used because, for our dataset, we only take the first 20 min as input and 25th minute as label per voyage instead of using data from the entire voyage. Other approaches may be necessary when using all data. 
     * models.py
        * Defines the LSTM with user-specified hidden layers and hidden dimensions.
        * Passes last hidden layer throgh Fully-Connected layer to provide final [LAT,LON] prediction
     * evaluate.py
        * calculates the MSELoss between predictions and labels
        * better approach is to have a euclidian distance error calculation between the [LAT,LON] points. This requires a transformation from [LAT,LON] to [X,Y,Z] euclidian space
      * train.py
        * configure the input directory for the specific type to train on
        * trains a model for only a specific VesselType

3. Running LSTM
    * Run "train.py" once you have changed the file_dir variable  
