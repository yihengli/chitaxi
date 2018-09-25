# Chi-Taxi
This repo is to analyze the Chicago Taxi data and predict future tax revenue per taxi

## How to use this code

This code repository is maintained as a python package. One can simply install it by using

``` bash
pip install .
```

However, as it's still under development, it's recommended to use following command to install it

``` bash
pip install -e .
```

In the editable mode, python won't try to move the entire folder into `site-pacakge`. A soft link will be created and any ad-hoc changes in the code can be direclty reflected in your work environments.

### Jupyter / Ipython

For jupyter and ipython, you can open the reload option, then any adhoc changes will be dynamically loaded without reimport.

```
%load_ext autoreload
%autoreload 2
```

### CLI Tools

When initializing the project, `chitaxi` can bind itself with a local folder, any input and output files can be directly looked up in the given workplace.

This can be done manually by revising the `config.yaml` file, or use CLI command 

```
chitaxi --config-data PATH_TO_WORKPLACE
```

```
Options:
  --config-reset
  --config-data TEXT
  --clean-taxi TEXT
  --help              Show this message and exit.
```

## How to load data

The original data is huge, around 6 - 10GB per month and can be downloaded from here:
[https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew/data](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew/data)

Simply put all downlaoded CSV files into one clean folder, use CLI can clean up the file and aggregate them into a HDF5 format dataset and put the h5 file into workspace.

> Note:
> 1. The dataset is large, expected final output is above 20GB
> 2. If the RAM is limited, you have to split each CSV as small as possible. This can be done by using date filters on chicago website

```
chitaxi --clean-taxi PATH_TO_CSV_FOLDER
```

Alternatively, one can directly download the pre-cleaned h5 format [here](https://www.dropbox.com/s/epqjbs2pmzagmo1/chitaxi.h5?dl=0):

Under your workspace:
```
.
├── chitaxi.h5
```

Then you slice the data into any python kernel:

``` python
from chitaxi.datasets.loader import get_data_taxi

# Get data within a range
df = get_data_taxi(start='20140101', end='20140201')

# Get data for a specific year
df = get_data_taxi(year=2015)
```

Make sure you have the correct `hdf` format dataset in your `CONFIG-DATA` folder.

## How to run model

The feature and label data can be directly downloaded from [here](https://www.dropbox.com/s/b1g2xwy799hrb5u/baseline_data.zip?dl=0)
