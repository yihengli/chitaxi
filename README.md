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

In addition to the default python import methods, some features such as crawlers/extractors can be enabled directly from the terminal as the CLI tool. Simply call `nlpbonds --help` in the terminal

```
Options:
  --config-reset                  Reset the config settings
  --config-data TEXT              Define the data folder
```

## How to load data

``` python
from chitaxi.datasets.loader import get_data_taxi

# Get data within a range
df = get_data_taxi(start='20140101', end='20140201')

# Get data for a specific year
df = get_data_taxi(year=2015)
```

Make sure you have the correct `hdf` format dataset in your `CONFIG-DATA` folder.