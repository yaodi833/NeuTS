# NeuTS
This is an extension of NeuTraj(https://github.com/yaodi833/NeuTraj), which accelerates the similarity computation by neural metric learning.
The code has been tested successfully with Python 3.7.

## Require Packages
Pytorch, Numpy

## Running Procedures

### Create Folders
Please create 3 empty folders:

*`data`: Path of the original data which is organized to a time series. 

*`features`: This folder contains the features that generated after the preprocessing.py. It contains four files: `DATASETNAME_ts_value`, `DATASETNAME_ts_grid`, `DATASETNAME_ts_index` and `DATASETNAME_ts_label` and `seed_distance`. 

*`model`: It is used for placing the NeuTS model of each training epoch.

### Download Data
Due to the file limit of Github, we put the dataset on other sites. Please first download the data and put it in `data` folder. Three time series datasets and the learned best models can be download at: 

*`data`: It contains original time series dataset.  https://pan.baidu.com/s/1TxJ2StuCe5gfTtn_ELmOSg Extraction code: hiwq

*`features`: This folder contains the processed features and precomputed seed distances for three time series datasets. https://pan.baidu.com/s/16tmRwEsz4WSLBTG6k-_Dyg Extraction code: x2fx

*`best_models`: It contains the learned best models. https://pan.baidu.com/s/1kQzCd7oAxAII5FscTWrMLA Extraction code: 3jtc

*`UCRSuite`: Modified UCR Suite for time series classification. https://pan.baidu.com/s/1gthOreFy5FznWQqITNZL3Q Extraction code: cs7w

### Preprocessing
Run `preprocessing.py`. It filters the original data and maps the coordinates to grids. After such process, intermediate files which contain `DATASETNAME_ts_value`, `DATASETNAME_ts_grid`, `DATASETNAME_ts_index` and `DATASETNAME_ts_label` are generated. Then, we calculate the pair-wise distance under the distance measure and get the `seed_distance`.

### Training & Evaluating
Run `train.py`. It trains NeuTS under the supervision of seed distance. The parameters of NeuTS can be modified in /tools/config.py
