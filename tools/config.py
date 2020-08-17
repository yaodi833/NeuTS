# Data path
# Experiment settings for ED dataset
LABEL_PATH = './features/ElectricDevices_all_ts_label'
corrdatapath = './features/ElectricDevices_all_ts_value'
gridxypath = './features/ElectricDevices_all_ts_grid'
# distancepath = './features/ElectricDevices_discret_frechet_distance_all_8100'
distancepath = './features/ElectricDevices_cdtw_distance_all_8100'
# distancepath = './features/ElectricDevices_dtw_distance_all_8100'
# distancepath = './features/ElectricDevices_erp_distance_all_8100'
# distancepath = './features/ElectricDevices_hausdorff_distance_all_8100'

load_model = './best_model/ElectricDevices_cdtw_GRU_training_8100_1966_incellTrue_config_False_0.01_20_10_0.2_ElectricDevices_False_128_test_0.4208_0.4016_0.551_3578.05230994786_9805.492781080073_6227.440471132212.h5'

# Experiment settings for IPD dataset
# LABEL_PATH = './features/ItalyPowerDemand_ts_label'
# corrdatapath = './features/ItalyPowerDemand_all_ts_value'
# gridxypath = './features/ItalyPowerDemand_all_ts_grid'
# distancepath = './features/ItalyPowerDemand_discret_frechet_distance_all_1000'
# distancepath = './features/ItalyPowerDemand_dtw_distance_all_1000'
# distancepath = './features/ItalyPowerDemand_cdtw_distance_all_1000'
# distancepath = './features/ItalyPowerDemand_erp_distance_all_1000'
# distancepath = './features/ItalyPowerDemand_hausdorff_distance_all_1000'

# load_model = './best_model/ItalyPowerDemand_erp_GRU_training_1000_1060_incellTrue_config_False_0.01_20_10_0.2_ItalyPowerDemand_False_128_test_0.7653333333333333_0.8778_1.0_6058.532666917471_6210.760460623303_152.22779370583112.h5'

# Experiment settings for UWG dataset
# LABEL_PATH = './features/UWaveGestureLibraryAll_ts_label'
# corrdatapath = './features/UWaveGestureLibraryAll_ts_value'
# gridxypath = './features/UWaveGestureLibraryAll_ts_grid'
# distancepath = './features/UWaveGestureLibraryAll_discret_frechet_distance_all_4400'
# distancepath = './features/UWaveGestureLibraryAll_dtw_distance_all_4400'
# distancepath = './features/UWaveGestureLibraryAll_cdtw_distance_all_4400'
# distancepath = './features/UWaveGestureLibraryAll_dtw_distance_all_4400'
# distancepath = './features/UWaveGestureLibraryAll_erp_distance_all_4400'
# distancepath = './features/UWaveGestureLibraryAll_hausdorff_distance_all_4400'

# load_model = './best_model/UWaveGestureLibraryAll_erp_GRU_training_4000_218_incellTrue_config_False_0.01_20_10_0.2_UWaveGestureLibraryAll_False_128_test_0.743_0.8109333333333333_0.9906666666666667_19867.086369947618_20435.017287715873_567.9309177682585.h5'




# Training Prarmeters
GPU = "1"
learning_rate = 0.01
seeds_radio=0.2
epochs = 80000
batch_size = 20
sampling_num = 10
dimensional = 1  # For ED and IPD
# dimensional = 3  # For UWG

distance_type = distancepath.split('/')[2].split('_')[1]
data_type = distancepath.split('/')[2].split('_')[0]

if distance_type == 'dtw':
    mail_pre_degree = 16
else:
    mail_pre_degree = 8

# Test Config
datalength = 8100   # For ED dataset
em_batch = 4050     # For ED dataset

# datalength = 4400 # For UWG dataset
# em_batch = 1100   # For UWG dataset

# datalength = 1000 # For IPD dataset
# em_batch = 1000   # For IPD dataset


test_num = 500      # This is validataion samples

# Model Parameters
d = 128
stard_unit = False
incell = True
recurrent_unit = 'GRU' #LSTM or SimpleRNN
spatial_width  = 2

gird_size = [1100, 1100, 1100] # For ED and IPD
# gird_size = [55,55,55] # For UWG

def config_to_str():
   configs = 'learning_rate = {} '.format(learning_rate)+ '\n'+\
             'mail_pre_degree = {} '.format(mail_pre_degree)+ '\n'+\
             'seeds_radio = {} '.format(seeds_radio) + '\n' +\
             'epochs = {} '.format(epochs)+ '\n'+\
             'datapath = {} '.format(corrdatapath) +'\n'+ \
             'datatype = {} '.format(data_type) + '\n' + \
             'corrdatapath = {} '.format(corrdatapath)+ '\n'+ \
             'distancepath = {} '.format(distancepath) + '\n' + \
             'distance_type = {}'.format(distance_type) + '\n' + \
             'recurrent_unit = {}'.format(recurrent_unit) + '\n' + \
             'batch_size = {} '.format(batch_size)+ '\n'+\
             'sampling_num = {} '.format(sampling_num)+ '\n'+\
             'incell = {}'.format(incell)+ '\n'+ \
             'stard_unit = {}'.format(stard_unit)
   return configs


if __name__ == '__main__':
    print ('../model/model_training_600_{}_acc_{}'.format((0),1))
    print (config_to_str())
