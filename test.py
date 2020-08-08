import pickle as cPickle
from tools import config
import torch

uwave = 'UWaveGestureLibraryAll_discret_GRU_training_4400_393_incellTrue_config_False_0.01_20_10_0.2_UWaveGestureLibraryAll_False_128_train_0.4662_test_0.58628_0.8288_12905.1692071_14500.0193455_1594.85013838.h5'
italy = 'ItalyPowerDemand_discret_GRU_training_1000_1974_incellTrue_config_False_0.01_20_10_0.2_ItalyPowerDemand_False_128_train_0.453_test_0.6018_0.8234_3948.94110815_4628.19846882_679.257360669.h5'
electric = 'ElectricDevices_discret_GRU_training_4800_252_incellTrue_config_False_0.01_20_10_0.2_ElectricDevices_False_128_train_0.5112_test_0.6012_0.7158_1359.37024331_2190.64210547_831.271862163.h5'

if __name__ == '__main__':
    print(config.distancepath)
    dis = cPickle.load(open(config.distancepath))
    print (dis.shape)
    print (dis[0])
