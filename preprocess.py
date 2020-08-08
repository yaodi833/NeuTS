from tools.preprocess_ts import feature_generation, feature_generation_3D
from tools.distance_compution import trajectory_distance_combain, trajecotry_distance_list
import pickle as cPickle
import numpy as np


def distance_comp(coor_path, data_name, number, distance_type='hausdorff'):
    ts_value = cPickle.load(open(coor_path, 'rb'))[0]
    np_ts_value = []
    for t in ts_value[:number+5]:
        np_ts_value.append(np.array(t))
    print(np_ts_value[0])
    print(np_ts_value[0].shape)
    print(len(np_ts_value))

    trajecotry_distance_list(np_ts_value, batch_size=100, processors=28, distance_type=distance_type,
                             data_name=data_name)

    trajectory_distance_combain(
        number, batch_size=100, metric_type=distance_type, data_name=data_name)


if __name__ == '__main__':
    value_path, data_name = feature_generation(path='./data/ItalyPowerDemand/')
    value_path, data_name = feature_generation_3D(
        path='./data/UWaveGestureLibraryAll/')
    value_path, data_name = feature_generation(path='./data/ElectricDevices/')

    # distance_comp('./features/ItalyPowerDemand_all_ts_value',
    #               'ItalyPowerDemand', 1000, distance_type='discret_frechet')
    # distance_comp('./features/ItalyPowerDemand_all_ts_value',
    #               'ItalyPowerDemand', 1000, distance_type='cdtw')
    # distance_comp('./features/ItalyPowerDemand_all_ts_value',
    #               'ItalyPowerDemand', 1000, distance_type='dtw')
    # distance_comp('./features/ItalyPowerDemand_all_ts_value',z
    #               'ItalyPowerDemand', 1000, distance_type='erp')
    # distance_comp('./features/ItalyPowerDemand_all_ts_value',
    #               'ItalyPowerDemand', 1000,  distance_type='hausdorff')


    # distance_comp('./features/ElectricDevices_selected_all_ts_value',
    #               'ElectricDevices', 8100, distance_type='discret_frechet')
    # distance_comp('./features/ElectricDevices_selected_all_ts_value',
    #               'ElectricDevices', 8100, distance_type='erp')
    # distance_comp('./features/ElectricDevices_selected_all_ts_value',
    #               'ElectricDevices', 8100,  distance_type='hausdorff')
    # distance_comp('./features/ElectricDevices_selected_all_ts_value',
    #               'ElectricDevices', 8100, distance_type='cdtw')
    # distance_comp('./features/ElectricDevices_selected_all_ts_value',
    #               'ElectricDevices', 8100, distance_type='dtw')

    # distance_comp('./features/UWaveGestureLibraryAll_ts_value', 'UWaveGestureLibraryAll', 4400, distance_type='discret_frechet')
    # distance_comp('./features/UWaveGestureLibraryAll_ts_value',   'UWaveGestureLibraryAll',4400,  distance_type = 'hausdorff')
    # distance_comp('./features/UWaveGestureLibraryAll_ts_value','UWaveGestureLibraryAll', 4400, distance_type='cdtw')
    # distance_comp('./features/UWaveGestureLibraryAll_ts_value', 'UWaveGestureLibraryAll',4400, distance_type='erp')
    # distance_comp('./features/UWaveGestureLibraryAll_ts_value','UWaveGestureLibraryAll', 4400, distance_type='dtw')
