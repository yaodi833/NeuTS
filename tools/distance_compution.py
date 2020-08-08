import pickle as cPickle
import tools.traj_dist.distance as tdist
import numpy as np
import multiprocessing
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
from pydtw import dtw2d

def trajectory_distance(traj_feature_map, traj_keys, distance_type="hausdorff", batch_size=50, processors=30):
    # traj_keys= traj_feature_map.keys()
    trajs = []
    for k in traj_keys:
        traj = []
        for record in traj_feature_map[k]:
            traj.append([record[1], record[2]])
        trajs.append(np.array(traj))

    pool = multiprocessing.Pool(processes=processors)
    # print np.shape(distance)
    batch_number = 0
    for i in range(len(trajs)):
        if (i != 0) & (i % batch_size == 0):
            print(batch_size * batch_number, i)
            # trajectory_distance_batch(i, trajs[batch_size * batch_number:i], trajs, distance_type,
            #                                              'geolife')
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size * batch_number:i], trajs, distance_type,
                                                         'geolife'))
            batch_number += 1
    pool.close()
    pool.join()


def trajecotry_distance_list(trajs, distance_type="hausdorff", batch_size=50, processors=30, data_name='porto'):
    # trajectory_distance_batch(1, trajs[0: batch_size], trajs, distance_type,
    #                           data_name)
    pool = multiprocessing.Pool(processes=processors)
    # print np.shape(distance)
    batch_number = 0
    for i in range(len(trajs)):
        if (i != 0) & (i % batch_size == 0):
            print(batch_size * batch_number, i)
            # trajectory_distance_batch(i, trajs[batch_size * batch_number:i], trajs, distance_type,
            #                   data_name)
            pool.apply_async(trajectory_distance_batch,
                             (i, trajs[batch_size * batch_number:i], trajs, distance_type,
                              data_name))
            batch_number += 1
    pool.close()
    pool.join()


def trajectory_distance_batch(i, batch_trjs, trjs, metric_type="hausdorff", data_name='porto', is_store = True):
    if metric_type == 'cdtw':
        trs_matrix = tdist.cdist(
            batch_trjs, trjs, metric=metric_type, eps=1)
    elif metric_type == 'hausdorff' and len(trjs[0][0]) != 2:
        trs_matrix = compute_hausdorff(batch_trjs, trjs)
    # elif metric_type == 'dtw':
    #     trs_matrix = compute_dtw(batch_trjs, trjs)
    else:
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type)
    if is_store==True:
        cPickle.dump(trs_matrix, open('./features/' + data_name +
                                      '_' + metric_type + '_distance_' + str(i), 'wb'))
    print('complete: ' + str(i))
    return trs_matrix


def compute_hausdorff(traj_list_1, traj_list_2):
    nb_traj_1 = len(traj_list_1)
    nb_traj_2 = len(traj_list_2)
    M = np.zeros((nb_traj_1, nb_traj_2))
    for i in range(nb_traj_1):
        traj_list_1_i = np.array(traj_list_1[i])
        for j in range(nb_traj_2):
            traj_list_2_j = np.array(traj_list_2[j])
            d1 = directed_hausdorff(traj_list_1_i, traj_list_2_j)
            d2 = directed_hausdorff(traj_list_2_j, traj_list_1_i)
            M[i, j] = max(d1[0], d2[0])
    return M


def compute_dtw(traj_list_1, traj_list_2, radial = 1):
    nb_traj_1 = len(traj_list_1)
    nb_traj_2 = len(traj_list_2)
    M = np.zeros((nb_traj_1, nb_traj_2))
    for i in range(nb_traj_1):
        traj_list_1_i = np.array(traj_list_1[i])
        print ('complete fastdtw: {}'.format(i))
        for j in range(nb_traj_2):
            traj_list_2_j = np.array(traj_list_2[j])
            # print(j, traj_list_2_j.shape, traj_list_1_i.shape)
            # _,d1,_, _ = dtw2d(traj_list_1_i, traj_list_2_j)
            # _,d2,_, _ = dtw2d(traj_list_2_j, traj_list_1_i)
            d1, _ = fastdtw(traj_list_1_i, traj_list_2_j, dist=euclidean, radial = radial)
            # d2, _ = fastdtw(traj_list_2_j, traj_list_1_i, dist=euclidean)
            M[i, j] = d1
    return M

def trajectory_distance_combain(trajs_len, batch_size=100, metric_type="hausdorff", data_name='porto'):
    distance_list = []
    a = 0
    for i in range(1, trajs_len + 1):
        if (i != 0) & (i % batch_size == 0):
            distance_list.append(
                cPickle.load(open('./features/' + data_name + '_' + metric_type + '_distance_' + str(i), 'rb')))
            print(distance_list[-1].shape)
    a = distance_list[-1].shape[1]
    distances = np.array(distance_list)
    print(distances.shape)
    all_dis = distances.reshape((trajs_len, a))
    print(all_dis.shape)
    cPickle.dump(all_dis, open('./features/' + data_name + '_' +
                               metric_type + '_distance_all_' + str(trajs_len), 'wb'))
    return all_dis


