from sklearn import metrics
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import pairwise_distances
import pickle as cPickle
from tools import config
import random
import time
from sklearn_extra.cluster import KMedoids
import numpy as np

random.seed(2019)

def cluster_evaluatation(labels_true, labels):
    n_gt_list, n_e_list, homo, comple, v_mea, adj_r, adj_m, sliho = \
        [], [], [], [], [], [], [], []
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_gt = len(set(labels_true)) - \
        (1 if -1 in labels_true else 0)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_gt_list.append(n_clusters_gt)
    n_e_list.append(n_clusters_)
    homo.append(metrics.homogeneity_score(labels_true, labels))
    comple.append(metrics.completeness_score(labels_true, labels))
    v_mea.append(metrics.v_measure_score(labels_true, labels))
    adj_r.append(metrics.adjusted_rand_score(labels_true, labels))

    print("Homogeneity/Completeness/V-measure/Adjusted Rand Index")
    print("{:.4f} & {:.4f} & {:.4f} & {:.4f}".format(
        homo[0], comple[0], v_mea[0], adj_r[0]))


def trajectory_clustering_exp_Hier_KM(gt_labels_path, embedding_path, dis_path, data_samples, k):
    gt_labels = cPickle.load(open(gt_labels_path,'rb'))[0]
    embeddings = cPickle.load(open(embedding_path, 'rb'))
    embedding_distance = pairwise_distances(embeddings)
    gt_distance = cPickle.load(open(dis_path, 'rb'))

    print(embedding_distance.shape, gt_distance.shape)
    print('------------------------------------')
    s_t = time.time()
    gt_db = [int(i)-1 for i in gt_labels[:data_samples]]
    labels_true = gt_db
    s_t = time.time()
    print("***********AgglomerativeClustering***********")
    db = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(
        embedding_distance[:data_samples, :data_samples])
    labels_emb = list(db.labels_)
    print("Embedding based distance: ")
    cluster_evaluatation(labels_true, labels_emb)

    db = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(
        gt_distance[:data_samples, :data_samples])
    labels_ext = list(db.labels_)
    print("Exact distance: ")
    cluster_evaluatation(labels_true, labels_ext)
    print("Embedding vs Exact distance: ")
    cluster_evaluatation(labels_ext, labels_emb)

    print('***********KMedoids***********')
    db = KMedoids(n_clusters=k, metric='precomputed').fit(
    embedding_distance[:data_samples, :data_samples])
    labels_emb = list(db.labels_)
    print("Embedding based distance: ")
    cluster_evaluatation(labels_true, labels_emb)

    db = KMedoids(n_clusters=k, metric='precomputed').fit(
        gt_distance[:data_samples, :data_samples])
    labels_ext = list(db.labels_)
    print("Exact distance: ")
    cluster_evaluatation(labels_true, labels_ext)
    print("Embedding vs Exact distance: ")
    cluster_evaluatation(labels_ext, labels_emb)

if __name__ == '__main__':
    gt_dis = ['./features/UWaveGestureLibraryAll_discret_frechet_distance_all_4400',
              './features/ItalyPowerDemand_discret_frechet_distance_all_1000',
              './features/ElectricDevices_discret_frechet_distance_all_8100']
    em_path = ['./features/embedding_4400_UWaveGestureLibraryAll_discret',
               './features/embedding_1000_ItalyPowerDemand_discret',
               './features/embedding_8100_ElectricDevices_discret']

    gt_labels = ['./features/UWaveGestureLibraryAll_ts_label',
                 './features/ItalyPowerDemand_all_ts_label',
                 './features/ElectricDevices_all_ts_label']

    trajectory_clustering_exp_Hier_KM(
        gt_labels[0], em_path[0], gt_dis[0], 4400, 8)
    trajectory_clustering_exp_Hier_KM(
        gt_labels[1], em_path[1], gt_dis[1], 1000, 2)
    trajectory_clustering_exp_Hier_KM(
        gt_labels[2], em_path[2], gt_dis[2], 8100, 5)
