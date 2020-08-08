import tools.config as config
import numpy as np
import torch.autograd as autograd
import torch
# from geo_rnns.spatial_memory_lstm_pytorch import SpatialCoordinateRNNPytorch
import time
import pickle as cPickle

def test_comput_embeddings(self, spatial_net, test_batch = 1025):
    if config.recurrent_unit=='GRU' or config.recurrent_unit=='SimpleRNN':
        hidden = autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda()
    else:
        hidden = (autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda(),
                  autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda())
    embeddings_list = []
    j = 0
    s = time.time()
    while j < self.padded_trajs.shape[0]:
        for i in range(self.batch_size):
            out = spatial_net.rnn([autograd.Variable(torch.Tensor(self.padded_trajs[j:j+test_batch]),
                                                     requires_grad=False).cuda(),
                                   self.trajs_length[j:j+test_batch]],hidden)
            # embeddings = out.data.cpu().numpy()
            embeddings = out.data
        j += test_batch
        embeddings_list.append(embeddings)
        if (j% 1000) == 0:
            print (j)
    print ('embedding time of {} trajectories: {}'.format(self.padded_trajs.shape[0], time.time()-s))
    embeddings_list = torch.cat(embeddings_list, dim=0)
    print (embeddings_list.size())
    return embeddings_list.cpu().numpy()

def classies_analysis(test_range, all_accuracies, label_path=config.LABEL_PATH):
    labels,_,_ = cPickle.load(open(label_path, 'rb'))
    print(len(labels))
    label_class_dict = {}
    for index, i in enumerate(test_range):
        if labels[i] in label_class_dict.keys():
            label_class_dict[labels[i]].append(all_accuracies[index])
        else:
            label_class_dict[labels[i]] = [all_accuracies[index]]
    for key in label_class_dict.keys():
        print('-------------------------------')
        sample_num = len(label_class_dict[key])
        accuracies = np.array(label_class_dict[key])
        print("Average HR10 of class {} : {}".format(
            key, sum(accuracies[:, 0])/sample_num))
        print("Average HR50 of class {} : {}".format(
            key, sum(accuracies[:, 1])/sample_num))
        print("Average R10@50 of class {} : {}".format(
            key, sum(accuracies[:, 2])/sample_num))

def test_model(self, traj_embeddings, test_range, print_batch=1000, similarity = False, r10in50 = False):
    top_10_count, l_top_10_count = 0, 0
    top_50_count, l_top_50_count = 0, 0
    top10_in_top50_count = 0
    test_traj_num = 0
    range_num = test_range[-1]
    all_true_distance, all_test_distance = [], []
    error_true, error_test, errorr1050 = 0., 0., 0.
    all_accuracies= []
    for i in test_range:

        if similarity:
            # This is for the exp similarity
            test_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[i] - e)))))
                             for j, e in enumerate(traj_embeddings)]
            t_similarity = np.exp(-self.distance[i][:len(traj_embeddings)]*config.mail_pre_degree)
            true_distance = list(enumerate(t_similarity))
            learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1], reverse=True)
            s_true_distance = sorted(true_distance, key=lambda a: a[1], reverse=True)
            s_learned_distance = sorted(learned_distance, key=lambda a: a[1])
        else:
            # This is for computing the distance
            test_distance = [(j, float(np.sum(np.square(traj_embeddings[i] - e))))
                             for j, e in enumerate(traj_embeddings)]
            true_distance = list(enumerate(self.distance[i][:len(traj_embeddings)]))
            learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1])
            s_true_distance = sorted(true_distance, key=lambda a: a[1])
            s_learned_distance = sorted(learned_distance, key=lambda a: a[1])

        top10_recall = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:11]]]
        top50_recall = [l[0] for l in s_test_distance[:51] if l[0] in [j[0] for j in s_true_distance[:51]]]
        top10_in_top50 = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:51]]]


        if len(top10_recall) - 1 < 0:
            print(s_true_distance[:11])
            print(s_test_distance[:11])
            print(top10_recall)
        else:
            top_10_count += len(top10_recall) - 1
            top_50_count += len(top50_recall) - 1
            top10_in_top50_count += len(top10_in_top50) - 1

        all_accuracies.append(
            [float(len(top10_recall) - 1)/10, float(len(top50_recall) - 1)/50, float(len(top10_in_top50) - 1)/10])

        l_top10_recall = [l[0] for l in s_learned_distance[:11] if l[0] in [j[0] for j in s_true_distance[:11]]]
        l_top50_recall = [l[0] for l in s_learned_distance[:51] if l[0] in [j[0] for j in s_true_distance[:51]]]

        l_top_10_count += len(l_top10_recall) - 1
        l_top_50_count += len(l_top50_recall) - 1

        all_true_distance.append(s_true_distance[:50])
        all_test_distance.append(s_test_distance[:50])

        true_top_10_distance = 0.
        for ij in s_true_distance[:11]:
            true_top_10_distance += self.distance[i][ij[0]]
        test_top_10_distance = 0.
        for ij in s_test_distance[:11]:
            # print (i, ij)
            test_top_10_distance += self.distance[i][ij[0]]
        test_top_10_distance_r10in50 = 0.
        temp_distance_in_test50 = []
        for ij in  s_test_distance[:51]:
            temp_distance_in_test50.append([ij,self.distance[i][ij[0]]])
        sort_dis_10in50 = sorted(temp_distance_in_test50, key= lambda x: x[1])
        test_top_10_distance_r10in50 = sum([iaj[1] for iaj in sort_dis_10in50[:11]])

        error_true += true_top_10_distance
        error_test += test_top_10_distance
        errorr1050 += test_top_10_distance_r10in50

        test_traj_num += 1
        if (i+1 % print_batch) == 0:
            # print test_distance
            print ('**----------------------------------**\n{}\n{}\n{}\n{}'.format(s_test_distance[:20], s_true_distance[:20], top10_recall, top50_recall))

    if r10in50 :
        error_test = errorr1050

    print ('Test on {} trajs'.format(test_traj_num))
    print (')Search Top 50 recall {}'.format(float(top_50_count) / (test_traj_num * 50)))
    print ('Search Top 10 recall {}'.format(float(top_10_count) / (test_traj_num * 10)))
    print ('Search Top 10 in Top 50 recall {}'.format(float(top10_in_top50_count) / (test_traj_num * 10)))
    print ('Error true:{}'.format((float(error_true) / (test_traj_num * 10))*84000))
    print ('Error test:{}'.format((float(error_test) / (test_traj_num * 10))*84000))
    print ('Error div :{}'.format((float(abs(error_test-error_true)) / (test_traj_num * 10))*84000))

    classies_analysis(test_range, all_accuracies)
    return (float(top_10_count) / (test_traj_num * 10), \
           float(top_50_count) / (test_traj_num * 50),\
           float(top10_in_top50_count) / (test_traj_num * 10), \
           (float(error_true) / (test_traj_num * 10)) * 84000, \
           (float(error_test) / (test_traj_num * 10)) * 84000, \
           (float(abs(error_test - error_true)) / (test_traj_num * 10)) * 84000)


if __name__ == '__main__':
    print (config.config_to_str())
