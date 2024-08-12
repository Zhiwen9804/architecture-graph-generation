import os
import numpy as np
import pickle
import networkx as nx

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from utils.utils import load_dataset
from utils.graph_utils import graphs_to_tensor, mask_x, init_features, node_flags

def get_node_feature(graphs, max_node, max_feat_num):
    all_feature = torch.zeros([len(graphs), max_node, max_feat_num])
    for i in range(len(graphs)):
        feature = []
        for node, att in graphs[i].nodes(data=True):
            feature.append(att['feature'])
        feature = torch.tensor(np.array(feature))
        all_feature[i][:len(feature)] = feature
    return all_feature
    
def graphs_to_dataloader(graph_list, max_node_num, max_feat_num):

    adjs_tensor = graphs_to_tensor(graph_list, max_node_num)
    x_tensor = get_node_feature(graph_list, max_node_num, max_feat_num)
    flags = node_flags(graph_list, max_node_num)

    dataset = TensorDataset(x_tensor, adjs_tensor, flags)
    return dataset

def get_dataset(data_dir, file_name, max_node_num=40, max_feat_num=11):
    graph_list = load_dataset(data_dir, file_name)
    test_size = int(0.2 * len(graph_list))
    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
    train_dataset = graphs_to_dataloader(train_graph_list, max_node_num, max_feat_num)
    test_dataset = graphs_to_dataloader(test_graph_list, max_node_num, max_feat_num)
    return train_dataset, test_dataset

if __name__ == "__main__":   
    data_dir = '/home/zhiwen/architec/modify_door/dataset/nx_file/'
    file_name = 'att_no_edge_class'
    
    train, test = get_dataset(data_dir, file_name)