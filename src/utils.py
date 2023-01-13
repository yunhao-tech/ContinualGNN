import argparse
import logging
import numpy as np
import random
import logging
import torch
from torch_geometric.data import Data
import os


def parse_argument():
    parser = argparse.ArgumentParser(description = 'pytorch version of GraphSAGE')
    # data options
    parser.add_argument('--data', type = str, default = 'cora')
    
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--seed', type = int, default = 13)
    parser.add_argument('--cuda', action = 'store_true', help = 'use CUDA')
    parser.add_argument('--num_neg_samples', type = int, default = 10)
    parser.add_argument('--num_layers', type = int, default = 2)
    parser.add_argument('--embed_size', type = int, default = 64)
    parser.add_argument('--learning_rate', type = float, default = 0.1)

    parser.add_argument('--detect_strategy', type = str, default = 'bfs')  # 'simple' / 'bfs'
    parser.add_argument('--new_ratio', type = float, default = 0.0)

    parser.add_argument('--memory_size', type = int, default = 0)
    parser.add_argument('--memory_strategy', type = str, default = 'class')   # 'random' / 'class'
    parser.add_argument('--p', type = float, default = 1)
    parser.add_argument('--alpha', type = float, default = 0.0)
    
    parser.add_argument('--ewc_lambda', type = float, default = 0.0)
    parser.add_argument('--ewc_type', type = str, default = 'ewc')  # 'l2' / 'ewc'

    parser.add_argument('--eval', action = 'store_true')

    args = parser.parse_args()

    return args

def print_args(args):
    config_str = 'Parameters: '
    for name, value in vars(args).items():
        config_str += str(name) + ': ' + str(value) + '; '
    logging.info(config_str)


def check_device(cuda):
    if torch.cuda.is_available():
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device_id = torch.cuda.current_device()
            print('using device', device_id, torch.cuda.get_device_name(device_id))
    device = torch.device("cuda" if cuda else "cpu")
    logging.info('Device:' + str(device))
    return device



def node_classification(trut, pred, name = ''):
    from sklearn import metrics
    f1 = np.round(metrics.f1_score(trut, pred, average="macro"), 6)
    acc = np.round(metrics.f1_score(trut, pred, average="micro"), 6)
    logging.info(name + '   Macro F1:' +  str(f1) \
            + ";    Micro F1:" +  str(acc))
    return f1, acc

# ------------------------------------------------------------------------------------------------------

def update_viewed_training_nodes_and_edges(coming_edges, viewed_training_nodes, viewed_training_edges, valid_all_nodes_list):
    """
    Parameters
    ----------
    coming_edges : ndarray of shape (num_coming_edges, 2)
            the streaming edges at one timestep
    
    viewed_training_nodes : 1D list
            all viewed nodes in previous steps that are included in training set
            set to "None" for timestep 0 
    
    viewed_training_edges : ndarray of shape (num_edges, 2)
            the edges between nodes in "viewed_training_nodes"
            set to "None" for timestep 0 
    
    valid_all_nodes_list : 1D list
            pre-defined valid set 
                            
    Returns
    ----------
    updated_viewed_training_nodes : 1D list
            updated viewed training nodes, taking into account the nodes in coming edges
    
    updated_viewed_training_edges : ndarray of shape (new_num_edges, 2)
            updated viewed training edges, taking into account the coming edges
    """
    if viewed_training_nodes is None:
        if viewed_training_edges is not None:
            raise ValueError("Please check input: if viewed nodes or viewed edges is None, another must also be None.")
        nodes = set(coming_edges.reshape(-1,))
        training_nodes = list(nodes - set(valid_all_nodes_list))
        training_edges = np.array([edge for edge in coming_edges if (edge[0] in training_nodes) and
                        (edge[1] in training_nodes)])
        return training_nodes, training_edges
    
    coming_nodes = set(coming_edges.reshape(-1,))
    viewed_nodes = coming_nodes | set(viewed_training_nodes)
    updated_viewed_training_nodes = list(viewed_nodes - set(valid_all_nodes_list))
    new_training_edges = [edge for edge in coming_edges if (edge[0] in updated_viewed_training_nodes) and
                        (edge[1] in updated_viewed_training_nodes)]
    updated_viewed_training_edges = np.concatenate((viewed_training_edges, new_training_edges), axis=0)
    return updated_viewed_training_nodes, updated_viewed_training_edges


def generate_graph(viewed_training_nodes, viewed_training_edges, all_features, all_labels):
    """
    Parameters
    ----------
    viewed_training_nodes : 1D list
            all viewed nodes before this timestep (inclus) that are included in training set
    
    viewed_training_edges : ndarray of shape (num_edges, 2)
            the edges between nodes in "viewed_training_nodes"
    
    all_features : ndarray of shape (num_nodes, num_features)-->(2708, 1433)
            features of all nodes, including training set and valid set
    
    all_labels : ndarray of shape (num_nodes,)-->(2708,)
            label of all nodes, including training set and valid set
                            
    Returns
    ----------
    graph : of type torch_geometric.data.data.Data
            constructed by "viewed_training_nodes" and "viewed_training_edges"
    """
    nodes_ranged = {node: number for number, node in enumerate(list(viewed_training_nodes))}
    inverse_edges = ([[edge[1], edge[0]] for edge in viewed_training_edges])
    training_edges = np.concatenate((viewed_training_edges, inverse_edges), axis=0)
    cur_edges_ranged = torch.tensor([[nodes_ranged[edge[0]],
                                      nodes_ranged[edge[1]]] for edge in training_edges],
                                    dtype=torch.long)
    cur_graph = Data(x=all_features[viewed_training_nodes], y=all_labels[viewed_training_nodes], 
                     edge_index=cur_edges_ranged.t().contiguous(), num_nodes=len(viewed_training_nodes))
    cur_graph.validate(raise_on_error=True)
    return cur_graph

