from typing import Tuple
import numpy as np
import torch
from torch_geometric.data import Data

def update_viewed_training_nodes_and_edges(
	coming_edges: np.ndarray,
	viewed_training_nodes: list,
	viewed_training_edges: np.ndarray,
	valid_all_nodes_list: list) -> Tuple[list, np.ndarray]:
    """
    Parameters
    ----------
    coming_edges : ndarray of shape (num_coming_edges, 2)
            the streaming edges at one timestep
    
    viewed_training_nodes : 1D list
            all viewed nodes until current timestep that are in training set
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
    new_training_edges = np.array([edge for edge in coming_edges if (edge[0] in updated_viewed_training_nodes) and
                        (edge[1] in updated_viewed_training_nodes)])
    # updated_viewed_training_edges = np.concatenate((viewed_training_edges, new_training_edges), axis=0)
    updated_viewed_training_edges_dup = np.concatenate((viewed_training_edges, new_training_edges), axis=0)
    updated_viewed_training_edges_dup = np.unique(updated_viewed_training_edges_dup, axis=0)
    updated_viewed_training_edges = []
    for edge in updated_viewed_training_edges_dup:
        if ([edge[0], edge[1]] not in updated_viewed_training_edges) and \
                ([edge[1], edge[0]] not in updated_viewed_training_edges):
            updated_viewed_training_edges.append(list(edge))
    return updated_viewed_training_nodes, np.array(updated_viewed_training_edges)

def generate_graph(
	viewed_training_nodes: list,
	viewed_training_edges: np.ndarray,
	all_features: np.ndarray,
	all_labels: np.ndarray) -> Data:
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
    graph : of type torch_geometric.data.Data
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

def update_viewed_all_nodes_and_edges(
	coming_edges: np.ndarray,
	viewed_all_nodes: list,
	viewed_all_edges: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Parameters
    ----------
    coming_edges : ndarray of shape (num_coming_edges, 2)
            the streaming edges at one timestep
    
    viewed_all_nodes : 1D list
            all viewed nodes until current timestep
            set to "None" for timestep 0 
    
    viewed_all_edges : ndarray of shape (num_edges, 2)
            the edges between nodes in "viewed_all_nodes"
            set to "None" for timestep 0 
    
    valid_all_nodes_list : 1D list
            pre-defined valid set 
                            
    Returns
    ----------
    updated_viewed_all_nodes : 1D list
            updated viewed nodes, taking into account the nodes in coming edges
    
    updated_viewed_all_edges : ndarray of shape (new_num_edges, 2)
            updated viewed edges, taking into account the coming edges
    """
    if viewed_all_nodes is None:
        if viewed_all_edges is not None:
            raise ValueError("Please check input: if viewed nodes or viewed edges is None, another must also be None.")
        nodes = list(set(coming_edges.reshape(-1,)))
        return nodes, coming_edges
    
    coming_nodes = set(coming_edges.reshape(-1,))
    updated_viewed_all_nodes = list(coming_nodes | set(viewed_all_nodes))
    updated_viewed_all_edges_dup = np.concatenate((viewed_all_edges, coming_edges), axis=0)
    updated_viewed_all_edges_dup = np.unique(updated_viewed_all_edges_dup, axis=0)
    updated_viewed_all_edges = []
    for edge in updated_viewed_all_edges_dup:
        if ([edge[0], edge[1]] not in updated_viewed_all_edges) and \
                ([edge[1], edge[0]] not in updated_viewed_all_edges):
            updated_viewed_all_edges.append(list(edge))
    return updated_viewed_all_nodes, np.array(updated_viewed_all_edges)

def generate_whole_graph(
	viewed_all_nodes: list,
	viewed_all_edges: np.ndarray,
	valid_all_nodes_list: list,
	all_features: np.ndarray,
	all_labels: np.ndarray) -> Tuple[Data, list]:
    """
    Parameters
    ----------
    viewed_all_nodes : 1D list
            all viewed nodes until current timestep
    
    viewed_all_edges : ndarray of shape (num_edges, 2)
            the edges between nodes in "viewed_all_nodes"
            
    valid_all_nodes_list : 1D list
            pre-defined valid set 
    
    all_features : ndarray of shape (num_nodes, num_features)-->(2708, 1433)
            features of all nodes, including training set and valid set
    
    all_labels : ndarray of shape (num_nodes,)-->(2708,)
            label of all nodes, including training set and valid set
                            
    Returns
    ----------
    graph : of type torch_geometric.data.Data
            constructed by "viewed_training_nodes" and "viewed_training_edges"
    
    valid_nodes : 1D list
            valid nodes index in the graph 
    """
    nodes_ranged = {node: number for number, node in enumerate(viewed_all_nodes)}
    valid_nodes = [number for (node, number) in nodes_ranged.items() if node in valid_all_nodes_list]
    inverse_edges = ([[edge[1], edge[0]] for edge in viewed_all_edges])
    all_edges = np.concatenate((viewed_all_edges, inverse_edges), axis=0)
    cur_edges_ranged = torch.tensor([[nodes_ranged[edge[0]],
                                      nodes_ranged[edge[1]]] for edge in all_edges],
                                    dtype=torch.long)
    cur_graph = Data(x=all_features[viewed_all_nodes], y=all_labels[viewed_all_nodes], 
                     edge_index=cur_edges_ranged.t().contiguous(), num_nodes=len(viewed_all_nodes))
    cur_graph.validate(raise_on_error=True)
    return cur_graph, valid_nodes