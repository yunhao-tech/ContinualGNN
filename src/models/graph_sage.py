import sys
import logging
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')
from layers.sage_conv import SAGEConv
from layers.sampler import Sampler
from layers.aggregator import Aggregator


class GraphSAGE(nn.Module):

    def __init__(self, layers, in_features, adj_lists, args):
        super(GraphSAGE, self).__init__()
        # layers contains an input layer = feature_size, an output layer = label_size and num_layers sage_conv layers
        self.layers = layers
        self.num_layers = len(layers) - 2
        self.in_features = torch.Tensor(in_features).to(args.device)
        self.adj_lists = adj_lists # list of edges, or sparse representation of adjacent matrix
        self.num_neg_samples = args.num_neg_samples # no effect, dead variable
        self.device = args.device

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(SAGEConv(layers[i], layers[i + 1]))
        self.sampler = Sampler(adj_lists) # sample from immediate neighbourhoods
        self.aggregator = Aggregator()

        self.weight = nn.Parameter(torch.Tensor(layers[-2], layers[-1]))
        self.xent = nn.CrossEntropyLoss()

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)


    def forward(self, nodes):
        # nodes: node attributes
        layer_nodes, layer_mask = self._generate_layer_nodes(nodes)
        features = self.in_features[layer_nodes[0]]
        for i in range(self.num_layers):
            cur_nodes, mask = layer_nodes[i + 1], layer_mask[i]            
            aggregate_features = self.aggregator.aggregate(mask, features)
            features = self.convs[i].forward(x = features[cur_nodes], aggregate_x = aggregate_features)
        return nn.functional.log_softmax(torch.matmul(features, self.weight), 1)
    

    def loss(self, nodes, labels = None):
        preds = self.forward(nodes)
        return self.xent(preds, labels.squeeze())


    def _generate_layer_nodes(self, nodes):
        # layer_nodes stores a list of nodes in each layer, neighbors are sampled at each layer
        # layer_nodes initialized with input nodes
        layer_nodes = list([nodes])
        layer_mask = list()
        for i in range(self.num_layers):
            # sample neighbor nodes of center nodes
            nodes_idxs, unique_neighs, mask = self.sampler.sample_neighbors(layer_nodes[0])
            layer_nodes[0] = nodes_idxs
            layer_nodes.insert(0, unique_neighs)
            layer_mask.insert(0, mask.to(self.device))
            # Don't quite understand the insert logic here, why put the neighbor nodes in front of the center nodes?
        return layer_nodes, layer_mask


    def get_embeds(self, nodes):
        # can be integrated into forward
        layer_nodes, layer_mask = self._generate_layer_nodes(nodes)
        features = self.in_features[layer_nodes[0]]
        for i in range(self.num_layers):
            cur_nodes, mask = layer_nodes[i + 1], layer_mask[i]            
            aggregate_features = self.aggregator.aggregate(mask, features)
            features = self.convs[i].forward(x = features[cur_nodes], aggregate_x = aggregate_features)
        return features.data.numpy()