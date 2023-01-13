from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GraphSAGE

from torch import Tensor
from torch_geometric.typing import Adj

# according to pyG BasicGNN source code https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html
# and example https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
class Model(torch.nn.Module):
	def __init__(
		self,
		in_channels: int,
		hidden_channels: int,
		out_channels: int,
		num_layers: int = 3, #including the last linear one
		ewc_type = Literal['ewc','l2'],
		ewc_lambda: float = 0,
	):
		super().__init__()

		self.sage = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers-1)
		self.lin = Linear(in_features=hidden_channels, out_features=out_channels)
		self.ewc_type = ewc_type
		self.ewc_lambda = ewc_lambda

	def reset_parameters(self):
		self.sage.reset_parameters()
		self.lin.reset_parameters()

	def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
		x = self.sage(x=x, edge_index=edge_index)
		x = self.lin(x)
		return F.log_softmax(x, 1)

	def save(self, path):
		torch.save(self, path)
		
	# seems not necessary for the moment
	# @torch.no_grad()
	# def inference(self, loader: NeighborLoader,
	# 							device: Optional[torch.device] = None,
	# 							progress_bar: bool = False) -> Tensor:
	# 		r"""Performs layer-wise inference on large-graphs using
	# 		:class:`~torch_geometric.loader.NeighborLoader`.
	# 		:class:`~torch_geometric.loader.NeighborLoader` should sample the the
	# 		full neighborhood for only one layer.
	# 		This is an efficient way to compute the output embeddings for all
	# 		nodes in the graph.
	# 		Only applicable in case :obj:`jk=None` or `jk='last'`.
	# 		"""
	# 		assert isinstance(loader, NeighborLoader)
	# 		assert len(loader.dataset) == loader.data.num_nodes
	# 		assert len(loader.node_sampler.num_neighbors) == 1
	# 		assert not self.training

	# 		if progress_bar:
	# 				pbar = tqdm(total=len(self.convs) * len(loader))
	# 				pbar.set_description('Inference')

	# 		x_all = loader.data.x.cpu()
	# 		loader.data.n_id = torch.arange(x_all.size(0))
	# 		for conv in range(self.convs):
	# 				xs: List[Tensor] = []
	# 				for batch in loader:
	# 						x = x_all[batch.n_id].to(device)
	# 						if hasattr(batch, 'adj_t'):
	# 								edge_index = batch.adj_t.to(device)
	# 						else:
	# 								edge_index = batch.edge_index.to(device)
	# 						x = conv(x, edge_index)[:batch.batch_size]
	# 						x = F.relu(x)
	# 						xs.append(x.cpu())
	# 						if progress_bar:
	# 								pbar.update(1)
	# 				x_all = torch.cat(xs, dim=0)
	# 		if progress_bar:
	# 				pbar.close()
	# 		return x_all