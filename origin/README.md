# Streaming Graph Neural Networks via Continual Learning

Code for [Streaming Graph Neural Networks via Continual Learning](https://dl.acm.org/doi/abs/10.1145/3340531.3411963)（CIKM 2020). ContinualGNN is a streaming graph neural network based on continual learning so that the model is trained incrementally and up-to-date node representations can be obtained at each time step.

### Requirements

- python = 3.8.5
- pytorch = 1.7.1
- scikit-learn = 0.23.2

### Usages

- ContinualGNN (proposed model) on Cora:

```
cd src/
python main_stream.py --data=cora --new_ratio=0.8 --memory_size=250 --ewc_lambda=80.0
```

- OnlineGNN (lower bound) on Cora:

```
python main_stream.py --data=cora
```

If using cuda, set `--cuda`.

### Credit

The `origin` directory is a copy of https://github.com/Junshan-Wang/ContinualGNN

### Analysis of code structure

- `extensions`
  - `detection.py`: compute $\mathcal{I}(\Delta G^t)$
  - `memory_handler.py`: compute $\mathcal{M}$
- `handlers`
  - `data_handler.py`: load data, train-valid split
  - `model_handler.py`: handle load/save model for `main_sage` use
  - `stream_data_handler.py`: subclass of data_handler for `main_stream` use.
- `layers`:
  - `aggregator.py`: mask \* features
  - `sage_conv.py`: using concatenation instead of mean aggregator in the paper
  - `sampler.py`: sample 10 nodes from neighborhood
- `models`:
  - `ewc.py`: wrap the ewc loss around gnn model
  - `graph_sage.py`: GraphSage model
- `main_sage.py`: seems a test version for plain sage model
- `main_stream.py`: true final version
