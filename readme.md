# M2 Data Streaming course repo

## Project: [River](https://github.com/online-ml/river)-implementation of [Streaming GNN](https://github.com/Junshan-Wang/ContinualGNN)

### General suggestions for contribution to River source code

1. read the [contribution guidelines](https://github.com/online-ml/river/blob/main/CONTRIBUTING.md)
2. read [regression](https://github.com/online-ml/river/blob/601d7b21919e2cec03dafc3c0e04e71251182495/river/linear_model/lin_reg.py) class as a starting point, read its documentation and code style.
3. implement the target method
4. compare it with [onelearn](https://github.com/onelearn/onelearn), an already made but no more active online learning package.
5. ~~avoid numpy dependency (as well as other packages) that focuses on batch models which poorly performs on one-shot data steam~~ according to Max, the deep learning method will go to this [repo](https://github.com/online-ml/river-torch), where we are free to use torch or numpy, the main river package is in charge of simple methods.
6. a not good enough implementation may temporally go to [river-extra](https://github.com/online-ml/river-extra) package.
7. ask on discord or on github
8. according to Cedric, the deep-river maintainer, integrate GNN to current river api is too complicated(particularly river requires `dict` type data. translate gnn data structure to `dict` will be nothing but painful). Meanwhile, as we dive into the proposed algorithm, we find it doesn't follow the online learning paradigm but rather the continual learning paradigm, which has access to data in previous snapshots. Or maybe the online paradigm for graph learning remains to be clarified.

### Special suggestion on project

1. use `scikit-network` instead of `networkx`
2. ask MB, she knows.

### Cora

Cora is a typical dataset for gnn research. It is a directed graph whose edge represents the reference relationship between two papers. The node attribute is a 1433-dim boolean array that represents the occurrence of keyword.
