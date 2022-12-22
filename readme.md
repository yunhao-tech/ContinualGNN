# M2 Data Streaming course repo

## Project: [River](https://github.com/online-ml/river)-implementation of [Streaming GNN](https://github.com/Junshan-Wang/ContinualGNN)

### General suggestions for contribution to River source code

1. read the [contribution guidelines](https://github.com/online-ml/river/blob/main/CONTRIBUTING.md)
2. read [regression](https://github.com/online-ml/river/blob/601d7b21919e2cec03dafc3c0e04e71251182495/river/linear_model/lin_reg.py) class as a starting point, read its documentation and code style.
3. implement the target method
4. compare it with [onelearn](https://github.com/onelearn/onelearn), an already made but no more active online learning package.
5. avoid numpy dependency (as well as other packages) that focuses on batch models which poorly performs on one-shot data steam
6. a not good enough implementation may temporally go to [river-extra](https://github.com/online-ml/river-extra) package.
7. ask on discord or on github

### Special suggestion on project

1. use `scikit-network` instead of `networkx`
2. ask MB, she knows.
