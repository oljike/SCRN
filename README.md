# SCRN
## Regularized Structurally Constrained Recurrent neural networks

Code for the paper [Reproducing and Regularizing the SCRN Model]
(http://aclweb.org/anthology/C18-1145) (COLING 2018)

### Requirements
Code is written in Python 3 and requires TensorFlow 1.1+. It also requires the following Python modules: `numpy`, `pyphen`, `argparse`. You can install them via:
```
sudo pip3 install numpy 
```

### Data
Data should be put into the `data/` directory, split into `train.txt`, `valid.txt`, and `test.txt`. Each line of the .txt file should be a sentence. The English Penn Treebank (PTB) data is given as the default.

### Saves
To save a model a separate folder 'saves/' should be created

### Model
To reproduce the SCRN medium configuration result on English PTB from Table 3
```
python3 SCRN_word_model.py
```
