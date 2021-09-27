
## Execution

M2H2 dataset: python m2h2.py

### Command-Line Arguments

* --no-cuda: Does not use GPU
* --lr: Learning rate
* --l2: L2 regularization weight
* --rec-dropout: Recurrent dropout
* --dropout: Dropout
* --batch-size: Batch size
* --epochs: Number of epochs
* --attention: Attention type
* --tensorboard: Enables tensorboard log


## Deactivate MISA

In m2h2.py line comment line 87 to 94 and line  197 to deactivate MISA and uncomment line 100 and 198. (join.py and utils folder are required to working with MISA)

## m2h2.pkl

You can download this file from [here](https://drive.google.com/file/d/1TFtoJMAPYoJdnXKsQpNgGXU3rSSMell6/view?usp=sharing)

## Requirements


* Python 3
* PyTorch 1.0
* Pandas 0.23
* Scikit-Learn 0.20
* TensorFlow (optional; required for tensorboard)
* tensorboardX (optional; required for tensorboard)




