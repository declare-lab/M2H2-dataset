====================================================
Execution
====================================================
M2H2 dataset: python m2h2.py

Command-Line Arguments

--no-cuda: Does not use GPU
--lr: Learning rate
--l2: L2 regularization weight
--rec-dropout: Recurrent dropout
--dropout: Dropout
--batch-size: Batch size
--epochs: Number of epochs
--attention: Attention type
--tensorboard: Enables tensorboard log


Deactivate MISA

In m2h2.py line comment line 87 to 94 and line  197 to deactivate MISA and uncomment line 100 and 198. (join.py and utils folder are required to working with MISA)


====================================================
Requirements
====================================================

1.Python 3
2.PyTorch 1.0
3.Pandas 0.23
4.Scikit-Learn 0.20
5.TensorFlow (optional; required for tensorboard)
6.tensorboardX (optional; required for tensorboard)




