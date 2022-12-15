# General Koopman
## Requirements
- PyTorch Lightning
- SciPy
- Torch
- NumPy
## How to train the models
The models need to be trained in the order outlined in the manuscript. Either use the training and validation data in the data folder or collect your own data with pendulum_simulations. 

#### Running pendulum_training.py
The context AE must be trained first. Just set the path to config_data. Pretrain the eigenfunction networks next. Set the path to the context AE you just trained 

