
import torch
import io
import copy


def K_blocks(omegas, delta_t):
    '''
        Purpose: Make a dictionary containing all of the blocks of our koopman operator. The values from this dictionary are
        later passed into sci_py's block_diag function to construct K

        Inputs: Estimated real part (omegas[even]) and imaginary part (omegas[odd]). Must be an even number of elements in omegas
                    
        Outputs: Dictionary with keys block# and values 2x2 tensor with entries exp(real*dt)*[[cos(imag*dt), -sin(imag*dy)],[sin(imag*dt), cos(imag*dt)]]

    '''
    #initialize dictionary
    blocks = {}

    for i in range(0, len(omegas)-1, 2):

        # get real part
        R_part = omegas[i]
        # get imaginary part
        I_part = omegas[i+1]
    
        dt = delta_t

        growth_decay = torch.exp(R_part*dt)
        oscillations = torch.tensor([[torch.cos(I_part*dt), -torch.sin(I_part*dt)], [torch.sin(I_part*dt), torch.cos(I_part*dt)]])
        oscillations = oscillations.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # store these blocks in the dictionary
        blocks[f"block {i}"] = torch.mul(oscillations, growth_decay)

    return blocks


class EarlyStopping():
  def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
    # Number of epochs that should be waited for the validation error improve.
    self.patience = patience

    # Minimum amount of improvement in validation error.
    self.min_delta = min_delta

    # Restores weights to what they were when validation error was the lowest.
    self.restore_best_weights = restore_best_weights

    self.best_model = None
    self.best_loss = None
    self.counter = 0
    self.status = ""
    
  def __call__(self, model, val_loss):
    # Setting values for the first epoch.
    if self.best_loss == None:
      self.best_loss = val_loss
      self.best_model = copy.deepcopy(model)

    # If current validation loss is better than the best so far.
    elif self.best_loss - val_loss > self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
      self.best_model.load_state_dict(model.state_dict())

    # If current validation loss is worse than the best, increase count and restore better weights.
    elif self.best_loss - val_loss < self.min_delta:
      self.counter += 1

      if self.counter >= self.patience:
        self.status = f"Stopped on {self.counter}"
        
        if self.restore_best_weights:
          model.load_state_dict(self.best_model.state_dict())

        return True
        
    self.status = f"{self.counter}/{self.patience}"
    return False
