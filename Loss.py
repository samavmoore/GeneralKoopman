import torch
import torch.nn as nn
import torch.nn.functional as F

class l2_reg(nn.Module):
    def __init__(self):
        super(l2_reg, self).__init__()

    def forward(self, params):
        sum = 0
        for name, param in params:
            if "weight" in name:
                sum += torch.linalg.norm(param)**2
        return sum

class inf_loss(nn.Module):
    def __init__(self):
        super(inf_loss, self).__init__()

    def forward(self, states, estimated_state):
        norm = torch.linalg.norm(states-estimated_state, float('inf'))
        return norm

class context_recon_loss(nn.Module):
    def __init__(self):
        super(context_recon_loss, self).__init__()

    def forward(self, context, estimated_context):
        return F.mse_loss(context, estimated_context)

class state_recon_loss(nn.Module):
    def __init__(self):
        super(state_recon_loss, self).__init__()
        
    def forward(self, states, estimated_state):
        return F.mse_loss(states, estimated_state)

class future_embedding_loss(nn.Module):
    def __init__(self):
        super(future_embedding_loss, self).__init__() 

    def forward(self, future_embeddings, estimated_future_embeddings):
        return F.mse_loss(future_embeddings, estimated_future_embeddings)

class future_state_loss(nn.Module):
    def __init__(self):
        super(future_state_loss, self).__init__() 
        
    def forward(self, future, estimated_future_state):
        return F.mse_loss(future, estimated_future_state)