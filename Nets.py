import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(self, raw_context_shape: int=2,
                     context_rep_shape: int=1,
                     hid_layer_1_shape: int=64):
        super().__init__()

        self.context_encoder_l1 = nn.Linear(in_features=raw_context_shape, out_features=hid_layer_1_shape)
        self.context_encoder_l2 = nn.Linear(in_features=hid_layer_1_shape, out_features=hid_layer_1_shape)
        self.context_encoder_l3 = nn.Linear(in_features=hid_layer_1_shape, out_features=context_rep_shape)

    def forward(self, raw_context):
        ''' Purpose: Embed raw context in a latent space
        '''
        output1 = F.relu(self.context_encoder_l1(raw_context))
        output2 = F.relu(self.context_encoder_l2(output1))
        encoded_context = self.context_encoder_l3(output2)

        return encoded_context

class ContextDecoder(nn.Module):
    def __init__(self, raw_context_shape: int=2,
                     context_rep_shape: int=1,
                    hid_layer_1_shape: int=64):
        super().__init__()

        self.context_decoder_l1 = nn.Linear(in_features=context_rep_shape, out_features=hid_layer_1_shape)
        self.context_decoder_l2 = nn.Linear(in_features=hid_layer_1_shape, out_features=hid_layer_1_shape)
        self.context_decoder_l3 = nn.Linear(in_features=hid_layer_1_shape, out_features=raw_context_shape)

    def forward(self, encoded_context):
        ''' Purpose: Reconstruct context from a latent representation
        '''
        output1 = F.relu(self.context_decoder_l1(encoded_context))
        output2 = F.relu(self.context_decoder_l2(output1))
        context_hat = self.context_decoder_l3(output2)

        return context_hat

class Eigenfunction(nn.Module):
    def __init__(self, eigenfunction_input_shape: int=3,
                     eigenfunction_output_shape: int=2, 
                     eigenfunction_hidden_shape1: int=48, 
                     eigenfunction_hidden_shape2: int=96):
        super().__init__() 

        self.eigenfunction_l1 = nn.Linear(in_features=eigenfunction_input_shape, out_features=eigenfunction_hidden_shape1)
        self.eigenfunction_l2 = nn.Linear(in_features=eigenfunction_hidden_shape1, out_features=eigenfunction_hidden_shape2)
        self.eigenfunction_l3 = nn.Linear(in_features=eigenfunction_hidden_shape2, out_features=eigenfunction_hidden_shape2)
        self.eigenfunction_l4 = nn.Linear(in_features=eigenfunction_hidden_shape2, out_features=eigenfunction_hidden_shape1)
        self.eigenfunction_l5 = nn.Linear(in_features=eigenfunction_hidden_shape1, out_features=eigenfunction_output_shape)

    def forward(self, state, encoded_context):
        ''' Purpose: Put state into eigenfunction coordinates
        '''
        x = torch.cat((encoded_context, state), -1)
        output1 = F.relu(self.eigenfunction_l1(x))
        output2 = F.relu(self.eigenfunction_l2(output1))
        output3 = F.relu(self.eigenfunction_l3(output2))
        output4 = F.relu(self.eigenfunction_l4(output3))
        embeddings = self.eigenfunction_l5(output4)

        return embeddings

class Inv_Eigenfunction(nn.Module):
    def __init__(self, n_states: int=2,
                    eigenfunction_output_shape: int=2,
                    eigenfunction_hidden_shape1: int=48,
                    eigenfunction_hidden_shape2: int=96):
        super().__init__()

        self.inv_eigenfunction_l1 = nn.Linear(in_features=eigenfunction_output_shape, out_features=eigenfunction_hidden_shape1)
        self.inv_eigenfunction_l2 = nn.Linear(in_features=eigenfunction_hidden_shape1, out_features=eigenfunction_hidden_shape2)
        self.inv_eigenfunction_l3 = nn.Linear(in_features=eigenfunction_hidden_shape2, out_features=eigenfunction_hidden_shape2)
        self.inv_eigenfunction_l4 = nn.Linear(in_features=eigenfunction_hidden_shape2, out_features=eigenfunction_hidden_shape1)
        self.inv_eigenfunction_l5 = nn.Linear(in_features=eigenfunction_hidden_shape1, out_features=n_states)

    def forward(self, embeddings):
        ''' Purpose: Put system into state space coordinates from eigenfuction coordinates
        '''
        output1 = F.relu(self.inv_eigenfunction_l1(embeddings))
        output2 = F.relu(self.inv_eigenfunction_l2(output1))
        output3 = F.relu(self.inv_eigenfunction_l3(output2))
        output4 = F.relu(self.inv_eigenfunction_l4(output3))
        state_hat = self.inv_eigenfunction_l5(output4)

        return state_hat

class Spectrum(nn.Module):
    def __init__(self, spectrum_hidden_shape1: int=48,
                    spectrum_hidden_shape2: int=64,
                    spectrum_input_shape: int=2,
                    spectrum_output_shape: int=2):
        super().__init__()

        self.spectrum_l1 = nn.Linear(in_features=spectrum_input_shape, out_features=spectrum_hidden_shape1)
        self.spectrum_l2 = nn.Linear(in_features=spectrum_hidden_shape1, out_features=spectrum_hidden_shape2)
        self.spectrum_l3 = nn.Linear(in_features=spectrum_hidden_shape2, out_features=spectrum_hidden_shape1)
        self.spectrum_l4 = nn.Linear(in_features=spectrum_hidden_shape1, out_features=spectrum_output_shape)

    def forward(self, embeddings):
        ''' Purpose: predict eigenvalues from the output of an eigenfuction
        '''
        #x = torch.cat((encoded_context, embeddings), -1)
        output1 = F.relu(self.spectrum_l1(embeddings))
        output2 = F.relu(self.spectrum_l2(output1))
        output3 = F.relu(self.spectrum_l3(output2))
        eigs = self.spectrum_l4(output3)

        return eigs
