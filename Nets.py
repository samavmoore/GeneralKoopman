import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(self, network_dict):
        super().__init__()

        raw_context_shape = network_dict['n_context_params']
        context_hidden_shape = 48
        self.encoded_context_shape = network_dict['encoded_context_shape']

        self.context_encoder_l1 = nn.Linear(in_features=raw_context_shape, out_features=context_hidden_shape)
        self.context_encoder_l2 = nn.Linear(in_features=context_hidden_shape, out_features=context_hidden_shape)
        self.context_encoder_l3 = nn.Linear(in_features=context_hidden_shape, out_features=self.encoded_context_shape)

    def forward(self, raw_context):
        ''' Purpose: Embed raw context in a latent space
        '''
        output1 = F.relu(self.context_encoder_l1(raw_context))
        output2 = F.relu(self.context_encoder_l2(output1))
        encoded_context = self.context_encoder_l3(output2)

        return encoded_context

class ContextDecoder(nn.Module):
    def __init__(self, network_dict):
        super().__init__()

        raw_context_shape = network_dict['n_context_params']
        context_hidden_shape = 48
        self.encoded_context_shape = network_dict['encoded_context_shape']

        self.context_decoder_l1 = nn.Linear(in_features=self.encoded_context_shape, out_features=context_hidden_shape)
        self.context_decoder_l2 = nn.Linear(in_features=context_hidden_shape, out_features=context_hidden_shape)
        self.context_decoder_l3 = nn.Linear(in_features=context_hidden_shape, out_features=raw_context_shape)

    def forward(self, encoded_context):
        ''' Purpose: Reconstruct context from a latent representation
        '''
        output1 = F.relu(self.context_decoder_l1(encoded_context))
        output2 = F.relu(self.context_decoder_l2(output1))
        context_hat = self.context_decoder_l3(output2)

        return context_hat

class Eigenfunction(nn.Module):
    def __init__(self, network_dict):
        super().__init__()

        self.encoded_context_shape = network_dict['encoded_context_shape']
        self.n_states = network_dict["n_states"]
        eigenfunction_input_shape = self.n_states + self.encoded_context_shape
        eigenfunction_hidden_shape1 = 64
        eigenfunction_hidden_shape2 = 64
        eigenfunction_hidden_shape3 = 64
        self.eigenfunction_output_shape = network_dict['eigenfunction_output_shape']

        self.eigenfunction_l1 = nn.Linear(in_features=eigenfunction_input_shape, out_features=eigenfunction_hidden_shape1)
        self.eigenfunction_l2 = nn.Linear(in_features=eigenfunction_hidden_shape1, out_features=eigenfunction_hidden_shape2)
        self.eigenfunction_l3 = nn.Linear(in_features=eigenfunction_hidden_shape2, out_features=eigenfunction_hidden_shape3)
        self.eigenfunction_l4 = nn.Linear(in_features=eigenfunction_hidden_shape3, out_features=eigenfunction_hidden_shape3)
        self.eigenfunction_l5 = nn.Linear(in_features=eigenfunction_hidden_shape3, out_features=self.eigenfunction_output_shape)

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
    def __init__(self, network_dict):
        super().__init__()

        self.encoded_context_shape = network_dict['encoded_context_shape']
        self.n_states = network_dict["n_states"]
        eigenfunction_hidden_shape1 = 64
        eigenfunction_hidden_shape2 = 64
        eigenfunction_hidden_shape3 = 64
        self.eigenfunction_output_shape = network_dict['eigenfunction_output_shape']
        self.inv_eigen_input_shape =  self.eigenfunction_output_shape #+ self.encoded_context_shape 


        self.inv_eigenfunction_l1 = nn.Linear(in_features=self.inv_eigen_input_shape, out_features=eigenfunction_hidden_shape3)
        self.inv_eigenfunction_l2 = nn.Linear(in_features=eigenfunction_hidden_shape3, out_features=eigenfunction_hidden_shape3)
        self.inv_eigenfunction_l3 = nn.Linear(in_features=eigenfunction_hidden_shape3, out_features=eigenfunction_hidden_shape2)
        self.inv_eigenfunction_l4 = nn.Linear(in_features=eigenfunction_hidden_shape2, out_features=eigenfunction_hidden_shape1)
        self.inv_eigenfunction_l5 = nn.Linear(in_features=eigenfunction_hidden_shape1, out_features=self.n_states)

    def forward(self, embeddings, encoded_context=None):
        ''' Purpose: Put system into state space coordinates from eigenfuction coordinates
        '''
        #x = torch.cat((encoded_context, embeddings), -1)
        x = embeddings
        output1 = F.relu(self.inv_eigenfunction_l1(x))
        output2 = F.relu(self.inv_eigenfunction_l2(output1))
        output3 = F.relu(self.inv_eigenfunction_l3(output2))
        output4 = F.relu(self.inv_eigenfunction_l4(output3))
        state_hat = self.inv_eigenfunction_l5(output4)

        return state_hat

class Spectrum(nn.Module):
    def __init__(self, network_dict):
        super().__init__()

        self.eigenfunction_output_shape = network_dict['eigenfunction_output_shape']
        spectrum_input_shape = self.eigenfunction_output_shape # + network_dict['encoded_context_shape']
        spectrum_hidden_shape = 48
        spectrum_output_shape = self.eigenfunction_output_shape

        self.spectrum_l1 = nn.Linear(in_features=spectrum_input_shape, out_features=spectrum_hidden_shape)
        self.spectrum_l2 = nn.Linear(in_features=spectrum_hidden_shape, out_features=spectrum_hidden_shape)
        self.spectrum_l3 = nn.Linear(in_features=spectrum_hidden_shape, out_features=spectrum_hidden_shape)
        self.spectrum_l4 = nn.Linear(in_features=spectrum_hidden_shape, out_features=spectrum_output_shape)

    def forward(self, embeddings, encoded_context=None):
        ''' Purpose: predict eigenvalues from the output of an eigenfuction
        '''
        #x = torch.cat((encoded_context, embeddings), -1)
        x = embeddings
        output1 = F.relu(self.spectrum_l1(x))
        output2 = F.relu(self.spectrum_l2(output1))
        output3 = F.relu(self.spectrum_l3(output2))
        eigs = self.spectrum_l4(output3)

        return eigs
