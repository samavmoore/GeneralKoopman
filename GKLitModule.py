import torch
from pytorch_lightning import LightningModule, Trainer
from GeneralKoopman.Nets import ContextDecoder, ContextEncoder, Eigenfunction, Inv_Eigenfunction, Spectrum
from GeneralKoopman.Loss import future_embedding_loss, future_state_loss, inf_loss, state_recon_loss, context_recon_loss, l2_reg
from GeneralKoopman.ConfigData import KoopmanDataModule
from GeneralKoopman.ExtraFunctions import K_blocks
from torchviz import make_dot


class GKM(LightningModule):
    def __init__(self, network_dict, learning_rate):
        super().__init__()

        self.context_encoder = ContextEncoder(network_dict)
        self.context_decoder = ContextDecoder(network_dict)

        self.eigenfunction = Eigenfunction(network_dict)
        self.inv_eigenfunction = Inv_Eigenfunction(network_dict)

        self.spectrum = Spectrum(network_dict)

        self.n_shifts = network_dict['n_shifts']
        self.delta_t = network_dict['delta_t']

        self.alpha_0 = network_dict['alpha_0']
        self.alpha_1 = network_dict['alpha_1']
        self.alpha_2 = network_dict['alpha_2']
        self.alpha_3 = network_dict['alpha_3']
        self.alpha_4 = network_dict['alpha_4']
        self.lam = network_dict['lambda']

        fut_state_mse = future_state_loss()
        fut_state_mse.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.future_state_loss = fut_state_mse

        fut_emb_mse = future_embedding_loss()
        fut_emb_mse.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.future_embedding_loss = fut_emb_mse

        state_recon_mse = state_recon_loss()
        state_recon_mse.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.state_recon_loss = state_recon_mse

        context_recon_mse = context_recon_loss()
        context_recon_mse.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.context_recon_loss = context_recon_mse

        inf_norm = inf_loss()
        inf_norm.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.inf_loss = inf_norm

        two_norm_reg = l2_reg()
        two_norm_reg.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.l2_reg = two_norm_reg

        self.n_states = network_dict["n_states"]
        
        self.encoded_context_shape = network_dict['encoded_context_shape']
        self.eigenfunction_output_shape = network_dict['eigenfunction_output_shape']

        self.learning_rate = learning_rate

    def forward(self, state, context, future):

        outputs = {}
        # encode context
        encoded_context= self.context_encoder(context)

        # reconstruct context
        estimated_context = self.context_decoder(encoded_context)
        outputs['estimated_context'] = estimated_context

        # embed the current state
        current_embeddings = self.eigenfunction(state, encoded_context)

        # reconstruct the current state
        estimated_current_state = self.inv_eigenfunction(current_embeddings, encoded_context)
        outputs['estimated_current_state'] = estimated_current_state

        future_embeddings = self.embed_true_future_states(future, encoded_context)
        outputs['future_embeddings'] = future_embeddings

        # estimate the spectrum with the current state
        omegas = self.spectrum(current_embeddings, encoded_context)
        
        # predict future states, and future embeddings
        outputs['estimated_future_state'], outputs['estimated_future_embeddings'] = \
            self.predict_the_future(current_embeddings, omegas, encoded_context)

        # return all of the relevant estimates to calculate loss 
        return outputs
        
    def training_step(self, batch, batch_idx):
        states, context, future = batch
        inputs = {'states': states, 'context': context, 'future': future}

        preds = self(*inputs.values())

        estimated_context = preds['estimated_context']
        estimated_current_state = preds['estimated_current_state']
        future_embeddings = preds['future_embeddings']
        estimated_future_state = preds['estimated_future_state']
        estimated_future_embeddings = preds['estimated_future_embeddings']

        weighted_future_state_loss = self.alpha_0*self.future_state_loss(future, estimated_future_state)
        weighted_state_recon_loss = self.alpha_1*self.state_recon_loss(states, estimated_current_state)
        weighted_context_recon_loss = self.alpha_2*self.context_recon_loss(context, estimated_context)
        weighted_future_embedding_loss = self.alpha_3*self.future_embedding_loss(future_embeddings, estimated_future_embeddings)
        weighted_inf_loss = self.alpha_4*self.inf_loss(states, estimated_current_state) 
        weighted_l2_reg = self.lam* self.l2_reg(self.named_parameters())
    
        loss = weighted_future_state_loss \
            + weighted_state_recon_loss \
            + weighted_context_recon_loss \
            + weighted_future_embedding_loss \
            + weighted_inf_loss \
            + weighted_l2_reg

        logs = {'future_state_loss': weighted_future_state_loss, 'state_recon_loss': weighted_state_recon_loss, \
                'context_recon_loss': weighted_context_recon_loss, 'future_embedding_loss' :weighted_future_embedding_loss, \
                  'inf_loss' : weighted_inf_loss, 'l2_reg': weighted_l2_reg, 'train_loss': loss}

        #loss = self.loss_fn(inputs, preds, self.named_parameters())
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        states, context, future = batch
        inputs = {'states': states.float(), 'context': context.float(), 'future': future.float()}

        preds = self(*inputs.values())

        estimated_context = preds['estimated_context']
        estimated_current_state = preds['estimated_current_state']
        future_embeddings = preds['future_embeddings']
        estimated_future_state = preds['estimated_future_state']
        estimated_future_embeddings = preds['estimated_future_embeddings']

        weighted_future_state_loss = self.alpha_0*self.future_state_loss(future, estimated_future_state)
        weighted_state_recon_loss = self.alpha_1*self.state_recon_loss(states, estimated_current_state)
        weighted_context_recon_loss = self.alpha_2*self.context_recon_loss(context, estimated_context)
        weighted_future_embedding_loss = self.alpha_3*self.future_embedding_loss(future_embeddings, estimated_future_embeddings)
        weighted_inf_loss = self.alpha_4*self.inf_loss(states, estimated_current_state) 
        weighted_l2_reg = self.lam* self.l2_reg(self.named_parameters())
    
        loss = weighted_future_state_loss \
            + weighted_state_recon_loss \
            + weighted_context_recon_loss \
            + weighted_future_embedding_loss \
            + weighted_inf_loss \
            + weighted_l2_reg

        logs = {'val_future_state_loss': weighted_future_state_loss, 'val_state_recon_loss': weighted_state_recon_loss, \
                'val_context_recon_loss': weighted_context_recon_loss, 'val_future_embedding_loss' :weighted_future_embedding_loss, \
                  'val_inf_loss' : weighted_inf_loss, 'val_l2_reg': weighted_l2_reg, 'val_loss': loss}

        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt
    
    def predict_the_future(self, current_embeddings, omegas, encoded_context, time_steps = None):
        ''' Purpose: Predict the future in eigenfunction coordinates and state space coordinates

            Inputs: current_embeddings - initial conditions in eigenfunction coordinates
                    omegas - initialization of the eigenvalues of our koopman operator
                    encoded_context - context representation for each obsevation in the batch

            Outputs: Estimated_future_states - tensor of shape (batch_size, self.n_shifts, self.n_states)
                     Estimated_future_embeddings - tensor of shape (batch_size, self.n_shifts, self.eigenfunction_output_shape)
        '''

        if not time_steps:
            time_steps = self.n_shifts

        batch_size = current_embeddings.size(0)

        # initialize future prediction tensors
        estimated_future_state = torch.zeros(batch_size, time_steps, self.n_states)
        estimated_future_state = estimated_future_state.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        estimated_future_embeddings = torch.zeros(batch_size, time_steps, self.eigenfunction_output_shape)
        estimated_future_embeddings = estimated_future_embeddings.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        for j, (intital_embeddings, initial_omegas, context_encodings) in enumerate(zip(current_embeddings, omegas, encoded_context)):
            temp_embeddings = intital_embeddings
            temp_omegas = initial_omegas

            for i in range(time_steps):
                # create blocks 
                blocks = K_blocks(temp_omegas, self.delta_t)

                # construct koopman operator and push to device
                koopman_operator = torch.block_diag(*blocks.values())
                koopman_operator = koopman_operator.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                # advance eigenfunction ahead one step
                temp_embeddings = torch.matmul(koopman_operator, temp_embeddings)

                # store in output tensor
                estimated_future_embeddings[j,i,:] = temp_embeddings

                # estimate state
                estimated_future_state[j,i,:] = self.inv_eigenfunction(temp_embeddings, context_encodings)

                # estimate spectrum
                temp_omegas = self.spectrum(temp_embeddings, context_encodings)


        return estimated_future_state, estimated_future_embeddings

    def embed_true_future_states(self, future, encoded_context):
        ''' Purpose: Put true future states in eigenfunction coordinates

            Inputs: future - batch of future states
                    encoded context - batch of context representations

            Output: future embeddings - batch of future embeddings
        '''
        batch_size = future.size(0)

        # initialize future embeddings and push to device
        future_embeddings = torch.zeros(batch_size, self.n_shifts, self.eigenfunction_output_shape)
        future_embeddings = future_embeddings.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # embed the future states
        for i, (ith_context, ith_future) in enumerate(zip(encoded_context, future)):
            # repeat context representation for each future state
            ith_context_repeated = ith_context.repeat(self.n_shifts, 1)

            # run eigenfunction network for each observation in the batch
            future_embeddings[i,:,:] = self.eigenfunction(ith_context_repeated, ith_future)

        return future_embeddings
