import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from Nets import ContextDecoder, ContextEncoder, Eigenfunction, Inv_Eigenfunction, Spectrum
from ExtraFunctions import K_blocks
############### ------------------------------ Context Pretraining Module ----------------------------------------------------------

class Context(LightningModule):

    def __init__(self, hid_layer_1_shape: int=64,
                       learning_rate: float=.01):
        super().__init__()
        self.save_hyperparameters()

        self.Encoder = ContextEncoder(hid_layer_1_shape=hid_layer_1_shape)
        self.Decoder = ContextDecoder(hid_layer_1_shape=hid_layer_1_shape)

        self.learning_rate = learning_rate

    def forward(self, raw_context):

        encoded_context = self.Encoder(raw_context)
        estimated_context = self.Decoder(encoded_context)

        return estimated_context

    def training_step(self, batch, batch_idx):
        inputs = batch
        preds = self(inputs)

        loss = F.mse_loss(inputs, preds)

        self.log('loss',loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        preds = self(inputs)

        val_loss = F.mse_loss(inputs, preds)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt


############### ------------------------------ Eigenfunction Pretraining Module ----------------------------------------------------------

class EigenPretrain(LightningModule):
    def __init__(self, hid_layer_shape1: int=48,
                       hid_layer_shape2: int=96,
                       learning_rate: float=0.0047,
                       Context_NN_Path: str='my/path'):
        super().__init__()
        self.save_hyperparameters()

        self.Context_NN = Context.load_from_checkpoint(Context_NN_Path)
        self.Context_NN.freeze()

        self.Eigenfunction = Eigenfunction(eigenfunction_hidden_shape1=hid_layer_shape1, eigenfunction_hidden_shape2=hid_layer_shape1)
        self.Inv_Eigenfunction = Inv_Eigenfunction(eigenfunction_hidden_shape1=hid_layer_shape1, eigenfunction_hidden_shape2=hid_layer_shape2)

        self.learning_rate = learning_rate

    def forward(self, state, raw_context):

        context_rep = self.Context_NN.Encoder(raw_context)

        embedded_state = self.Eigenfunction(state, context_rep)
        recon_state = self.Inv_Eigenfunction(embedded_state)

        return recon_state
    
    def training_step(self, batch, batch_idx):
        states, raw_context = batch
        preds = self(states, raw_context)

        loss = F.mse_loss(states, preds)

        self.log('loss',loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        states, raw_context = batch
        preds = self(states, raw_context)

        val_loss = F.mse_loss(states, preds)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt

############### ------------------------------ Main Lightning Module ----------------------------------------------------------
    
class PendulumKoopModule(LightningModule):
    def __init__(self, n_states: int=2,
                    n_shifts: int =32,
                    delta_t: float=.02,
                    Context_NN_Path: str='my/path',
                    Eigenfunction_NN_Path: str='my/path',
                    eigenfunction_output_shape: int=2,
                    spectrum_hid_shape1: float=48,
                    spectrum_hid_shape2: float=64,
                    learning_rate: float=.0001,
                    loss_hyper_a0: float=.1,
                    loss_hypers_a1: float=.1,
                    loss_hyper_a2: float=1,
                    loss_hyper_a3: float=0,
                    loss_hyper_lambda: float=0):
        super().__init__()
        self.save_hyperparameters()

        self.Context_NN = Context.load_from_checkpoint(Context_NN_Path)
        self.Context_NN.freeze()

        self.Eigenfunction_NN = EigenPretrain.load_from_checkpoint(Eigenfunction_NN_Path)

        self.Spectrum = Spectrum(spectrum_hidden_shape1=spectrum_hid_shape1, spectrum_hidden_shape2=spectrum_hid_shape2)

        self.n_shifts = n_shifts
        self.delta_t = delta_t
        self.n_states = n_states
        self.eigenfunction_output_shape = eigenfunction_output_shape

        self.alpha_0 = loss_hyper_a0
        self.alpha_1 = loss_hypers_a1
        self.alpha_2 = loss_hyper_a2
        self.alpha_3 = loss_hyper_a3
        self.lam = loss_hyper_lambda

        self.learning_rate = learning_rate

    def forward(self, state, context, future):

        outputs = {}
        # encode context
        encoded_context = self.Context_NN.Encoder(context)

        # reconstruct context
        estimated_context = self.Context_NN.Decoder(encoded_context)
        outputs['estimated_context'] = estimated_context

        # embed the current state
        current_embeddings = self.Eigenfunction_NN.Eigenfunction(state, encoded_context)

        # reconstruct the current state
        estimated_current_state = self.Eigenfunction_NN.Inv_Eigenfunction(current_embeddings, encoded_context)
        outputs['estimated_current_state'] = estimated_current_state

        future_embeddings = self.embed_true_future_states(future, encoded_context)
        outputs['future_embeddings'] = future_embeddings

        # estimate the spectrum with the current state
        omegas = self.Spectrum(current_embeddings)
        
        # predict future states, and future embeddings
        outputs['estimated_future_state'], outputs['estimated_future_embeddings'] = \
            self.predict_the_future(current_embeddings, omegas, encoded_context)

        # return all of the relevant estimates to calculate loss 
        return outputs
        
    def training_step(self, batch, batch_idx):
        states, context, future = batch
        inputs = {'states': states, 'context': context, 'future': future}

        preds = self(*inputs.values())

        estimated_current_state = preds['estimated_current_state']
        future_embeddings = preds['future_embeddings']
        estimated_future_state = preds['estimated_future_state']
        estimated_future_embeddings = preds['estimated_future_embeddings']

        weighted_future_state_loss = self.alpha_0*F.mse_loss(future, estimated_future_state)
        weighted_state_recon_loss = self.alpha_1*F.mse_loss(states, estimated_current_state)
        weighted_future_embedding_loss = self.alpha_2*F.mse_loss(future_embeddings, estimated_future_embeddings)
        weighted_inf_loss = self.alpha_3*self.inf_loss(states, estimated_current_state) 
        weighted_l2_reg = self.lam*self.l2_reg(self.named_parameters())
        
        loss = weighted_future_state_loss \
            + weighted_state_recon_loss \
            + weighted_future_embedding_loss \
            + weighted_inf_loss \
            + weighted_l2_reg

        logs = {'future_state_loss': weighted_future_state_loss, 'state_recon_loss': weighted_state_recon_loss, \
                'future_embedding_loss' :weighted_future_embedding_loss, \
                  'inf_loss' : weighted_inf_loss, 'l2_reg': weighted_l2_reg, 'train_loss': loss}

        #loss = self.loss_fn(inputs, preds, self.named_parameters())
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        states, context, future = batch
        inputs = {'states': states.float(), 'context': context.float(), 'future': future.float()}

        preds = self(*inputs.values())

        estimated_current_state = preds['estimated_current_state']
        future_embeddings = preds['future_embeddings']
        estimated_future_state = preds['estimated_future_state']
        estimated_future_embeddings = preds['estimated_future_embeddings']

        weighted_future_state_loss = self.alpha_0*F.mse_loss(future, estimated_future_state)
        weighted_state_recon_loss = self.alpha_1*F.mse_loss(states, estimated_current_state)
        weighted_future_embedding_loss = self.alpha_2*F.mse_loss(future_embeddings, estimated_future_embeddings)
        weighted_inf_loss = self.alpha_3*self.inf_loss(states, estimated_current_state) 
        weighted_l2_reg = self.lam*self.l2_reg(self.named_parameters())
    
        loss = weighted_future_state_loss \
            + weighted_state_recon_loss \
            + weighted_future_embedding_loss \
            + weighted_inf_loss \
            + weighted_l2_reg

        logs = {'val_future_state_loss': weighted_future_state_loss, 'val_state_recon_loss': weighted_state_recon_loss, \
                'val_future_embedding_loss' :weighted_future_embedding_loss, \
                  'val_inf_loss' : weighted_inf_loss, 'val_l2_reg': weighted_l2_reg, 'val_loss': loss}

        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    ## define l2 regularization on the weights
    def l2_reg(self, params):
        sum = 0
        for name, param in params:
            if "weight" in name and param.requires_grad:
                sum += torch.linalg.norm(param)**2
        return sum
    
    ## define infinity loss
    def inf_loss(self, states, estimated_state):
        norm = torch.linalg.norm(states-estimated_state, float('inf'))
        return norm

    # configure optimizers
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
                estimated_future_state[j,i,:] = self.Eigenfunction_NN.Inv_Eigenfunction(temp_embeddings, context_encodings)

                # estimate spectrum
                temp_omegas = self.Spectrum(temp_embeddings)


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
            future_embeddings[i,:,:] = self.Eigenfunction_NN.Eigenfunction(ith_context_repeated, ith_future)

        return future_embeddings
