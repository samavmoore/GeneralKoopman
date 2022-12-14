import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from PretrainingLitModules import EigenPretrain
from torch.optim.lr_scheduler import CyclicLR
from Nets import Spectrum
from ExtraFunctions import K_blocks

######################## ------------------------------ Main Lightning Module ------------------------------############################

class PendulumKoopModule(LightningModule):
    def __init__(self, n_states: int=2,
                    n_shifts: int =32,
                    delta_t: float=.02,
                    skip_connections: bool=False,
                    Eigenfunction_NN_Path: str='my/path',
                    eigenfunction_output_shape: int=2,
                    spectrum_hid_shape1: float=64,
                    spectrum_hid_shape2: float=128,
                    learning_rate: float=.0001,
                    gradual_unfreeze=False,
                    loss_hyper_schedule=False,
                    # epoch specific loss hyper parameters
                    loss_hyper_a0_e0: float=.1,
                    loss_hypers_a1_e0: float=.1,
                    loss_hyper_a2_e0: float=1,

                    loss_hyper_a0_e1: float=.1,
                    loss_hypers_a1_e1: float=.1,
                    loss_hyper_a2_e1: float=1,

                    loss_hyper_a0_e2: float=.1,
                    loss_hypers_a1_e2: float=.1,
                    loss_hyper_a2_e2: float=1,

                    loss_hyper_a0_e3: float=.1,
                    loss_hypers_a1_e3: float=.1,
                    loss_hyper_a2_e3: float=1,

                    loss_hyper_a0_e4: float=.1,
                    loss_hypers_a1_e4: float=.1,
                    loss_hyper_a2_e4: float=1,

                    loss_hyper_a3: float=0.01,
                    loss_hyper_lambda: float=0):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.skipped = skip_connections
        self.gradual_unfreeze = gradual_unfreeze
        self.loss_hyper_schedule = loss_hyper_schedule

        if self.skipped:
            spectrum_input_shape = 3
        else:
            spectrum_input_shape = 2
     
        pretrained_NN = EigenPretrain.load_from_checkpoint(Eigenfunction_NN_Path)

        self.context_encoder = pretrained_NN.encoder
        self.context_decoder = pretrained_NN.decoder

        self.eigenfunction = pretrained_NN.eigenfunction
        self.inv_eigenfunction = pretrained_NN.inv_eigenfunction

        if gradual_unfreeze:
            self.eigenfunction.requires_grad_(False)
            self.inv_eigenfunction.requires_grad_(False)

        self.spectrum = Spectrum(spectrum_input_shape=spectrum_input_shape, spectrum_hidden_shape1=spectrum_hid_shape1, spectrum_hidden_shape2=spectrum_hid_shape2)

        self.n_shifts = n_shifts
        self.delta_t = delta_t
        self.n_states = n_states
        self.eigenfunction_output_shape = eigenfunction_output_shape

        self.alpha_0_e0 = loss_hyper_a0_e0
        self.alpha_1_e0 = loss_hypers_a1_e0
        self.alpha_2_e0 = loss_hyper_a2_e0

        self.alpha_0_e1 = loss_hyper_a0_e1
        self.alpha_1_e1 = loss_hypers_a1_e1
        self.alpha_2_e1 = loss_hyper_a2_e1

        self.alpha_0_e2 = loss_hyper_a0_e2
        self.alpha_1_e2 = loss_hypers_a1_e2
        self.alpha_2_e2 = loss_hyper_a2_e2

        self.alpha_0_e3 = loss_hyper_a0_e3
        self.alpha_1_e3 = loss_hypers_a1_e3
        self.alpha_2_e3 = loss_hyper_a2_e3

        self.alpha_0_e4 = loss_hyper_a0_e4
        self.alpha_1_e4 = loss_hypers_a1_e4
        self.alpha_2_e4 = loss_hyper_a2_e4

        self.alpha_3 = loss_hyper_a3
        self.lam = loss_hyper_lambda

        self.learning_rate = learning_rate

    def forward(self, state, context, future):

        outputs = {}
        # encode context
        encoded_context = self.context_encoder(context)

        # reconstruct context
        estimated_context = self.context_decoder(encoded_context)
        outputs['estimated_context'] = estimated_context

        # embed the current state
        current_embeddings = self.eigenfunction(state, encoded_context)

        if self.skipped:
            inv_inputs = torch.cat((encoded_context, current_embeddings), -1)
        else:
            inv_inputs = current_embeddings

        # reconstruct the current state
        estimated_current_state = self.inv_eigenfunction(inv_inputs)
        outputs['estimated_current_state'] = estimated_current_state

        future_embeddings = self.embed_true_future_states(future, encoded_context)
        outputs['future_embeddings'] = future_embeddings

        # estimate the spectrum with the current state
        omegas = self.spectrum(inv_inputs)
        
        # predict future states, and future embeddings
        outputs['estimated_future_state'], outputs['estimated_future_embeddings'] = self.predict_the_future(current_embeddings, omegas, encoded_context)

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

        #for name, param in self.named_parameters():
        #    print(name, param, param.grad)

        logs = {}

        logs['future_state_loss'] = F.mse_loss(future, estimated_future_state)
        logs['state_recon_loss'] = F.mse_loss(states, estimated_current_state)
        logs['future_embedding_loss'] = F.mse_loss(future_embeddings, estimated_future_embeddings)
        logs['inf_loss'] = (self.inf_loss(future[:,:2,:], estimated_future_state[:,:2,:]) + self.inf_loss(future_embeddings[:,:2,:], estimated_future_embeddings[:,:2,:]))
        logs['l2_reg'] = self.l2_reg(self.named_parameters())


        loss = self.calculate_loss(logs['future_state_loss'], logs['state_recon_loss'], logs['future_embedding_loss'], logs['inf_loss'], logs['l2_reg'])

        logs['train_loss'] = loss

        self.step_optimizers(loss)

        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def validation_step(self, batch, batch_idx):
        states, context, future = batch
        inputs = {'states': states, 'context': context, 'future': future}

        preds = self(*inputs.values())

        estimated_current_state = preds['estimated_current_state']
        future_embeddings = preds['future_embeddings']
        estimated_future_state = preds['estimated_future_state']
        estimated_future_embeddings = preds['estimated_future_embeddings']

        future_state_loss = F.mse_loss(future, estimated_future_state)
        state_recon_loss = F.mse_loss(states, estimated_current_state)
        future_embedding_loss = F.mse_loss(future_embeddings, estimated_future_embeddings)
        inf_loss = (self.inf_loss(future[:,:2,:], estimated_future_state[:,:2,:]) + self.inf_loss(future_embeddings[:,:2,:], estimated_future_embeddings[:,:2,:]))
        l2_reg = self.l2_reg(self.named_parameters())

        loss = future_state_loss + state_recon_loss + future_embedding_loss

        logs = {'val_future_state_loss': future_state_loss, 'val_state_recon_loss': state_recon_loss, \
                'val_future_embedding_loss' : future_embedding_loss, \
                  'val_inf_loss' : inf_loss, 'val_l2_reg': l2_reg, 'val_loss': loss}

        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    ## define l2 regularization on the weights
    def l2_reg(self, params):
        sum = 0
        for name, param in params:
            if "weight" in name and param.requires_grad:
                sum += torch.linalg.norm(param)**2
        return sum
    
    ## define infinity loss
    def inf_loss(self, inputs, preds):
        error = inputs-preds
        error = error.flatten()
        norm = torch.linalg.norm(error, float('inf'))
        return norm
    
    def calculate_loss(self, future_state_loss, state_recon_loss, future_embedding_loss, inf_loss, l2_reg):
        if self.loss_hyper_schedule:
            if self.current_epoch >= 4:
                weighted_future_state_loss = self.alpha_0_e4*future_state_loss
                weighted_state_recon_loss = self.alpha_1_e4*state_recon_loss
                weighted_future_embedding_loss = self.alpha_2_e4*future_embedding_loss
                weighted_inf_loss = self.alpha_3*inf_loss
                weighted_l2_reg = self.lam*l2_reg

            elif self.current_epoch ==3:
                weighted_future_state_loss = self.alpha_0_e3*future_state_loss
                weighted_state_recon_loss = self.alpha_1_e3*state_recon_loss
                weighted_future_embedding_loss = self.alpha_2_e3*future_embedding_loss
                weighted_inf_loss = self.alpha_3*inf_loss
                weighted_l2_reg = self.lam*l2_reg
            
            elif self.current_epoch == 2:
                weighted_future_state_loss = self.alpha_0_e2*future_state_loss
                weighted_state_recon_loss = self.alpha_1_e2*state_recon_loss
                weighted_future_embedding_loss = self.alpha_2_e2*future_embedding_loss
                weighted_inf_loss = self.alpha_3*inf_loss
                weighted_l2_reg = self.lam*l2_reg
            
            elif self.current_epoch ==1:
                weighted_future_state_loss = self.alpha_0_e1*future_state_loss
                weighted_state_recon_loss = self.alpha_1_e1*state_recon_loss
                weighted_future_embedding_loss = self.alpha_2_e1*future_embedding_loss
                weighted_inf_loss = self.alpha_3*inf_loss
                weighted_l2_reg = self.lam*l2_reg
            
            else:
                weighted_future_state_loss = self.alpha_0_e0*future_state_loss
                weighted_state_recon_loss = self.alpha_1_e0*state_recon_loss
                weighted_future_embedding_loss = self.alpha_2_e0*future_embedding_loss
                weighted_inf_loss = self.alpha_3*inf_loss
                weighted_l2_reg = self.lam*l2_reg
        else:
                weighted_future_state_loss = self.alpha_0_e0*future_state_loss
                weighted_state_recon_loss = self.alpha_1_e0*state_recon_loss
                weighted_future_embedding_loss = self.alpha_2_e0*future_embedding_loss
                weighted_inf_loss = self.alpha_3*inf_loss
                weighted_l2_reg = self.lam*l2_reg

        
        loss = weighted_future_state_loss \
                + weighted_state_recon_loss \
                + weighted_future_embedding_loss \
                + weighted_inf_loss \
                + weighted_l2_reg

        return loss

    # step layer specific optimizers by epoch for gradual unfreezing
    def step_optimizers(self, loss):
        opt_spect, opt_eigen_l7, opt_eigen_l6, opt_eigen_l5, opt_eigen_l4, opt_eigen_l3, opt_eigen_l2, opt_eigen_l1 = self.optimizers()
        lr_sched_spect, lr_sched_eigen_l7, lr_sched_eigen_l6, lr_sched_eigen_l5, lr_sched_eigen_l4, lr_sched_eigen_l3, lr_sched_eigen_l2, lr_sched_eigen_l1 = self.lr_schedulers()

        if self.gradual_unfreeze:
            if self.current_epoch ==5:
                opt_spect.zero_grad()
                opt_eigen_l7.zero_grad()
                opt_eigen_l6.zero_grad()
                opt_eigen_l5.zero_grad()
                opt_eigen_l4.zero_grad()
                opt_eigen_l3.zero_grad()
                opt_eigen_l2.zero_grad()
                opt_eigen_l1.zero_grad()

                self.manual_backward(loss)

                torch.nn.utils.clip_grad_norm_(self.parameters(), 8, norm_type=2.0, error_if_nonfinite=True)
                opt_spect.step()
                opt_eigen_l7.step()
                opt_eigen_l6.step()
                opt_eigen_l5.step()
                opt_eigen_l4.step()
                opt_eigen_l3.step()
                opt_eigen_l2.step()
                opt_eigen_l1.step()

                lr_sched_spect.step()
                lr_sched_eigen_l7.step()
                lr_sched_eigen_l6.step()
                lr_sched_eigen_l5.step()
                lr_sched_eigen_l4.step()
                lr_sched_eigen_l3.step()
                lr_sched_eigen_l2.step()
                lr_sched_eigen_l1.step()

            elif self.current_epoch ==5:
                opt_spect.zero_grad()
                opt_eigen_l7.zero_grad()
                opt_eigen_l6.zero_grad()
                opt_eigen_l5.zero_grad()
                opt_eigen_l4.zero_grad()
                opt_eigen_l3.zero_grad()
                opt_eigen_l2.zero_grad()

                self.manual_backward(loss)

                torch.nn.utils.clip_grad_norm_(self.parameters(), 8, norm_type=2.0, error_if_nonfinite=True)
                opt_spect.step()
                opt_eigen_l7.step()
                opt_eigen_l6.step()
                opt_eigen_l5.step()
                opt_eigen_l4.step()
                opt_eigen_l3.step()
                opt_eigen_l2.step()

                lr_sched_spect.step()
                lr_sched_eigen_l7.step()
                lr_sched_eigen_l6.step()
                lr_sched_eigen_l5.step()
                lr_sched_eigen_l4.step()
                lr_sched_eigen_l3.step()
                lr_sched_eigen_l2.step()

            if self.current_epoch ==4:
                opt_spect.zero_grad()
                opt_eigen_l7.zero_grad()
                opt_eigen_l6.zero_grad()
                opt_eigen_l5.zero_grad()
                opt_eigen_l4.zero_grad()
                opt_eigen_l3.zero_grad()

                self.manual_backward(loss)

                torch.nn.utils.clip_grad_norm_(self.parameters(), 8, norm_type=2.0, error_if_nonfinite=True)
                opt_spect.step()
                opt_eigen_l7.step()
                opt_eigen_l6.step()
                opt_eigen_l5.step()
                opt_eigen_l4.step()
                opt_eigen_l3.step()

                lr_sched_spect.step()
                lr_sched_eigen_l7.step()
                lr_sched_eigen_l6.step()
                lr_sched_eigen_l5.step()
                lr_sched_eigen_l4.step()
                lr_sched_eigen_l3.step()

            elif self.current_epoch ==3:
                opt_spect.zero_grad()
                opt_eigen_l7.zero_grad()
                opt_eigen_l6.zero_grad()
                opt_eigen_l5.zero_grad()
                opt_eigen_l4.zero_grad()

                self.manual_backward(loss)

                torch.nn.utils.clip_grad_norm_(self.parameters(), 8, norm_type=2.0, error_if_nonfinite=True)
                opt_spect.step()
                opt_eigen_l7.step()
                opt_eigen_l6.step()
                opt_eigen_l5.step()
                opt_eigen_l4.step()

                lr_sched_spect.step()
                lr_sched_eigen_l7.step()
                lr_sched_eigen_l6.step()
                lr_sched_eigen_l5.step()
                lr_sched_eigen_l4.step()

            elif self.current_epoch ==2:
                opt_spect.zero_grad()
                opt_eigen_l7.zero_grad()
                opt_eigen_l6.zero_grad()
                opt_eigen_l5.zero_grad()

                self.manual_backward(loss)

                torch.nn.utils.clip_grad_norm_(self.parameters(), 8, norm_type=2.0, error_if_nonfinite=True)
                opt_spect.step()
                opt_eigen_l7.step()
                opt_eigen_l6.step()
                opt_eigen_l5.step()

                lr_sched_spect.step()
                lr_sched_eigen_l7.step()
                lr_sched_eigen_l6.step()
                lr_sched_eigen_l5.step()

            elif self.current_epoch ==1:
                opt_spect.zero_grad()
                opt_eigen_l7.zero_grad()
                opt_eigen_l6.zero_grad()

                self.manual_backward(loss)

                torch.nn.utils.clip_grad_norm_(self.parameters(), 8, norm_type=2.0, error_if_nonfinite=True)
                opt_spect.step()
                opt_eigen_l7.step()
                opt_eigen_l6.step()

                lr_sched_spect.step()
                lr_sched_eigen_l7.step()
                lr_sched_eigen_l6.step()

            else:
                opt_spect.zero_grad()
                opt_eigen_l7.zero_grad()

                self.manual_backward(loss)

                torch.nn.utils.clip_grad_norm_(self.parameters(), 8, norm_type=2.0, error_if_nonfinite=True)
                opt_spect.step()
                opt_eigen_l7.step()

                lr_sched_spect.step()
                lr_sched_eigen_l7.step()
        else:
                opt_spect.zero_grad()
                opt_eigen_l7.zero_grad()
                opt_eigen_l6.zero_grad()
                opt_eigen_l5.zero_grad()
                opt_eigen_l4.zero_grad()
                opt_eigen_l3.zero_grad()
                opt_eigen_l2.zero_grad()
                opt_eigen_l1.zero_grad()

                self.manual_backward(loss)

                torch.nn.utils.clip_grad_norm_(self.parameters(), 8, norm_type=2.0, error_if_nonfinite=True)
                opt_spect.step()
                opt_eigen_l7.step()
                opt_eigen_l6.step()
                opt_eigen_l5.step()
                opt_eigen_l4.step()
                opt_eigen_l3.step()
                opt_eigen_l2.step()
                opt_eigen_l1.step()

                lr_sched_spect.step()
                lr_sched_eigen_l7.step()
                lr_sched_eigen_l6.step()
                lr_sched_eigen_l5.step()
                lr_sched_eigen_l4.step()
                lr_sched_eigen_l3.step()
                lr_sched_eigen_l2.step()
                lr_sched_eigen_l1.step()


    # configure optimizers for gradual unfreezing
    def configure_optimizers(self):
        l7_eigen_layers ={'params': self.eigenfunction.eigenfunction_l7.parameters(), 'params': self.inv_eigenfunction.inv_eigenfunction_l7.parameters()}
        l6_eigen_layers ={'params': self.eigenfunction.eigenfunction_l6.parameters(), 'params': self.inv_eigenfunction.inv_eigenfunction_l6.parameters()}
        l5_eigen_layers = {'params': self.eigenfunction.eigenfunction_l5.parameters(), 'params': self.inv_eigenfunction.inv_eigenfunction_l5.parameters()}
        l4_eigen_layers = {'params':self.eigenfunction.eigenfunction_l4.parameters(),'params': self.inv_eigenfunction.inv_eigenfunction_l4.parameters()}
        l3_eigen_layers = {'params': self.eigenfunction.eigenfunction_l3.parameters(), 'params': self.inv_eigenfunction.inv_eigenfunction_l3.parameters()}
        l2_eigen_layers = {'params':self.eigenfunction.eigenfunction_l2.parameters(), 'params': self.inv_eigenfunction.inv_eigenfunction_l2.parameters()}
        l1_eigen_layers = {'params':self.eigenfunction.eigenfunction_l1.parameters(), 'params': self.inv_eigenfunction.inv_eigenfunction_l1.parameters()}
        

        opt_spect = torch.optim.SGD(self.spectrum.parameters(), lr=.0001)
        opt_eigen_l7 = torch.optim.SGD([l7_eigen_layers], lr=.0001, momentum=.9)
        opt_eigen_l6 = torch.optim.SGD([l6_eigen_layers], lr=.0001, momentum=.9)
        opt_eigen_l5 = torch.optim.SGD([l5_eigen_layers], lr=.0001, momentum=.9)
        opt_eigen_l4 = torch.optim.SGD([l4_eigen_layers], lr=.0001, momentum=.9)
        opt_eigen_l3 = torch.optim.SGD([l3_eigen_layers], lr=.0001, momentum=.9)
        opt_eigen_l2 = torch.optim.SGD([l2_eigen_layers], lr=.0001, momentum=.9)
        opt_eigen_l1 = torch.optim.SGD([l1_eigen_layers], lr=.0001, momentum=.9)
        

        lr_sched_spect = {'scheduler': CyclicLR(opt_spect, base_lr=1e-2, max_lr=1e-1, step_size_up=2874, base_momentum=.8, max_momentum=.9, mode='triangular2')}
        lr_sched_eigen_l7 = {'scheduler': CyclicLR(opt_eigen_l7, base_lr=1e-2, max_lr=1e-1, step_size_up=2874, base_momentum=.8, max_momentum=.9, mode='triangular2')}
        lr_sched_eigen_l6 = {'scheduler': CyclicLR(opt_eigen_l6, base_lr=1e-2, max_lr=1e-1, step_size_up=2874, base_momentum=.8, max_momentum=.9, mode='triangular2')}
        lr_sched_eigen_l5 = {'scheduler': CyclicLR(opt_eigen_l5, base_lr=1e-3, max_lr=1e-2, step_size_up=2874, base_momentum=.8, max_momentum=.9, mode='triangular2')}
        lr_sched_eigen_l4 = {'scheduler': CyclicLR(opt_eigen_l4, base_lr=1e-3, max_lr=1e-2, step_size_up=2874, base_momentum=.8, max_momentum=.9, mode='triangular2')}
        lr_sched_eigen_l3 = {'scheduler': CyclicLR(opt_eigen_l3, base_lr=1e-4, max_lr=1e-3, step_size_up=2874, base_momentum=.8, max_momentum=.9, mode='triangular2')}
        lr_sched_eigen_l2 = {'scheduler': CyclicLR(opt_eigen_l2, base_lr=1e-4, max_lr=1e-3, step_size_up=2874, base_momentum=.8, max_momentum=.9, mode='triangular2')}
        lr_sched_eigen_l1 = {'scheduler': CyclicLR(opt_eigen_l1, base_lr=1e-4, max_lr=1e-3, step_size_up=2874, base_momentum=.8, max_momentum=.9, mode='triangular2')}

        return [opt_spect, opt_eigen_l7, opt_eigen_l6, opt_eigen_l5, opt_eigen_l4, opt_eigen_l3, opt_eigen_l2, opt_eigen_l1], [lr_sched_spect, lr_sched_eigen_l7, lr_sched_eigen_l6, lr_sched_eigen_l5, lr_sched_eigen_l4, lr_sched_eigen_l3, lr_sched_eigen_l2, lr_sched_eigen_l1]

    # gradual unfreezing
    def on_train_epoch_start(self):
        if self.gradual_unfreeze:
            if self.current_epoch==0:
                self.eigenfunction.eigenfunction_l7.requires_grad_(True)
                self.inv_eigenfunction.inv_eigenfunction_l7.requires_grad_(True)

            elif self.current_epoch==1:
                self.eigenfunction.eigenfunction_l6.requires_grad_(True)
                self.inv_eigenfunction.inv_eigenfunction_l6.requires_grad_(True)
            
            elif self.current_epoch==2:
                self.eigenfunction.eigenfunction_l5.requires_grad_(True)
                self.inv_eigenfunction.inv_eigenfunction_l5.requires_grad_(True)
            
            elif self.current_epoch==3:
                self.eigenfunction.eigenfunction_l4.requires_grad_(True)
                self.inv_eigenfunction.inv_eigenfunction_l4.requires_grad_(True)
            
            elif self.current_epoch==4:
                self.eigenfunction.eigenfunction_l3.requires_grad_(True)
                self.inv_eigenfunction.inv_eigenfunction_l3.requires_grad_(True)

            elif self.current_epoch==5:
                self.eigenfunction.eigenfunction_l2.requires_grad_(True)
                self.inv_eigenfunction.inv_eigenfunction_l2.requires_grad_(True)

            elif self.current_epoch==6:
                self.eigenfunction.eigenfunction_l1.requires_grad_(True)
                self.inv_eigenfunction.inv_eigenfunction_l1.requires_grad_(True)
    
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

        for j, (initial_embeddings, initial_omegas, context_encodings) in enumerate(zip(current_embeddings, omegas, encoded_context)):
            temp_embeddings = initial_embeddings
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

                if self.skipped:
                    inv_inputs = torch.cat((context_encodings, temp_embeddings), -1)
                else:
                    inv_inputs = temp_embeddings

                # estimate state
                estimated_future_state[j,i,:] = self.inv_eigenfunction(inv_inputs)

                # estimate spectrum
                temp_omegas = self.spectrum(inv_inputs)


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
            future_embeddings[i,:,:] = self.eigenfunction(ith_future, ith_context_repeated)

        return future_embeddings
