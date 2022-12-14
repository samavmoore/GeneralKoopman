import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from Nets import ContextDecoder, ContextEncoder, Eigenfunction, Inv_Eigenfunction


############### ------------------------------ Context Pretraining Module --------------------------------##################

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


############### ------------------------------ Eigenfunction Pretraining Module ----------------------------------################

class EigenPretrain(LightningModule):
    def __init__(self, hid_layer_shape1: int=64,
                       hid_layer_shape2: int=128,
                       context_pretrain: bool=True,
                       skipped_connections: bool=False,
                       learning_rate: float=0.00691,
                       Context_NN_Path: str='my/path'):
        super().__init__()
        self.save_hyperparameters()
        self.context_pretrained = context_pretrain

        if self.context_pretrained:
            self.Context_NN = Context.load_from_checkpoint(Context_NN_Path)
            self.Context_NN.freeze()
            self.encoder = self.Context_NN.Encoder
            self.decoder = self.Context_NN.Decoder

        else:
            self.encoder = ContextEncoder()
            self.decoder = ContextDecoder()


        self.skipped = skipped_connections

        if self.skipped:
            inv_input_shape = 3
        else:
            inv_input_shape = 2

        self.eigenfunction = Eigenfunction(eigenfunction_hidden_shape1=hid_layer_shape1, eigenfunction_hidden_shape2=hid_layer_shape2)
        self.inv_eigenfunction = Inv_Eigenfunction(inv_eigenfunction_input_shape=inv_input_shape, eigenfunction_hidden_shape1=hid_layer_shape1, eigenfunction_hidden_shape2=hid_layer_shape2)

        self.learning_rate = learning_rate

    def forward(self, state, raw_context):

        context_rep = self.encoder(raw_context)

        embedded_state = self.eigenfunction(state, context_rep)

        if self.skipped:
            inv_inputs = torch.cat((context_rep, embedded_state), -1)
        else:
            inv_inputs= embedded_state

        recon_state = self.inv_eigenfunction(inv_inputs)

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