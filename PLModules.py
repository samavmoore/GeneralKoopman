import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from Nets import ContextDecoder, ContextEncoder, Eigenfunction, Inv_Eigenfunction, Spectrum


class Context(LightningModule):
    def __init__(self, hid_layer_1_shape=64,
                       learning_rate=1e-3):
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



class EigenPretrain(LightningModule):
    def __init__(self, hid_layer_shape1 = 48,
                       hid_layer_shape2=96,
                       learning_rate=1e-4,
                       Context_NN_Path=None):
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
    