from ConfigData import ContextDataModule, EigenPretrainModule, PendulumKoopmanDataModule
from MainLitModule import PendulumKoopModule
from PretrainingLitModules import Context, EigenPretrain
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def pretrain_context():

    context_dat_name = "your path/GeneralKoopman/context_data.npy"

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath="your path", \
                        filename='Context_AE-{epoch:02d}-{val_loss:.2f}')

    ES_callback = EarlyStopping(monitor='val_loss', patience=2)

    seed_everything(33, workers=True)

    context_dm = ContextDataModule(context_dat_name, batch_size = 64)

    AE_model = Context(learning_rate=.01)

    context_trainer = Trainer(max_epochs=500, callbacks=[checkpoint_callback, ES_callback])
    context_trainer.fit(AE_model, context_dm)

def pretrain_eigenfunctions():

    train_dat_name = "your path/GeneralKoopman/Pendulum_train_data.npy"

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='your path', \
                        filename='Eigen_Pretrain-WSkip-{epoch:02d}-{val_loss:.2f}')

    ES_callback = EarlyStopping(monitor='val_loss', patience=3)

    seed_everything(44, workers=True)

    eig_pretrain_dm = EigenPretrainModule(train_dat_name, batch_size = 256)

    trained_context_path = "your pretrained model path from above"

    # choose whether or not you want skipped connections. make sure to set the same flag when you do the main training
    eig_pretrain_model = EigenPretrain(Context_NN_Path=trained_context_path, learning_rate=0.001, skipped_connections=True)

    eig_trainer = Trainer(max_epochs=500, callbacks=[checkpoint_callback, ES_callback])
    eig_trainer.fit(eig_pretrain_model, eig_pretrain_dm)


#### run the main training loop without skipped connections (passing the context representation) to the gamma network and the inverse eigenfunction network
def Train_Main_WPretrain_WOSkip():
    train_dat_name = "/Users/Sam/Documents/GeneralKoopmanModels/GeneralKoopman/Pendulum_train_data.npy"
    val_dat_name = "/Users/Sam/Documents/GeneralKoopmanModels/GeneralKoopman/Pendulum_val_data.npy"

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='/Users/Sam/Documents/GeneralKoopmanModels/GeneralKoopman/Saved_Models',\
                                        filename='Main_training-WO-Skipped-{epoch:02d}-{val_loss:.2f}')

    ES_callback = EarlyStopping(monitor='val_loss', patience=20)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    seed_everything(38, workers=True)

    dm = PendulumKoopmanDataModule(train_data_path=train_dat_name, val_data_path=val_dat_name, batch_size = 64, n_shifts=10)

    pretrained_eigen_path = "your pretrained path from above"

    model=PendulumKoopModule(n_shifts=10, Eigenfunction_NN_Path=pretrained_eigen_path, skip_connections=False)

    eig_trainer = Trainer(max_epochs=500, val_check_interval=.2, callbacks=[checkpoint_callback, lr_monitor, ES_callback])

    eig_trainer.fit(model, dm)

#### run the main training loop with skipped connections (passing the context representation) to the gamma network and the inverse eigenfunction network
def Train_Main_WPretrain_WSkip():
    train_dat_name = "your path/GeneralKoopman/Pendulum_train_data.npy"
    val_dat_name = "your path/GeneralKoopman/Pendulum_val_data.npy"

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='your path',\
                                        filename='Main-WSkip-{epoch:02d}-{val_loss:.2f}')

    ES_callback = EarlyStopping(monitor='val_loss', patience=40)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    seed_everything(38, workers=True)

    dm = PendulumKoopmanDataModule(train_data_path=train_dat_name, val_data_path=val_dat_name, batch_size = 128, n_shifts=10)

    pretrained_eigen_path = "your path to the pretrained model above"

    model=PendulumKoopModule(n_shifts=10, Eigenfunction_NN_Path=pretrained_eigen_path, skip_connections=True)

    eig_trainer = Trainer(max_epochs=500, log_every_n_steps=20, callbacks=[checkpoint_callback, lr_monitor, ES_callback])

    eig_trainer.fit(model, dm)


if __name__=="__main__":
    # run these

    pretrain_context()
    #pretrain_eigenfunctions()
    #Train_Main_WPretrain_WOSkip()
    #Train_Main_WPretrain_WSkip()
    
    