import lightning as L
import torch
import yaml
from model.module import FlowMatchingModule
from data_module import MNISTDataModule
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')

def train_model(model_cfg: dict, train_cfg: dict, data_cfg: dict):

    #set seed for random number generators across pytorch, numpy
    L.seed_everything(train_cfg['seed'], workers=True)

    #initialize logger
    logger = TensorBoardLogger(
        save_dir=train_cfg['save_path'],
        name=train_cfg['run_name']
    )

    #create data module for splice junction dataset
    datamodule = MNISTDataModule(
        data_dir=data_cfg['data_dir'],
        train_val_split=data_cfg['train_val_split'],
        batch_size=data_cfg['batch_size']
    )

    #initialize model
    model = FlowMatchingModule(model_cfg, train_cfg)
    if model_cfg['compile']:
        model = torch.compile(model)

    #docs: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
    #monintor: logged quantity to be monitored (ex. val/loss)
    early_stopping = EarlyStopping(monitor=train_cfg['early_stopping_val'], 
                                   patience=train_cfg['patience'], 
                                   check_finite=True)
    
    accelerator = "auto"
    if torch.cuda.is_available():
        accelerator = "gpu"

    #initiate trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1, 
        logger=logger,
        callbacks=[early_stopping],
        min_epochs=1,
        max_epochs=train_cfg['max_epochs'],
        gradient_clip_val=train_cfg['max_grad_norm'],
        default_root_dir=train_cfg['save_path']
    )

    #train the model on data
    trainer.fit(model=model, datamodule=datamodule)

    #test the model
    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    return model


if __name__ == "__main__":
    #load config file
    with open('setup.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    #begin training
    train_model(model_cfg=cfg["model"],
          train_cfg=cfg["train"],
          data_cfg=cfg["data"])