import lightning as L
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.unet import MNISTUNet
from model.cvf import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from torchmetrics import MeanMetric


optimizer_map = {'adamW': AdamW, 'adam': Adam}

class FlowMatchingModule(L.LightningModule):
    def __init__(self, model_cfg: dict, train_cfg: dict):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        #load model
        self.model = MNISTUNet(
            channels = model_cfg['channels'],
            num_residual_layers = model_cfg['num_residual_layers'],
            t_embed_dim = model_cfg['t_embed_dim'],
            y_embed_dim = model_cfg['y_embed_dim'],
        )
        
        #probability to set label to null
        self.eta = model_cfg['eta']

        #initialize conditional probability path
        self.path = GaussianConditionalProbabilityPath(
            p_simple_shape = model_cfg['input_size'], #sampled Gaussian must match image size
            alpha = LinearAlpha(),
            beta = LinearBeta()
        )

        #set training parameters
        self.optimizer_type = optimizer_map[train_cfg['optimizer']]
        self.learning_rate = train_cfg['learning_rate']
        self.monitor = train_cfg['early_stopping_val']

        #initialize metrics as ModuleDict to enforce them as attributes of the LightningModule
        self.metrics = nn.ModuleDict()
        for split in ['train/', 'val/', 'test/']:
            self.metrics[split] = MeanMetric() #collect CFM loss

    def model_step(self, batch, data_split, batch_idx):

        # Step 1: Sample z,y from p_data
        z, y = batch
        batch_size = z.shape[0]

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        mask = torch.rand(batch_size) > self.eta
        y[mask] = 10

        # Step 3: Sample t and x
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)

        x = self.path.sample_conditional_path(z, t)

        # Step 4: Regress and output loss
        u_t_theta = self.model(x, t, y)
        u_ref = self.path.conditional_vector_field(x, z, t)
        cfm_loss = torch.mean((u_t_theta - u_ref) ** 2)

        #log metrics
        self.metrics[data_split](cfm_loss)
        self.log(f'{data_split}cfm_loss', self.metrics[data_split], on_step=False, on_epoch=True, prog_bar=True)
        
        #loss required for backprop
        return cfm_loss


    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, "train/", batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model_step(batch, "val/", batch_idx)
    
    def test_step(self, batch, batch_idx):
        self.model_step(batch, "test/", batch_idx)

    def configure_optimizers(self):
        optimizer = self.optimizer_type(params=self.trainer.model.parameters(), 
                                        lr=self.learning_rate,
                                        amsgrad=True)
        
        #learning rate scheduler used in spliceai pytorch implementation
        scheduler = ReduceLROnPlateau(optimizer=optimizer)

        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.monitor
                }
            }
