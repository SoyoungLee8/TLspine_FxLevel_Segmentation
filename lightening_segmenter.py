import monai
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import torchmetrics

def Activation(tensor, T=1):
    if tensor.shape[1] != 1:
        return F.softmax(tensor/T,1)
    else:
        return F.sigmoid(tensor/T)
    
    
class Segmentor(pl.LightningModule):
    def __init__(self, network, lossfn, metricfn, experiment_name):
        super().__init__()
        
        self.net = network
        self.lossfn = lossfn
        self.metricfn = metricfn
        self.experiment_name = experiment_name
        self.best_val_loss_epoch = np.inf            
        self.best_valid_epoch = 0
        
        if isinstance(lossfn,list) and isinstance(metricfn,list):
            assert len(lossfn) == len(metricfn)
        
    def forward(self, x):
        return self.net(x)
        
    def pipeline(self, x, ys, fname, plot=False):
        yhat = self.net(x)
            
        if isinstance(self.lossfn, list):
            loss, metric_fracture, metric_position = 0, 0, 0
            yhat = list(yhat) # tuple to list
            for i in range(len(yhat)):
                yhat[i] = Activation(yhat[i])
                loss += self.lossfn[i](yhat[i], ys[i])
                
            # if plot:
            #     for idx in range(len(x)):
            #         print(f'the below image is {fname[idx]}')
            #         visualize(image=x[idx,0], y_fracture = ys[0][idx,0], yhat_fracture = refine_fracture(yhat[0])[idx,0], y_position = ys[1][idx,0], yhat_position = refine_position(yhat[1])[idx,0])
            metric_fracture = self.metricfn[0](yhat[0], ys[0])
            metric_position = self.metricfn[1](yhat[1], ys[1])
#           
            return loss, [metric_fracture, metric_position]
        else:
            yhat = Activation(yhat)   
            loss = self.lossfn(yhat, ys)
            metric = self.metricfn(yhat, ys)
            if plot:
                for idx in range(len(x)):
                    visualize(image=x[idx,0], y = ys[idx,0], yhat = torch.argmax(yhat,1)[idx,0])

            return loss, metric
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9)
        # return optimizer
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def training_step(self, batch, batch_idx):
        ############################# DEPEND ON DATASET #################################
        x, y_fracture, y_position, fname = batch['x'], batch['y_fracture'], batch['y_position'], batch['fname']
        #################################################################################      
        loss, metric = self.pipeline(x, [y_fracture, y_position], fname, False)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True,)
        if isinstance(metric,list):
            self.log('fx_dice', metric[0], on_step=True, on_epoch=True, prog_bar=True,)
            self.log('level_dice', metric[1], on_step=True, on_epoch=True, prog_bar=True,)
        else:
            self.log('dice', metric, on_step=True, on_epoch=True, prog_bar=True,)            
        return loss
      
    def validation_step(self, batch, batch_idx):
        ############################# DEPEND ON DATASET #################################
        x, y_fracture, y_position, fname = batch['x'], batch['y_fracture'], batch['y_position'], batch['fname']
        #################################################################################      
        loss, metric = self.pipeline(x, [y_fracture, y_position], fname, False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True,)
        if isinstance(metric,list):
            self.log('val_fx_dice', metric[0], on_epoch=True, prog_bar=True,)
            self.log('val_level_dice', metric[1], on_epoch=True, prog_bar=True,)
        else:
            self.log('val_dice', metric, on_epoch=True, prog_bar=True,)            
        return {"val_loss":loss}
    
    def test_step(self, batch, batch_idx):
        ############################# DEPEND ON DATASET #################################
        x, y_fracture, y_position, fname = batch['x'], batch['y_fracture'], batch['y_position'], batch['fname']
        #################################################################################      
        # loss, metric = self.pipeline(x, [y_fracture, y_position], fname, False)
        # loss, metric, test_y_fracture, test_yhat_fracture, test_y_position, test_yhat_position = self.pipeline(x, [y_fracture, y_position], fname, False, True)
        ys_roc, yhat_roc = self.pipeline(x, [y_fracture, y_position], fname, False)

        return {'ys_roc':ys_roc,'yhat_roc':yhat_roc}
        
    def validation_epoch_end(self, outputs):
        val_losses = []
        for output in outputs:
            val_losses.append(output["val_loss"].cpu().detach().numpy())
        val_loss_epoch = np.mean(val_losses)
        self.log('val_loss_epoch', val_loss_epoch)
        
        if val_loss_epoch < self.best_val_loss_epoch and self.current_epoch>0:
            self.best_valid_epoch = self.current_epoch
            self.best_val_loss_epoch = val_loss_epoch             
        print(
            f"current epoch: {self.current_epoch}, "
            f"current epoch val_loss: {val_loss_epoch:.4f}, "
            f"best epoch val_loss: {self.best_val_loss_epoch:.4f}, "
            f"at epoch: {self.best_valid_epoch}, " 
        )
        