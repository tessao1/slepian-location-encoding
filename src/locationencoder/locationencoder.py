import numpy as np
import torch
from torch import optim, nn
from torchmetrics import AUROC, Accuracy, F1Score

import lightning.pytorch as pl

import locationencoder.pe as PE
import locationencoder.nn as NN
from utils.losses import AN_loss

from sklearn.metrics import (
    accuracy_score, 
    jaccard_score,
    mean_absolute_error,
    f1_score,
    roc_auc_score
)
from locationencoder.pe.utils_mask import CoastlineMask

def get_positional_encoding(name, hparams=None):
    if name == "direct":
        return PE.Direct()
    elif name == "slepian":
        return PE.Slepian(
            legendre_polys=hparams['legendre_polys'])
    elif name == "sphericalharmonics":

        # default to analytical
        if "harmonics_calculation" not in hparams.keys():
            hparams["harmonics_calculation"] = "analytic"

        if "harmonics_calculation" in hparams.keys() and hparams['harmonics_calculation'] == "discretized":
            return PE.DiscretizedSphericalHarmonics(legendre_polys=hparams['legendre_polys'])
        else:
            return PE.SphericalHarmonics(legendre_polys=hparams['legendre_polys'],
                                         harmonics_calculation=hparams['harmonics_calculation'])
    elif name == "slepianhybrid":
        # default to analytical
        if "harmonics_calculation" not in hparams.keys():
            hparams["harmonics_calculation"] = "analytic"

        return PE.SlepianSHHybrid(
            legendre_polys=hparams['legendre_polys'],
            harmonics_calculation=hparams['harmonics_calculation'],
            sh_max_degree=hparams.get('sh_max_degree', 5)
        )
    elif name == "wavelets":
        return PE.Wavelets() 
    else:
        raise ValueError(f"{name} not a known positional encoding.")

def get_neural_network(name, input_dim, hparams=None):
    if name == "linear":
        return nn.Linear(input_dim, hparams['num_classes'])
    elif name == "mlp":
        return NN.MLP(
            num_inputs=input_dim,
            num_classes=hparams['num_classes'],
            dim_hidden=hparams['dim_hidden'],
            num_layers=hparams.get('num_layers', 3),
            dropout=hparams['dropout'] if "dropout" in hparams.keys() else False
        )
    elif name ==  "siren":
        return NN.SirenNet(
                dim_in=input_dim,
                dim_hidden=hparams['dim_hidden'],
                num_layers=hparams['num_layers'],
                dim_out=hparams['num_classes'],
                dropout=hparams['dropout'] if "dropout" in hparams.keys() else False
            )
    elif name == "fcnet":
        return NN.FCNet(
                num_inputs=input_dim,
                num_classes=hparams['num_classes'],
                dim_hidden=hparams['dim_hidden']
            )
    else:
        raise ValueError(f"{name} not a known neural networks.")

def get_param(hparams, key, default=False):
    """
    Convenience function that indexes the hyperparameter dict but returns a default value if not defined rather than
    an error
    """
    return hparams[key] if key in hparams.keys() else default

# define the LightningModule
class LocationEncoder(pl.LightningModule):
    def __init__(self, positional_encoding_name, neural_network_name, hparams):
        super().__init__()

        self.learning_rate = hparams["lr"]
        self.weight_decay = hparams["wd"]
        self.regression = get_param(hparams, "regression")
        self.use_highres_metrics = get_param(hparams, "use_highres_metrics", False)

        # Initialize loss function
        if self.use_highres_metrics:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # Store predictions and labels for end-of-epoch metric computation
            self.test_outputs = {'uniform': [], 'coastline': [], 'island': []}
        else:
            self.loss_fn = AN_loss()
            
        self.positional_encoder = get_positional_encoding(positional_encoding_name, hparams)
        self.neural_network = get_neural_network(
            neural_network_name,
            input_dim=self.positional_encoder.embedding_dim,
            hparams=hparams
        )

        # this enables LocationEncoder.load_from_checkpoint(path)
        self.save_hyperparameters()

    def forward(self, lonlats):
        """Forward pass through positional encoder and neural network"""
        embedding = self.positional_encoder(lonlats)
        return self.neural_network(embedding)

    def _compute_loss(self, lonlats, label, prediction_logits):
        """Compute loss based on dataset type"""
        if self.use_highres_metrics:
            return self.loss_fn(prediction_logits.squeeze(), label.squeeze())
        else:
            return self.loss_fn(self, lonlats, label)

    def training_step(self, batch, batch_idx):
        lonlats, label = batch
        prediction_logits = self.forward(lonlats)
        loss = self._compute_loss(lonlats, label, prediction_logits)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lonlats, label = batch
        prediction_logits = self.forward(lonlats)
        loss = self._compute_loss(lonlats, label, prediction_logits)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        lonlats, label = batch
        prediction_logits = self.forward(lonlats)
        loss = self._compute_loss(lonlats, label, prediction_logits)

        if self.use_highres_metrics:
            return self._test_step_highres(lonlats, label, prediction_logits, loss, dataloader_idx)
        else:
            return self._test_step_standard(lonlats, label, prediction_logits, loss)

    def _test_step_highres(self, lonlats, label, prediction_logits, loss, dataloader_idx):
        """Test step for high-resolution dataset - accumulate predictions"""
        probs = torch.sigmoid(prediction_logits).squeeze()
        y_true = label.squeeze()
        
        dataset_name = ['uniform', 'coastline', 'island'][dataloader_idx]
        
        # Store predictions and labels for end-of-epoch computation
        self.test_outputs[dataset_name].append({
            'probs': probs.cpu(),
            'labels': y_true.cpu(),
            'loss': loss.item()
        })
        
        return {f"test_loss_{dataset_name}": loss}
    
    def on_test_epoch_end(self):
        """Compute metrics at the end of test epoch - like the notebook code"""
        if not self.use_highres_metrics:
            return
            
        for dataset_name in ['uniform', 'coastline', 'island']:
            if not self.test_outputs[dataset_name]:
                continue
                
            # Concatenate all batches
            all_probs = torch.cat([x['probs'] for x in self.test_outputs[dataset_name]])
            all_labels = torch.cat([x['labels'] for x in self.test_outputs[dataset_name]])
            avg_loss = np.mean([x['loss'] for x in self.test_outputs[dataset_name]])
            
            # Convert to numpy for sklearn
            all_probs_np = all_probs.numpy().flatten()
            all_labels_np = all_labels.numpy().flatten()
            all_preds_np = (all_probs_np > 0.5).astype(float)
            
            # Compute metrics exactly like the notebook
            accuracy = accuracy_score(all_labels_np, all_preds_np)
            f1 = f1_score(all_labels_np, all_preds_np)
            auc = roc_auc_score(all_labels_np, all_probs_np)
            
            # Log metrics
            self.log(f"test_loss_{dataset_name}", avg_loss, add_dataloader_idx=False)
            self.log(f"test_accuracy_{dataset_name}", accuracy, add_dataloader_idx=False)
            self.log(f"test_f1_{dataset_name}", f1, add_dataloader_idx=False)
            self.log(f"test_auc_{dataset_name}", auc, add_dataloader_idx=False)
            
            # Clear for next test run
            self.test_outputs[dataset_name] = []
        
    def _test_step_standard(self, lonlats, label, prediction_logits, loss):
        """Standard test step for other datasets"""
        # Determine prediction type
        if prediction_logits.size(1) == 1 and not self.regression:
            y_pred = (prediction_logits.squeeze() > 0).cpu()
            average = "binary"
        elif self.regression:
            y_pred = prediction_logits.cpu()
        else:
            y_pred = prediction_logits.argmax(-1).cpu()
            average = "macro"

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        # Regression metrics
        if self.regression:
            mae = mean_absolute_error(y_true=label.cpu(), y_pred=y_pred)
            self.log("test_MAE", mae, on_step=False, on_epoch=True)
            return {"test_loss": loss, "test_MAE": mae}
        
        # Classification metrics
        accuracy = float(accuracy_score(y_true=label.cpu(), y_pred=y_pred))
        iou = float(jaccard_score(y_true=label.cpu(), y_pred=y_pred, average=average, zero_division=0))
        
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("test_IoU", iou, on_step=False, on_epoch=True)
        
        # Compute local accuracy (coastline regions)
        mask_indices = CoastlineMask.is_in_masked_region(lonlats)
        if mask_indices.sum() > 0:
            y_pred_masked = y_pred[mask_indices.cpu()]
            y_masked = label[mask_indices].cpu()
            local_accuracy = float(accuracy_score(y_true=y_masked, y_pred=y_pred_masked))
            self.log("test_local_accuracy", local_accuracy, on_step=False, on_epoch=True)
        else:
            local_accuracy = None
        
        return {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_IoU": iou,
            "test_local_accuracy": local_accuracy
        }

    def predict_step(self, batch, batch_idx):
        lonlats, label = batch
        prediction_logits = self.forward(lonlats)
        return prediction_logits, lonlats, label

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {"params": self.neural_network.parameters()},
            {"params": self.positional_encoder.parameters(), "weight_decay": 0}
        ], lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer