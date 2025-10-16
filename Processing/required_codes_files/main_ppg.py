#bpbenchmark
#python main_ppg.py --data $GROUP/datasets/ppg_data/bp_benchmark/memmap --input-size 625 --architecture xresnet1d50 --finetune-dataset bpbenchmark --refresh-rate 1

#pulsedb
#python main_ppg.py --data $GROUP/datasets/pulseDB/merged/mixied_mimic_vital --input-size 1250 --architecture xresnet1d50 --finetune-dataset pulsedb_calibfree --refresh-rate 1


#s4 flags: --architecture s4 --precision 32 --s4-n 8 --s4-h 512 --batch-size 32 --epochs 30
###############
#generic
import torch
from torch import nn
import lightning.pytorch as lp
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import os
import subprocess
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from clinical_ts.xresnet1d import xresnet1d50,xresnet1d101, xresnet1d50_MCD
from clinical_ts.inception1d import inception1d

from clinical_ts.s4_model import S4Model
from clinical_ts.misc_utils import add_default_args, LRMonitorCallback
#################
#specific
from clinical_ts.timeseries_utils import *
from clinical_ts.schedulers import *
from clinical_ts.specifity_sensitivity import *

##lenet
from clinical_ts.lenet1d import lenet1d


from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pickle

from pathlib import Path
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#mlflow without autologging https://github.com/zjohn77/lightning-mlflow-hf/blob/74c30c784f719ea166941751bda24393946530b7/lightning_mlflow/train.py#L39
MLFLOW_AVAILABLE=True
try:
    import mlflow
    from lightning.pytorch.loggers import MLFlowLogger
    from omegaconf import DictConfig, ListConfig

    def log_params_from_namespace(hparams):
        
        for k in hparams.__dict__.keys():
            mlflow.log_param(k," " if str(hparams.__dict__[k])=="" else str(hparams.__dict__[k]))

except ImportError:
    MLFLOW_AVAILABLE=False


def get_git_revision_short_hash():
    return ""#subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def multihot_encode(x, num_classes):
    res = np.zeros(num_classes,dtype=np.float32)
    for y in x:
        res[y]=1
    return res

############################################################################################################
# simple constant baseline

class ConstantBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.params = nn.Parameter(torch.ones([num_classes]))
        
    def forward(self, x):
        return torch.unsqueeze(self.params,dim=0).expand(x.shape[0],self.params.shape[0])
    
############################################################################################################

        
class Main_PPG(lp.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.lr = self.hparams.lr

        print(hparams)
        if(hparams.finetune_dataset == "mimiciiibp" or hparams.finetune_dataset.startswith("bpbenchmark") or hparams.finetune_dataset.startswith("pulsedb")):
            num_classes = 2
            self.task = "regression"
        elif(hparams.finetune_dataset == "deepbeat"):
            num_classes = 2
            self.task = "classification"

        # also works in the segmentation case
        self.criterion = F.cross_entropy if (self.task == "classification")  else F.mse_loss

        if(hparams.architecture=="xresnet1d50"):
            self.model = xresnet1d50(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="xresnet1d50_MCD"):
            self.model = xresnet1d50_MCD(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="xresnet1d101"):
            self.model = xresnet1d101(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="inception1d"):
            self.model = inception1d(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="s4"):
            self.model = S4Model(d_input=hparams.input_channels, d_output=num_classes, l_max=self.hparams.input_size, d_state=self.hparams.s4_n, d_model=self.hparams.s4_h, n_layers = self.hparams.s4_layers,bidirectional=True)#,backbone="s4new")
        elif(hparams.architecture=="baseline"):
            self.model = ConstantBaseline(num_classes=num_classes)
        elif(hparams.architecture=="lenet1d"):
            self.model = lenet1d(input_channels=hparams.input_channels, num_classes=num_classes)          
        else:
            assert(False)

    def forward(self, x, **kwargs):
        if(len(x.shape)==2):#no separate channel axis
            x = x.unsqueeze(1) #add an empty channel axis
        
        if(self.hparams.normalize):#normalize all sequences (individually) to enforce min=0 and max=1
            xmin = x.min(dim=2,keepdim=True)[0]
            xmax = x.max(dim=2,keepdim=True)[0]+1e-8
            x2 = (x-xmin)/(xmax-xmin)
        else:
            x2 = x
            
        
        
        
          
        #temporary workaround to enforce fp32 inputs
        return self.model(x2.float(), **kwargs)
    
    def on_validation_epoch_end(self):
        for i in range(len(self.val_preds)):
            self.on_valtest_epoch_eval({"preds":self.val_preds[i], "targs":self.val_targs[i]}, dataloader_idx=i, test=False)
            self.val_preds[i].clear()
            self.val_targs[i].clear()
    
    def on_test_epoch_end(self):
        for i in range(len(self.test_preds)):
            self.on_valtest_epoch_eval({"preds":self.test_preds[i], "targs":self.test_targs[i]}, dataloader_idx=i, test=True)
            self.test_preds[i].clear()
            self.test_targs[i].clear()

    def eval_scores(self, targs,preds,classes=None):
        if(self.task == "regression"):
            maes = np.abs((targs-preds))
            mae = np.mean(maes,axis=0)
            #IEEE 1708a-2019 grades (fraction of cases)
            gradea = np.mean(maes<=5.,axis=0)
            gradeb = np.mean(np.logical_and(maes>5.,maes<=6.),axis=0)
            gradec = np.mean(np.logical_and(maes>6.,maes<=7.),axis=0)
            graded = np.mean(maes>7.,axis=0)
            
            
            return {"mae0":mae[0],"mae1":mae[1],"gradea0":gradea[0],"gradea1":gradea[1],"gradeb0":gradeb[0],"gradeb1":gradeb[1],"gradec0":gradec[0],"gradec1":gradec[1],"graded0":graded[0],"graded1":gradea[1]}
            
        elif(self.task == "classification"):
            #targs_hard = np.argmax(targs,axis=1)
            preds_hard = np.argmax(preds,axis=1)
            acc = np.sum(targs == preds_hard)/len(targs)
            f1 = f1_score(targs,preds_hard)
            #try:
            auc = roc_auc_score(targs, preds[:,1])
            #except:
            #    auc = 0.5 #only one class present
            
            # Calculate sensitivity and specificity using qumphy for regression, pos_label is True for sensitivity, False for specificity. 
            
            # when the sensitivity is greater than 90
            sensitivity_gt_90_sensitivity, sensitivity_gt_90_specificity  = eval_sensitivity_specificity(targs, preds[:,1], 0.9, pos_label=True, greater_than=True)
            
            # when the specificity is greater than 90
            specificity_gt_90_sensitivity, specificity_gt_90_specificity  = eval_sensitivity_specificity(targs, preds[:,1], 0.9, pos_label=False, greater_than=True)
            
            # when the sensitivity is greater than 80
            sensitivity_gt_80_sensitivity, sensitivity_gt_80_specificity  = eval_sensitivity_specificity(targs, preds[:,1], 0.8, pos_label=True, greater_than=True)
            
            # when the specificity is greater than 80
            specificity_gt_80_sensitivity, specificity_gt_80_specificity  = eval_sensitivity_specificity(targs, preds[:,1], 0.8, pos_label=False, greater_than=True)
            
            
            
            
            #print( sensitivity, specificity)
            #print(preds[:,1])
            #print(targs)
            
            return {"acc":acc, "f1":f1, "auc":auc,"sensitivity_gt_90_sensitivity": sensitivity_gt_90_sensitivity, "sensitivity_gt_90_specificity": sensitivity_gt_90_specificity,"specificity_gt_90_sensitivity": specificity_gt_90_sensitivity, "specificity_gt_90_specificity": specificity_gt_90_specificity, "sensitivity_gt_80_sensitivity": sensitivity_gt_80_sensitivity, "sensitivity_gt_80_specificity": sensitivity_gt_80_specificity,"specificity_gt_80_sensitivity": specificity_gt_80_sensitivity, "specificity_gt_80_specificity": specificity_gt_80_specificity }
        else:
            return {}

    def on_valtest_epoch_eval(self, outputs_all, dataloader_idx, test=False):
        #for dataloader_idx,outputs in enumerate(outputs_all): #multiple val dataloaders
            preds_all = torch.cat(outputs_all["preds"]).cpu()
            targs_all = torch.cat(outputs_all["targs"]).cpu()
            # apply softmax/sigmoid to ensure that aggregated scores are calculated based on them
            if(self.task=="classification"):
                preds_all = F.softmax(preds_all.float(),dim=-1)
                #targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds_all.device) 
            
            preds_all = preds_all.numpy()
            targs_all = targs_all.numpy()

            #instance level score
            res = self.eval_scores(targs_all,preds_all,classes=self.lbl_itos)
            res = {k+"_noagg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res.items()}
            self.log_dict(res)
            print("epoch",self.current_epoch,"test" if test else "val"+str(dataloader_idx),"noagg:",res)#,"agg:",res_agg)
            #aggregated score
            preds_all_agg,targs_all_agg = aggregate_predictions(preds_all,targs_all,self.test_idmaps[dataloader_idx] if test else self.val_idmaps[dataloader_idx],aggregate_fn=np.mean)
            res_agg = self.eval_scores(targs_all_agg,preds_all_agg,classes=self.lbl_itos)
            res_agg = {k+"_agg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res_agg.items()}
            self.log_dict(res_agg)
            
            #export preds_all_agg,targs_all_agg
            #if(test and self.hparams.export_predictions):
            if(test):   
                version_output_path = Path(logger[0].log_dir)
                # Ensure preds_all_agg and targs_all_agg are tensors
                if not isinstance(preds_all_agg, torch.Tensor):
                    preds_all_agg = torch.tensor(preds_all_agg)
                if not isinstance(targs_all_agg, torch.Tensor):
                    targs_all_agg = torch.tensor(targs_all_agg)                
                #np.savez(Path(self.hparams.output_path)/("preds_test0.npz" if dataloader_idx==0 else "preds_test.npz"),preds_all_agg.cpu().numpy(),targs_all_agg.cpu().numpy())
                np.savez(version_output_path / ("preds_test0.npz" if dataloader_idx == 0 else "preds_test.npz"),preds_all_agg.cpu().numpy(), targs_all_agg.cpu().numpy())
            print("epoch",self.current_epoch,("test" if test else "val")+str(dataloader_idx),"agg:",res_agg)#,"agg:",res_agg)
            
    def setup(self, stage):
        # configure dataset params
        chunkify_train = self.hparams.chunkify_train
        chunk_length_train = int(self.hparams.chunk_length_train*self.hparams.input_size) if chunkify_train else 0
        stride_train = int(self.hparams.stride_fraction_train*self.hparams.input_size)
        
        chunkify_valtest = True
        chunk_length_valtest = self.hparams.input_size if chunkify_valtest else 0
        stride_valtest = int(self.hparams.stride_fraction_valtest*self.hparams.input_size)

        train_datasets = []
        val_datasets = []
        test_datasets = []

        self.ds_mean = None
        self.ds_std = None

        for i,target_folder in enumerate(list(self.hparams.data.split(","))):
            target_folder = Path(target_folder)           
            
            df_mapped, lbl_itos,  mean, std = load_dataset(target_folder)
        
            if(self.hparams.finetune_dataset.startswith("pulsedb")):#pulsedb subset
                if(self.hparams.finetune_dataset.endswith("vital")):
                    df_mapped=df_mapped[df_mapped.source==1].copy()
                elif(self.hparams.finetune_dataset.endswith("mimic")):
                    df_mapped=df_mapped[df_mapped.source==0].copy()
            elif(self.hparams.finetune_dataset.startswith("bpbenchmark")):#bpbenchmark subset
                if(self.hparams.finetune_dataset.endswith("sensors")):
                    df_mapped=df_mapped[df_mapped.dataset=="sensors_dataset"].copy()
                elif(self.hparams.finetune_dataset.endswith("uci")):
                    df_mapped=df_mapped[df_mapped.dataset=="uci2_dataset"].copy()
                elif(self.hparams.finetune_dataset.endswith("bcg")):
                    df_mapped=df_mapped[df_mapped.dataset=="bcg_dataset"].copy() 
                elif(self.hparams.finetune_dataset.endswith("ppgbp")):
                    df_mapped=df_mapped[df_mapped.dataset=="ppgbp_dataset"].copy()
            if(self.hparams.finetune_dataset.startswith("bpbenchmark") or self.hparams.finetune_dataset.startswith("pulsedb")):#WORKAROUND
                df_mapped["sp"]=df_mapped["sp" if self.hparams.finetune_dataset.startswith("bpbenchmark") else "sbp_avg"].astype(np.float32)
                df_mapped["dp"]=df_mapped["dp" if self.hparams.finetune_dataset.startswith("bpbenchmark") else "dbp_avg"].astype(np.float32)
                df_mapped["label"]=df_mapped.apply(lambda row: np.array([row["sp"],row["dp"]],dtype=np.float32), axis=1)
            
            print("Folder:",target_folder,"Samples:",len(df_mapped))

            if(self.ds_mean is None):
                self.ds_mean = mean
                self.ds_std = std
                self.lbl_itos = lbl_itos

            tfms = ToTensor() if self.hparams.select_input_channel is None else transforms.Compose([ChannelFilter(self.hparams.select_input_channel),ToTensor()])
            
            if(self.hparams.disregard_splits):#ignore splits
                df_train= df_mapped
                df_val = df_mapped
                df_tests = [df_mapped]
            elif("set" in df_mapped.columns or "set_revised" in df_mapped.columns):#use predefined splits
                set_string = "set_revised" if "set_revised" in df_mapped.columns else "set" #in doubt use the revised splits
                if(set_string == "set_revised"):
                    print("INFO: using revised splits.")
                if(self.hparams.finetune_dataset.startswith("pulsedb")):
                    col_set = self.hparams.finetune_dataset.split("_")[1]
                    assert(col_set in ["calib", "calibfree", "aami"])
                    col_set = set_string+"_"+col_set
                else:
                    col_set = set_string
                df_train = df_mapped[(df_mapped[col_set]==0)&(df_mapped.fold<=self.hparams.num_training_folds)] if self.hparams.num_training_folds>0 else df_mapped[df_mapped[col_set]==0]
                if(len(df_mapped[df_mapped[col_set]==3])==0 or not self.hparams.join_calib_val):#no calibration set present
                    df_val = df_mapped[(df_mapped[col_set]==1)&(df_mapped.fold<=self.hparams.num_validation_folds)] if self.hparams.num_validation_folds>0 else df_mapped[df_mapped[col_set]==1]
                else:#join calib and val
                    df_val = df_mapped[(df_mapped[col_set]>=1)&(df_mapped[col_set]<=2)&(df_mapped.fold<=self.hparams.num_validation_folds)] if self.hparams.num_validation_folds>0 else df_mapped[(df_mapped[col_set]>=1)&(df_mapped[col_set]<=2)]
                if(len(df_mapped[df_mapped[col_set]==3])==0):#no calibration set present
                    df_tests = [df_mapped[df_mapped[col_set]==2]]
                else:
                    if(self.hparams.join_calib_val):
                        df_tests = [df_mapped[df_mapped[col_set]==3]]
                    else:
                        df_tests = [df_mapped[df_mapped[col_set]==2],df_mapped[df_mapped[col_set]==3]] #calib, test scores (test0 corresponds to the calibration set..test1 corresponds to the real test set).
            else:# no fold assignments given
                max_folds = np.max(np.unique(df_mapped.fold))
                df_train = df_mapped[(df_mapped.fold<=self.hparams.num_training_folds)] if self.hparams.num_training_folds>0 else df_mapped[df_mapped.fold<max_folds-1]
                df_val = df_mapped[df_mapped.fold==max_folds-1]
                df_tests = [df_mapped[df_mapped.fold==max_folds]]

        
            #    print("val",df_val.label.value_counts())
            #    print("test",df_test.label.value_counts())
            
            train_datasets.append(TimeseriesDatasetCrops(df_train,self.hparams.input_size,data_folder=target_folder,chunk_length=chunk_length_train,min_chunk_length=self.hparams.input_size, stride=stride_train,transforms=tfms,col_lbl ="label" ,memmap_filename=target_folder/("memmap.npy")))
            val_datasets.append(TimeseriesDatasetCrops(df_val,self.hparams.input_size,data_folder=target_folder,chunk_length=chunk_length_valtest,min_chunk_length=self.hparams.input_size, stride=stride_valtest,transforms=tfms,col_lbl ="label",memmap_filename=target_folder/("memmap.npy")))
            for df_test in df_tests:
                test_datasets.append(TimeseriesDatasetCrops(df_test,self.hparams.input_size,data_folder=target_folder,chunk_length=chunk_length_valtest,min_chunk_length=self.hparams.input_size, stride=stride_valtest,transforms=tfms,col_lbl ="label",memmap_filename=target_folder/("memmap.npy")))
            
            print("\n",target_folder)
            if(i<len(self.hparams.data)):
                print("train dataset:",len(train_datasets[-1]),"samples")
            print("val dataset:",len(val_datasets[-1]),"samples")
            print("test dataset:",len(test_datasets[-1]),"samples")

        if(len(train_datasets)>1): #multiple data folders
            print("\nCombined:")
            self.train_dataset = ConcatDatasetTimeseriesDatasetCrops(train_datasets)
            self.val_datasets = [ConcatDatasetTimeseriesDatasetCrops(val_datasets)]+val_datasets
            print("train dataset:",len(self.train_dataset),"samples")
            print("val datasets (total):",len(self.val_datasets[0]),"samples")
            self.test_datasets = [ConcatDatasetTimeseriesDatasetCrops(test_datasets)]+test_datasets
            print("test datasets (total):",len(self.test_datasets[0]),"samples")
        else: #just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_datasets = val_datasets
            self.test_datasets = test_datasets
        
        #create empty lists for results
        self.val_preds=[[] for _ in range(len(self.val_datasets))]
        self.val_targs=[[] for _ in range(len(self.val_datasets))]
        self.test_preds=[[] for _ in range(len(self.test_datasets))]
        self.test_targs=[[] for _ in range(len(self.test_datasets))]
        
        # store idmaps for aggregation
        self.val_idmaps = [ds.get_id_mapping() for ds in self.val_datasets]
        self.test_idmaps = [ds.get_id_mapping() for ds in self.test_datasets]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True, drop_last = True)
        
    def val_dataloader(self):
        return [DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=8) for ds in self.val_datasets]
    
    def test_dataloader(self):
        return [DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=8) for ds in self.test_datasets]
        
    def _step(self,data_batch, batch_idx, train, test=False, dataloader_idx=0):
        #if(torch.sum(torch.isnan(data_batch[0])).item()>0):#debugging
        #    print("nans",torch.sum(torch.isnan(data_batch[0])).item())
        preds_all = self.forward(data_batch[0])

        if(self.hparams.finetune_dataset=="mimiciiibp"):
            loss = self.criterion(preds_all.view(-1),data_batch[1].view(-1))#flatten everything (and convert bp integers to float)
        else:
            loss = self.criterion(preds_all,data_batch[1])

        self.log("train_loss" if train else ("test_loss" if test else "val_loss"), loss)
        
        if(not train and not test):
            self.val_preds[dataloader_idx].append(preds_all.detach())
            self.val_targs[dataloader_idx].append(data_batch[1])
        elif(not train and test):
            self.test_preds[dataloader_idx].append(preds_all.detach())
            self.test_targs[dataloader_idx].append(data_batch[1])

        return loss
    
    def training_step(self, train_batch, batch_idx):
               
        
        # x is the batch of PPG signals, and y is the batch of corresponding labels (blood pressure values)
        #x, y= train_batch
        # Print the first signal and its corresponding label
        #print("First signal in the batch:", x[0])
        #print("Corresponding blood pressure label:", y[0])
        
        return self._step(train_batch,batch_idx,train=True) 
        
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,train=False,test=False, dataloader_idx=dataloader_idx)
    
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        return self._step(test_batch,batch_idx,train=False,test=True, dataloader_idx=dataloader_idx)
    
    def configure_optimizers(self):
        
        if(self.hparams.optimizer == "sgd"):
            opt = torch.optim.SGD
        elif(self.hparams.optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
            
        params = self.parameters()

        optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay)

        if(self.hparams.lr_schedule=="const"):
            scheduler = get_constant_schedule(optimizer)
        elif(self.hparams.lr_schedule=="warmup-const"):
            scheduler = get_constant_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="warmup-cos"):
            scheduler = get_cosine_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=0.5)
        elif(self.hparams.lr_schedule=="warmup-cos-restart"):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)
        elif(self.hparams.lr_schedule=="warmup-poly"):
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)   
        elif(self.hparams.lr_schedule=="warmup-invsqrt"):
            scheduler = get_invsqrt_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="linear"): #linear decay to be combined with warmup-invsqrt c.f. https://arxiv.org/abs/2106.04560
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, self.hparams.epochs*len(self.train_dataloader()))
        else:
            assert(False)
        return (
        [optimizer],
        [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        ])
        
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
            
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    
    def load_state_dict(self, state_dict):
        #S4-compatible load_state_dict
        for name, param in self.named_parameters():
            param.data = state_dict[name].data.to(param.device)
        for name, param in self.named_buffers():
            param.data = state_dict[name].data.to(param.device)
            
class Main_PPG_MCD(Main_PPG):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        #nn.Module.__init__(self)
        self.lr = self.hparams.lr

        print(hparams)
        if(hparams.finetune_dataset == "mimiciiibp" or hparams.finetune_dataset.startswith("bpbenchmark") or hparams.finetune_dataset.startswith("pulsedb")):
            if hparams.gnll:
                num_classes = 4
                self.criterion =nn.GaussianNLLLoss(eps=1e-6)
            else:
                num_classes = 2
                self.criterion = F.mse_loss
            self.task = "regression"
        elif(hparams.finetune_dataset == "deepbeat"):
            num_classes = 2
            self.task = "classification"
            self.criterion = F.cross_entropy

        # also works in the segmentation case
        #self.criterion = F.cross_entropy if self.task == "classification" else nn.GaussianNLLLoss(eps=1e-6)# F.mse_loss

        if(hparams.architecture=="xresnet1d50_MCD"):
            self.model = xresnet1d50_MCD(input_channels=hparams.input_channels, num_classes=num_classes, act_head="relu", ps_head=0.1)
        elif(hparams.architecture=="xresnet1d101"):
            self.model = xresnet1d101(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="inception1d"):
            self.model = inception1d(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="s4"):
            self.model = S4Model(d_input=hparams.input_channels, d_output=num_classes, l_max=self.hparams.input_size, d_state=self.hparams.s4_n, d_model=self.hparams.s4_h, n_layers = self.hparams.s4_layers,bidirectional=True)#,backbone="s4new")
        elif(hparams.architecture=="baseline"):
            self.model = ConstantBaseline(num_classes=num_classes)
        elif(hparams.architecture=="lenet1d"):
            self.model = lenet1d(input_channels=hparams.input_channels, num_classes=num_classes)          
        else:
            assert(False)
        self.test_vars = None
            
    def _step(self, data_batch, batch_idx, train, test=False, dataloader_idx=0):
        preds_all = self.forward(data_batch[0])  # [batch_size, 4] if gnll=True, [batch_size, 2] if gnll=False
        if self.task == "regression":
            if self.hparams.gnll:
                # Split outputs: mu1, sigma1_sq, mu2, sigma2_sq
                mu1, sigma1_sq, mu2, sigma2_sq = preds_all[:, 0], preds_all[:, 1], preds_all[:, 2], preds_all[:, 3]
                y1, y2 = data_batch[1][:, 0], data_batch[1][:, 1]  # Systolic, diastolic BP
                # Compute GNLL for each variable and sum
                loss1 = self.criterion(mu1, y1, sigma1_sq)
                loss2 = self.criterion(mu2, y2, sigma2_sq)
                loss = loss1 + loss2
            else:
                # Use MSE loss directly on predictions and targets
                loss = self.criterion(preds_all, data_batch[1])  # preds_all: [batch_size, 2], data_batch[1]: [batch_size, 2]
        else:
            # Classification case remains unchanged
            loss = self.criterion(preds_all, data_batch[1])  # Cross-entropy for classification
    
        self.log("train_loss" if train else ("test_loss" if test else "val_loss"), loss)
    
        if not train and not test:
            self.val_preds[dataloader_idx].append(preds_all.detach())
            self.val_targs[dataloader_idx].append(data_batch[1])
        elif not train and test:
            self.test_preds[dataloader_idx].append(preds_all.detach())
            self.test_targs[dataloader_idx].append(data_batch[1])
    
        return loss

    

    #def test_step(self, test_batch, batch_idx, dataloader_idx=0):
    #    return self._step(test_batch,batch_idx,train=False,test=True, dataloader_idx=dataloader_idx)
    def eval_scores(self, targs, preds, classes=None):
        if self.task == "regression":
            if self.hparams.gnll:
                # Extract mean predictions (mu1, mu2) for systolic and diastolic BP
                preds_means = preds[:, [0, 2]]  # Select mu1 (index 0) and mu2 (index 2)
            else:
                preds_means = preds  # Use preds directly if gnll=False
    
            maes = np.abs((targs - preds_means))  # Compute MAE using mean predictions
            mae = np.mean(maes, axis=0)
            # IEEE 1708a-2019 grades (fraction of cases)
            gradea = np.mean(maes <= 5.0, axis=0)
            gradeb = np.mean(np.logical_and(maes > 5.0, maes <= 6.0), axis=0)
            gradec = np.mean(np.logical_and(maes > 6.0, maes <= 7.0), axis=0)
            graded = np.mean(maes > 7.0, axis=0)
    
            return {
                "mae0": mae[0],
                "mae1": mae[1],
                "gradea0": gradea[0],
                "gradea1": gradea[1],
                "gradeb0": gradeb[0],
                "gradeb1": gradeb[1],
                "gradec0": gradec[0],
                "gradec1": gradec[1],
                "graded0": graded[0],
                "graded1": graded[1],
            }
        else:
            return {}

    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        if self.test_vars is None:
            self.test_vars = [[] for _ in range(len(self.test_datasets))]
        self.model.eval()  # Enable dropout during testing for MCD
        # Enable dropout layers specifically
        dropout_layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(module)
                module.train()  # Enable dropout for this layer

        x, y = test_batch
        num_samples = 11

        # Perform X forward passes with dropout enabled
        preds_list = []
        for _ in range(num_samples):
            preds = self.forward(x)  # [batch_size, 4] or [batch_size, 2]
            preds_list.append(preds)

        # Compute mean predictions across X samples
        preds_all = torch.stack(preds_list).mean(dim=0)  # [batch_size, num_classes]
        var_all = torch.stack(preds_list).std(dim=0)  # [batch_size, num_classes]

        # Compute loss using mean predictions
        if self.task == "regression" and self.hparams.gnll:
            mu1, sigma1_sq, mu2, sigma2_sq = preds_all[:, 0], preds_all[:, 1], preds_all[:, 2], preds_all[:, 3]
            y1, y2 = y[:, 0], y[:, 1]
            loss1 = self.criterion(mu1, y1, sigma1_sq)
            loss2 = self.criterion(mu2, y2, sigma2_sq)
            loss = loss1 + loss2
        else:
            loss = self.criterion(preds_all, y)

        self.log("test_loss", loss)
        self.test_preds[dataloader_idx].append(preds_all.detach())
        self.test_targs[dataloader_idx].append(y)
        self.test_vars[dataloader_idx].append(var_all.detach())

        return loss

    def on_test_epoch_end(self):
        cpu_test_vars = [[var.cpu() for var in vars_sublist] for vars_sublist in self.test_vars]
        # Save the CPU-based tensors to the pickle file
        with open('variances_test.pkl', 'wb') as f:
            pickle.dump(cpu_test_vars, f)
        #with open('variances_test.pkl', 'wb') as f: 
        #    pickle.dump(self.test_vars, f)
        for i in range(len(self.test_preds)):
            self.on_valtest_epoch_eval({"preds":self.test_preds[i], "targs":self.test_targs[i]}, dataloader_idx=i, test=True)
            self.test_preds[i].clear()
            self.test_targs[i].clear()
        
######################################################################################################
# MISC
######################################################################################################
def load_from_checkpoint(pl_model, checkpoint_path):
    """ load from checkpoint function that is compatible with S4
    """
    lightning_state_dict = torch.load(checkpoint_path)
    state_dict = lightning_state_dict["state_dict"]
    
    for name, param in pl_model.named_parameters():
        param.data = state_dict[name].data
    for name, param in pl_model.named_buffers():
        param.data = state_dict[name].data


    
#####################################################################################################
#ARGPARSER
#####################################################################################################
def add_model_specific_args(parser):
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--architecture", type=str, help="xresnet1d50/xresnet1d101/inception1d/s4/baseline/lenet1d", default="xresnet1d50")
    
    parser.add_argument("--s4-n", type=int, default=8, help='S4: N (Sashimi default:64)')
    parser.add_argument("--s4-h", type=int, default=512, help='S4: H (Sashimi default:64)')
    parser.add_argument("--s4-layers", type=int, default=4, help='S4: number of layers (Sashimi default:8)')
    parser.add_argument("--s4-batchnorm", action='store_true', help='S4: use BN instead of LN')
    parser.add_argument("--s4-prenorm", action='store_true', help='S4: use prenorm')
     
    return parser

def add_application_specific_args(parser):
    parser.add_argument("--num-training-folds", type=int, default=0,help="number of training folds (0 for all)")
    parser.add_argument("--num-validation-folds", type=int, default=0,help="number of validation folds (0 for all)")

    parser.add_argument("--normalize", action='store_true', help='Normalize input sequences individually to reach min=0 and max=1')
    parser.add_argument("--finetune-dataset", type=str, help="mimiciiibp/deepbeat/bpbenchmark_(|sensor|uci|bcg|ppgbp)/pulsedb_(calib|calibfree|aami)_(|vital|mimic)", default="mimiciiibp")
    parser.add_argument("--chunk-length-train", type=float, default=1.,help="training chunk length in multiples of input size")
    parser.add_argument("--stride-fraction-train", type=float, default=1.,help="training stride in multiples of input size")
    parser.add_argument("--stride-fraction-valtest", type=float, default=1.,help="val/test stride in multiples of input size")
    parser.add_argument("--chunkify-train", action='store_true')
    parser.add_argument("--gnll", action='store_true')

    parser.add_argument("--eval-only", type=str, help="path to model checkpoint for evaluation", default="")

    parser.add_argument('--select-input-channel', action='append', type=int, help='Select specific input channels (use multiple times for multiple channels)- by default all channels will be used')
    parser.add_argument("--disregard-splits", action='store_true',help="disregard dataset splits (to be used in conjunction with eval-only) for evaluation on entire datasets")
    parser.add_argument("--join-calib-val", action='store_true',help="join calibration and validation set")
    parser.add_argument("--debug", action='store_true', help="Enable debugging mode")
    
    return parser
            
###################################################################################################
#MAIN
###################################################################################################
if __name__ == '__main__':
    parser = add_default_args()
    parser = add_model_specific_args(parser)
    parser = add_application_specific_args(parser)

    hparams = parser.parse_args()
    hparams.executable = "main_ppg_mcd"
    hparams.revision = get_git_revision_short_hash()

    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)
        
    model = Main_PPG_MCD(hparams)

    logger = [TensorBoardLogger(
        save_dir=hparams.output_path,
        #version="",#hparams.metadata.split(":")[0],
        name="")]
    print("Output directory:",logger[0].log_dir)

    if(MLFLOW_AVAILABLE):
        mlflow.set_experiment(hparams.executable)
        run = mlflow.start_run(run_name=hparams.metadata)
        mlf_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(run.info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            log_model=False,
        )
        mlf_logger._run_id = run.info.run_id
        mlf_logger.log_hyperparams = log_params_from_namespace       
        logger.append(mlf_logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger[0].log_dir,
        filename="best_model",
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor= "auc_agg_val0" if hparams.finetune_dataset=="deepbeat" else 'mae0_agg_val0' ,#val_loss/dataloader_idx_0
        mode='max' if hparams.finetune_dataset=="deepbeat" else 'min')

    lr_monitor = LearningRateMonitor(logging_interval="step")
    #lr_monitor2 = LRMonitorCallback(start=False,end=True)#interval="step")

    callbacks = [checkpoint_callback,lr_monitor]#,lr_monitor2]

    if(hparams.refresh_rate>0):
        callbacks.append(TQDMProgressBar(refresh_rate=hparams.refresh_rate))

    trainer = lp.Trainer(
        num_sanity_val_steps=0,#no debugging
        #overfit_batches=50,#debugging

        accumulate_grad_batches=hparams.accumulate,
        max_epochs=hparams.epochs,
        min_epochs=hparams.epochs,
        
        default_root_dir=hparams.output_path,
        
        logger=logger,
        callbacks = callbacks,
        benchmark=True,
    
        accelerator="gpu" if hparams.gpus>0 else "cpu",
        devices=hparams.gpus if hparams.gpus>0 else 1,
        num_nodes=hparams.num_nodes,
        precision=hparams.precision,
        #distributed_backend=hparams.distributed_backend,
        
        enable_progress_bar=hparams.refresh_rate>0)
        
    if(hparams.auto_batch_size):#auto tune batch size batch size
        tuner=Tuner(trainer)
        tuner.scale_batch_size(model, mode="binsearch")

    if(hparams.lr_find):# lr find
        tuner=Tuner(trainer)
        lr_finder = tuner.lr_find(model)

    if(hparams.epochs>0 and hparams.eval_only==""):
        trainer.fit(model,ckpt_path= None if hparams.resume=="" else hparams.resume)
        trainer.test(model,ckpt_path="best")

    elif(hparams.eval_only!=""):#eval only
        trainer.test(model,ckpt_path=hparams.eval_only)
    
    if(MLFLOW_AVAILABLE):
        mlflow.end_run()