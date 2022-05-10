from http.client import ImproperConnectionState
from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from primal_model import PrimalModel 
import torch_geometric.nn as pyg_nn 
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Dataset, Data, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import os.path as osp
import scipy.io as sio
from datetime import datetime
from tensorboardX import SummaryWriter
from dataset import QUASARDataset

GNN_TYPE = 'SAGE'
GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = 64
GNN_LAYER = 6
NODE_MODE = 1
DATA_GRAPH_TYPE = 1
NUM_EPOCHES = 800
DROPOUT = 0.0
MLP_LAYER = 2
RESIDUAL = True
BATCHNORM = False

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:2')

def train(dataset,writer,batch_size,num_epoches,modelname=None):
    model   = PrimalModel(node_feature_mode=NODE_MODE,
                     gnn_type=GNN_TYPE,
                     mp_hidden_dim=GNN_HIDDEN_DIM,mp_output_dim=GNN_OUT_DIM,mp_num_layers=GNN_LAYER, 
                     primal_node_mlp_hidden_dim=GNN_OUT_DIM,primal_node_mlp_output_dim=10,
                     node_mlp_num_layers=MLP_LAYER,
                     primal_edge_mlp_hidden_dim=GNN_OUT_DIM,primal_edge_mlp_output_dim=10,
                     edge_mlp_num_layers=MLP_LAYER,
                     dropout_rate=DROPOUT,
                     relu_slope=0.1,
                     residual=RESIDUAL,
                     batchnorm=BATCHNORM)
    print(model)
    model.double() # convert all parameters to double
    model.to(device)
    opt = optim.Adam(model.parameters(),lr=0.01)
    scheduler = MultiStepLR(opt,milestones=[int(0.5*num_epoches),int(0.75*num_epoches)],gamma=0.1)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    # train
    for epoch in range(num_epoches):
        total_primal_loss = 0

        model.train()
        for batch in loader:
            opt.zero_grad()
            batch.to(device)

            _, X = model(batch)
            primal_loss = model.loss(batch,X)
            primal_loss.backward()
            opt.step()
            # print('batch loss: {:.4f}.'.format(primal_loss.item()))
            total_primal_loss += primal_loss.item() * batch.num_graphs
        
        scheduler.step()

        total_primal_loss /= len(loader.dataset)
        print("Epoch {}. Loss: {:.4f}.".format(epoch, total_primal_loss))

        writer.add_scalar("primal loss", total_primal_loss, epoch)

        if (modelname is not None) and (epoch > 199) and (epoch % 199 == 1):
            filename = f'{modelname}_{epoch}.pth'
            torch.save(model.state_dict(),filename)
    return model

setname = 'N30-1000'
# setname = 'small'
if setname == 'N30-1000':
    num_graphs = 1000
elif setname == 'small':
    num_graphs = 100

dir = f'/home/hank/Datasets/QUASAR/{setname}'
batch_size = 50
dataset = QUASARDataset(dir,num_graphs=num_graphs,remove_self_loops=True,graph_type=DATA_GRAPH_TYPE)
writer = SummaryWriter("./log/" + setname + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

modelname = f'./models/primal_model_{setname}_{GNN_TYPE}_{GNN_LAYER}_{RESIDUAL}_{BATCHNORM}'
model = train(dataset,writer,batch_size,NUM_EPOCHES)

filename = f'{modelname}.pth'
torch.save(model.state_dict(),filename)