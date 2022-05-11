import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import torch_geometric.nn as pyg_nn 
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Dataset, Data, DataLoader
import torch.optim as optim
import os.path as osp
import scipy.io as sio
from datetime import datetime
from tensorboardX import SummaryWriter
from dataset import QUASARDataset
from model import ModelS
from torch.optim.lr_scheduler import MultiStepLR

GNN_TYPE = 'SAGE'
GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = GNN_HIDDEN_DIM
GNN_LAYER = 7
NODE_MODE = 1
DATA_GRAPH_TYPE = 1
NUM_EPOCHES = 800
DROPOUT = 0.1
MLP_LAYER = 2
RESIDUAL = True
LOSSPOW = 1

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1')

def train(dataset,writer,batch_size,num_epoches,schedule=False):
    # build model
    model   = ModelS(node_feature_mode=1,
                     gnn_type=GNN_TYPE,
                     mp_hidden_dim=GNN_HIDDEN_DIM,mp_output_dim=GNN_OUT_DIM,mp_num_layers=GNN_LAYER, 
                     primal_node_mlp_hidden_dim=GNN_OUT_DIM,primal_node_mlp_output_dim=10,
                     dual_node_mlp_hidden_dim=GNN_OUT_DIM,dual_node_mlp_output_dim=10,
                     primal_edge_mlp_hidden_dim=GNN_OUT_DIM,primal_edge_mlp_output_dim=10, 
                     dual_edge_mlp_hidden_dim=GNN_OUT_DIM,dual_edge_mlp_output_dim=6,
                     primal_mlp_num_layers=MLP_LAYER, 
                     dual_mlp_num_layers=MLP_LAYER, 
                     dropout_rate=DROPOUT,
                     relu_slope=0.1,
                     residual=RESIDUAL,
                     losspow=LOSSPOW)
    print(model)
    model.double() # convert all parameters to double
    model.to(device)
    opt = optim.Adam(model.parameters(),lr=0.01)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    if schedule:
        scheduler = MultiStepLR(opt,milestones=[int(0.5*num_epoches),int(0.75*num_epoches)],gamma=0.1)

    # train
    for epoch in range(num_epoches):
        total_loss = 0
        total_primal_loss = 0
        total_dual_loss = 0

        model.train()
        for batch in loader:
            opt.zero_grad()
            batch.to(device)

            _, X, S, Aty = model(batch)
            primal_loss, dual_loss = model.pd_loss(batch,X,S,Aty)
            loss = primal_loss + dual_loss
            loss.backward()
            opt.step()
            # print('batch loss: {:.4f}, primal: {:.4f}, dual: {:.4f}'.format(
                # loss.item(),primal_loss.item(),dual_loss.item()))

            total_loss += loss.item() * batch.num_graphs
            total_primal_loss += primal_loss.item() * batch.num_graphs
            total_dual_loss += dual_loss.item() * batch.num_graphs
        
        if schedule:
            scheduler.step()

        total_loss /= len(loader.dataset)
        total_primal_loss /= len(loader.dataset)
        total_dual_loss /= len(loader.dataset)
        print("Epoch {}. Loss: {:.4f}, Primal: {:.4f}, Dual: {:.4f}.".format(
            epoch, total_loss, total_primal_loss, total_dual_loss))

        writer.add_scalar("loss", total_loss, epoch)
        writer.add_scalar("primal loss", total_primal_loss, epoch)
        writer.add_scalar("dual loss", total_dual_loss, epoch)
        
    return model


# setname = 'N30-1000'
setname = 'small'
if setname == 'N30-1000':
    num_graphs = 1000
elif setname == 'small':
    num_graphs = 100

dir = f'/home/hank/Datasets/QUASAR/{setname}'
batch_size = 50
dataset = QUASARDataset(dir,num_graphs=num_graphs,remove_self_loops=True,graph_type=DATA_GRAPH_TYPE)
writer = SummaryWriter("./log/" + setname + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

modelname = f'./models/shared_model_{setname}_{GNN_TYPE}_{GNN_HIDDEN_DIM}_{GNN_LAYER}_{RESIDUAL}'
model = train(dataset,writer,batch_size,NUM_EPOCHES)

filename = f'{modelname}.pth'
torch.save(model.state_dict(),filename)