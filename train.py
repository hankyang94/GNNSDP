from nbformat import write
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

GNN_TYPE = 'GATConv'
GNN_HIDDEN_DIM = 128
GNN_OUT_DIM = 128
GNN_LAYER = 8

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:3')

def train(dataset,writer,batch_size,num_epoches):
    # build model
    model   = ModelS(node_feature_mode=3,
                     gnn_type=GNN_TYPE,
                     mp_hidden_dim=GNN_HIDDEN_DIM,mp_output_dim=GNN_OUT_DIM,mp_num_layers=GNN_LAYER, 
                     primal_node_mlp_hidden_dim=64,primal_node_mlp_output_dim=10,
                     dual_node_mlp_hidden_dim=64,dual_node_mlp_output_dim=10,
                     node_mlp_num_layers=0,
                     primal_edge_mlp_hidden_dim=64,primal_edge_mlp_output_dim=10, 
                     dual_edge_mlp_hidden_dim=64,dual_edge_mlp_output_dim=6, 
                     edge_mlp_num_layers=0, 
                     dropout_rate=0.0,
                     relu_slope=0.1)
    print(model)
    model.double() # convert all parameters to double
    model.to(device)
    opt = optim.Adam(model.parameters(),lr=0.01)
    # opt = optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-4)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    # train
    for epoch in range(num_epoches):
        total_loss = 0
        total_primal_loss = 0
        total_dual_loss = 0

        model.train()
        for batch in loader:
            opt.zero_grad()
            batch.to(device)

            x, X, S, Aty = model(batch)
            primal_loss, dual_loss = model.loss(batch,X,S,Aty)
            loss = primal_loss + dual_loss
            loss.backward()
            opt.step()
            print('batch loss: {:.4f}, primal: {:.4f}, dual: {:.4f}'.format(
                loss.item(),primal_loss.item(),dual_loss.item()))

            total_loss += loss.item() * batch.num_graphs
            total_primal_loss += primal_loss.item() * batch.num_graphs
            total_dual_loss += dual_loss.item() * batch.num_graphs
            
        total_loss /= len(loader.dataset)
        total_primal_loss /= len(loader.dataset)
        total_dual_loss /= len(loader.dataset)
        print("Epoch {}. Loss: {:.4f}, Primal: {:.4f}, Dual: {:.4f}.".format(
            epoch, total_loss, total_primal_loss, total_dual_loss))

        writer.add_scalar("loss", total_loss, epoch)
        writer.add_scalar("primal loss", total_primal_loss, epoch)
        writer.add_scalar("dual loss", total_dual_loss, epoch)
        
    return model

dir = '/home/hank/Datasets/QUASAR/small'
batch_size = 50
num_epoches = 200
dataset = QUASARDataset(dir,num_graphs=100,remove_self_loops=True)
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

model = train(dataset,writer,batch_size,num_epoches)

filename = f'./models/model_{GNN_TYPE}_{GNN_LAYER}_{GNN_HIDDEN_DIM}_{GNN_OUT_DIM}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pth'
torch.save(model.state_dict(),filename)