import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from dual_model import DualModel 
import torch_geometric.nn as pyg_nn 
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Dataset, Data, DataLoader
import torch.optim as optim
import os.path as osp
import scipy.io as sio
from datetime import datetime
from tensorboardX import SummaryWriter
from dataset import QUASARDataset

GNN_TYPE = 'SAGE'
GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = 64
GNN_LAYER = 4
NODE_MODE = 1
DATA_GRAPH_TYPE = 1
NUM_EPOCHES = 1000
DROPOUT = 0.0
MLP_LAYER = 1

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1')

def train(dataset,writer,batch_size,num_epoches):
    # build model
    model   = DualModel(node_feature_mode=NODE_MODE,
                     gnn_type=GNN_TYPE,
                     mp_hidden_dim=GNN_HIDDEN_DIM,mp_output_dim=GNN_OUT_DIM,mp_num_layers=GNN_LAYER, 
                     dual_node_mlp_hidden_dim=64,dual_node_mlp_output_dim=10,
                     node_mlp_num_layers=MLP_LAYER,
                     dual_edge_mlp_hidden_dim=64,dual_edge_mlp_output_dim=6,
                     edge_mlp_num_layers=MLP_LAYER, 
                     dropout_rate=DROPOUT,
                     relu_slope=0.1)
    print(model)
    model.double() # convert all parameters to double
    model.to(device)
    opt = optim.Adam(model.parameters(),lr=0.01)
    # opt = optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-4)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    # train
    for epoch in range(num_epoches):
        total_dual_loss = 0

        model.train()
        for batch in loader:
            opt.zero_grad()
            batch.to(device)

            x, S, Aty = model(batch)
            dual_loss = model.loss(batch,S,Aty)
            dual_loss.backward()
            opt.step()
            print('batch loss: {:.4f}.'.format(dual_loss.item()))
            total_dual_loss += dual_loss.item() * batch.num_graphs
            
        total_dual_loss /= len(loader.dataset)
        print("Epoch {}. Loss: {:.4f}.".format(
            epoch, total_dual_loss))

        writer.add_scalar("dual loss", total_dual_loss, epoch)
        
    return model

dir = '/home/hank/Datasets/QUASAR/small'
batch_size = 50
dataset = QUASARDataset(dir,num_graphs=100,remove_self_loops=True,graph_type=DATA_GRAPH_TYPE)
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

model = train(dataset,writer,batch_size,NUM_EPOCHES)

filename = f'./models/dual_model_{GNN_TYPE}_{GNN_LAYER}_{GNN_HIDDEN_DIM}_{GNN_OUT_DIM}_{NODE_MODE}_{DATA_GRAPH_TYPE}_{NUM_EPOCHES}_{DROPOUT*100}_{MLP_LAYER}.pth'
torch.save(model.state_dict(),filename)