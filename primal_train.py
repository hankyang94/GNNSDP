import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from primal_model import PrimalModel 
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
MLP_LAYER = 2

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')

def train(dataset,writer,batch_size,num_epoches):
    # build model
    model   = PrimalModel(node_feature_mode=NODE_MODE,
                     gnn_type=GNN_TYPE,
                     mp_hidden_dim=GNN_HIDDEN_DIM,mp_output_dim=GNN_OUT_DIM,mp_num_layers=GNN_LAYER, 
                     primal_node_mlp_hidden_dim=64,primal_node_mlp_output_dim=10,
                     node_mlp_num_layers=MLP_LAYER,
                     primal_edge_mlp_hidden_dim=64,primal_edge_mlp_output_dim=10,
                     edge_mlp_num_layers=MLP_LAYER, 
                     dropout_rate=DROPOUT,
                     relu_slope=0.1)
    print(model)
    model.double() # convert all parameters to double
    model.load_state_dict(torch.load('./models/primal_model_SAGE_4_64_64_1_1_1000_0.0_2.pth'))
    model.to(device)
    opt = optim.Adam(model.parameters(),lr=0.01)
    # opt = optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-4)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    # train
    for epoch in range(num_epoches):
        total_primal_loss = 0

        model.train()
        for batch in loader:
            opt.zero_grad()
            batch.to(device)

            x, X = model(batch)
            primal_loss = model.loss(batch,X)
            primal_loss.backward()
            opt.step()
            print('batch loss: {:.4f}.'.format(primal_loss.item()))
            total_primal_loss += primal_loss.item() * batch.num_graphs
            
        total_primal_loss /= len(loader.dataset)
        print("Epoch {}. Loss: {:.4f}.".format(
            epoch, total_primal_loss))

        writer.add_scalar("primal loss", total_primal_loss, epoch)
        
    return model

dir = '/home/hank/Datasets/QUASAR/small'
batch_size = 50
dataset = QUASARDataset(dir,num_graphs=100,remove_self_loops=True,graph_type=DATA_GRAPH_TYPE)
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

model = train(dataset,writer,batch_size,NUM_EPOCHES)

filename = f'./models/primal_model_{GNN_TYPE}_{GNN_LAYER}_{GNN_HIDDEN_DIM}_{GNN_OUT_DIM}_{NODE_MODE}_{DATA_GRAPH_TYPE}_{NUM_EPOCHES}_{DROPOUT*100}_{MLP_LAYER}_renew.pth'
torch.save(model.state_dict(),filename)