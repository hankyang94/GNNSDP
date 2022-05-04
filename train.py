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

def train(dataset,writer,batch_size,num_epoches):
    # build model
    model   = ModelS(node_feature_mode=2,
                     mp_hidden_dim=64,mp_output_dim=64,mp_num_layers=3, 
                     primal_node_mlp_hidden_dim=64,primal_node_mlp_output_dim=10,
                     dual_node_mlp_hidden_dim=64,dual_node_mlp_output_dim=10,
                     node_mlp_num_layers=2,
                     primal_edge_mlp_hidden_dim=64,primal_edge_mlp_output_dim=10, 
                     dual_edge_mlp_hidden_dim=64,dual_edge_mlp_output_dim=6, 
                     edge_mlp_num_layers=2, 
                     dropout_rate=0.2)
    model.double() # convert all parameters to double
    model.to(device)
    opt = optim.Adam(model.parameters(),lr=0.01)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    # train
    for epoch in range(num_epoches):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            batch.to(device)
            X, S, Aty = model(batch)
            loss = model.loss(batch,X,S,Aty)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            print('graph {}. loss: {:.4f}.'.format(i,loss.item()))
        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)
        print("Epoch {}. Loss: {:.4f}.".format(epoch, total_loss))
    return model

dir = '/home/hank/Datasets/QUASAR/small'
batch_size = 30
num_epoches = 20
dataset = QUASARDataset(dir,num_graphs=100,remove_self_loops=True)
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = train(dataset,writer,batch_size,num_epoches)
torch.save(model.state_dict(),'./models/model.pth')