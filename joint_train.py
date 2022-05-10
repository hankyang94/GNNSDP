import torch
from dual_model import DualModel 
from primal_model import PrimalModel
import torch_geometric.nn as pyg_nn 
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Dataset, Data, DataLoader
import torch.optim as optim
import os.path as osp
from datetime import datetime
from tensorboardX import SummaryWriter
from dataset import QUASARDataset

GNN_TYPE = 'SAGE'
GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = 64
GNN_LAYER = 4
NODE_MODE = 1
DATA_GRAPH_TYPE = 1
NUM_EPOCHES = 2000
DROPOUT = 0.0
MLP_LAYER = 2

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:2')

def train(dataset,writer,batch_size,num_epoches):
    pmodel   = PrimalModel(node_feature_mode=NODE_MODE,
                     gnn_type=GNN_TYPE,
                     mp_hidden_dim=GNN_HIDDEN_DIM,mp_output_dim=GNN_OUT_DIM,mp_num_layers=GNN_LAYER, 
                     primal_node_mlp_hidden_dim=64,primal_node_mlp_output_dim=10,
                     node_mlp_num_layers=MLP_LAYER,
                     primal_edge_mlp_hidden_dim=64,primal_edge_mlp_output_dim=10,
                     edge_mlp_num_layers=MLP_LAYER, 
                     dropout_rate=DROPOUT,
                     relu_slope=0.1)

    dmodel   = DualModel(node_feature_mode=NODE_MODE,
                     gnn_type=GNN_TYPE,
                     mp_hidden_dim=GNN_HIDDEN_DIM,mp_output_dim=GNN_OUT_DIM,mp_num_layers=GNN_LAYER, 
                     dual_node_mlp_hidden_dim=64,dual_node_mlp_output_dim=10,
                     node_mlp_num_layers=MLP_LAYER,
                     dual_edge_mlp_hidden_dim=64,dual_edge_mlp_output_dim=6,
                     edge_mlp_num_layers=MLP_LAYER, 
                     dropout_rate=DROPOUT,
                     relu_slope=0.1)
    pmodel.double()
    dmodel.double()
    pmodel.to(device)
    dmodel.to(device)
    popt = optim.Adam(pmodel.parameters(),lr=0.01)
    dopt = optim.Adam(dmodel.parameters(),lr=0.01)
    
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    # train
    for epoch in range(num_epoches):
        total_primal_loss = 0
        total_dual_loss = 0
        pmodel.train()
        dmodel.train()
        for batch in loader:
            popt.zero_grad()
            dopt.zero_grad()
            batch.to(device)

            _, X = pmodel(batch)
            _, S, Aty = dmodel(batch)

            primal_loss = pmodel.loss(batch,X)
            dual_loss = dmodel.loss(batch,S,Aty)
            primal_loss.backward()
            dual_loss.backward()
            popt.step()
            dopt.step()

            total_primal_loss += primal_loss.item() * batch.num_graphs
            total_dual_loss += dual_loss.item() * batch.num_graphs

        total_primal_loss /= len(loader.dataset)    
        total_dual_loss /= len(loader.dataset)
        print("Epoch {}. Loss: {:.4f}. Primal: {:.4f}. Dual: {:.4f}.".format(
            epoch, total_primal_loss + total_dual_loss, total_primal_loss, total_dual_loss))

        writer.add_scalar("loss", total_primal_loss + total_dual_loss, epoch)
        writer.add_scalar("primal loss", total_primal_loss, epoch)
        writer.add_scalar("dual loss", total_dual_loss, epoch)
        
    return pmodel, dmodel

dir = '/home/hank/Datasets/QUASAR/small'
batch_size = 50
dataset = QUASARDataset(dir,num_graphs=100,remove_self_loops=True,graph_type=DATA_GRAPH_TYPE)
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

pmodel, dmodel = train(dataset,writer,batch_size,NUM_EPOCHES)

pfname = "./models/p_model_{}_{}_{}_{}.pth".format(
    GNN_TYPE,GNN_LAYER,MLP_LAYER,NUM_EPOCHES)
dfname = "./models/d_model_{}_{}_{}_{}.pth".format(
    GNN_TYPE,GNN_LAYER,MLP_LAYER,NUM_EPOCHES)
    
torch.save(pmodel.state_dict(),pfname)
torch.save(dmodel.state_dict(),dfname)