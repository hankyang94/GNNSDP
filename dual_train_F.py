import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from dual_model import DualModelFtype 
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

def train(model,trainset,validateset,
          writer,batch_size,num_epoches,
          learning_rate,device,
          modelname=None,schedule=False):

    model.double()
    model.to(device)
    opt = optim.Adam(model.parameters(),lr=learning_rate)
    if schedule:
        scheduler = MultiStepLR(opt,milestones=[int(0.5*num_epoches),int(0.75*num_epoches)],gamma=0.1)

    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
    # train
    for epoch in range(num_epoches):
        total_loss = 0
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            batch.to(device)

            _, V, E = model(batch)
            loss = model.loss(batch,V,E)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        if schedule:
            scheduler.step()
        total_loss /= len(train_loader.dataset)
        # validate
        validate_loss = validate(model,validateset,device)

        print("Epoch {}. Train loss: {:.4f}. Validate loss: {:.4f}.".format(
            epoch, total_loss, validate_loss))

        writer.add_scalar("train loss", total_loss, epoch)
        writer.add_scalar("validate loss", validate_loss, epoch)

        if (modelname is not None) and (epoch % 200 == 1) and (epoch > 1):
            filename = f'{modelname}_{epoch}.pth'
            torch.save(model.state_dict(),filename)
    return model

def validate(model,dataset,device,batch_size=1):
    with torch.no_grad():
        model.eval()
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        total_loss = 0
        for batch in loader:
            batch.to(device)
            _, V, E = model(batch)
            loss = model.loss(batch,V,E)
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
    return total_loss

if __name__ == "__main__":
    GNN_TYPE = 'SAGE'
    GNN_HIDDEN_DIM = 64
    GNN_OUT_DIM = GNN_HIDDEN_DIM
    GNN_LAYER = 15
    LR = 0.005
    NODE_MODE = 1
    DATA_GRAPH_TYPE = 1
    NUM_EPOCHES = 2000
    DROPOUT = 0.2
    MLP_LAYER = 2
    DUAL_OUT_DIM = 6
    RESIDUAL = True
    BATCHNORM = True

    DEVICE = torch.device('cuda:0')

    trainsetname = 'N50-5000'
    NUM_GRAPHS = 5000
    trainmatversion = 2 # use h5py to read large .mat file
    validatesetname = 'N50-100'

    train_dir = f'/home/hank/Datasets/QUASAR/{trainsetname}'
    validate_dir = f'/home/hank/Datasets/QUASAR/{validatesetname}'
    batch_size = 50
    trainset = QUASARDataset(train_dir,num_graphs=NUM_GRAPHS,remove_self_loops=True,graph_type=DATA_GRAPH_TYPE,matversion=trainmatversion)
    validateset = QUASARDataset(validate_dir,num_graphs=100,remove_self_loops=True,graph_type=DATA_GRAPH_TYPE)
    writer = SummaryWriter("./log/" + trainsetname + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    modelname = f'./models/dual_model_{trainsetname}_{GNN_TYPE}_{GNN_LAYER}_{GNN_HIDDEN_DIM}_{MLP_LAYER}_{DUAL_OUT_DIM}_{RESIDUAL}_{BATCHNORM}_{NODE_MODE}'

    model = DualModelFtype(
        node_feature_mode=NODE_MODE,
        gnn_type=GNN_TYPE,
        mp_hidden_dim=GNN_HIDDEN_DIM,mp_output_dim=GNN_OUT_DIM,mp_num_layers=GNN_LAYER, 
        dual_node_mlp_hidden_dim=GNN_OUT_DIM,dual_node_mlp_output_dim=10,
        node_mlp_num_layers=MLP_LAYER,
        dual_edge_mlp_hidden_dim=GNN_OUT_DIM,dual_edge_mlp_output_dim=DUAL_OUT_DIM,
        edge_mlp_num_layers=MLP_LAYER,
        dropout_rate=DROPOUT,
        relu_slope=0.1,
        residual=RESIDUAL,
        batchnorm=BATCHNORM)
    print(model)
    # model.load_state_dict(torch.load('./models/dual_model_N30-3000_SAGE_31_64_2_6_True_True_stage_1.pth'))

    model = train(model,trainset,validateset,
                 writer,batch_size,num_epoches=NUM_EPOCHES,
                 learning_rate=LR,device=DEVICE,
                 modelname=modelname,schedule=False)
    filename = f'{modelname}.pth'
    torch.save(model.state_dict(),filename)