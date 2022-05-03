import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch_geometric.nn as pyg_nn 
import torch_geometric.utils as pyg_utils
from dataset import QUASARDataset

# Supervised GNN model
class ModelS(nn.Module):
    def __init__(self, mp_input_dim=6,
                       mp_hidden_dim=32,
                       mp_output_dim=64,
                       mp_num_layers=1, 
                       primal_node_mlp_hidden_dim=64,
                       primal_node_mlp_output_dim=10,
                       dual_node_mlp_hidden_dim=64,
                       dual_node_mlp_output_dim=10,
                       node_mlp_num_layers=1,
                       primal_edge_mlp_hidden_dim=64, 
                       primal_edge_mlp_output_dim=10, 
                       dual_edge_mlp_hidden_dim=64, 
                       dual_edge_mlp_output_dim=16, 
                       edge_mlp_num_layers=1, 
                       dropout_rate=0.2):
        super(ModelS,self).__init__()
        # Message passing
        self.mp_convs = nn.ModuleList()
        self.mp_convs.append(pyg_nn.SAGEConv(mp_input_dim,mp_hidden_dim))
        for i in range(mp_num_layers):
            self.mp_convs.append(pyg_nn.SAGEConv(mp_hidden_dim,mp_hidden_dim))
        self.mp_convs.append(pyg_nn.SAGEConv(mp_hidden_dim,mp_output_dim))

        # Post message passing
        # Primal node MLP
        self.primal_node_mlp = nn.ModuleList()
        self.primal_node_mlp.append(
            nn.Linear(mp_output_dim,primal_node_mlp_hidden_dim,dtype=torch.float64))
        for i in range(node_mlp_num_layers):
            self.primal_node_mlp.append(
                nn.Linear(primal_node_mlp_hidden_dim,primal_node_mlp_hidden_dim,dtype=torch.float64))
        self.primal_node_mlp.append(
            nn.Linear(primal_node_mlp_hidden_dim,primal_node_mlp_output_dim,dtype=torch.float64))
        # Dual node MLP
        self.dual_node_mlp = nn.ModuleList()
        self.dual_node_mlp.append(
            nn.Linear(mp_output_dim,dual_node_mlp_hidden_dim,dtype=torch.float64))
        for i in range(node_mlp_num_layers):
            self.dual_node_mlp.append(
                nn.Linear(dual_node_mlp_hidden_dim,dual_node_mlp_hidden_dim,dtype=torch.float64))
        self.dual_node_mlp.append(
            nn.Linear(dual_node_mlp_hidden_dim,dual_node_mlp_output_dim,dtype=torch.float64))
        # Primal edge MLP
        self.primal_edge_mlp = nn.ModuleList()
        self.primal_edge_mlp.append(
            nn.Linear(mp_output_dim,primal_edge_mlp_hidden_dim,dtype=torch.float64))
        for i in range(edge_mlp_num_layers):
            self.primal_edge_mlp.append(
                nn.Linear(primal_edge_mlp_hidden_dim,primal_edge_mlp_hidden_dim,dtype=torch.float64))
        self.primal_edge_mlp.append(
            nn.Linear(primal_edge_mlp_hidden_dim,primal_edge_mlp_output_dim,dtype=torch.float64))
        # Dual edge MLP
        self.dual_edge_mlp = nn.ModuleList()
        self.dual_edge_mlp.append(
            nn.Linear(mp_output_dim,dual_edge_mlp_hidden_dim,dtype=torch.float64))
        for i in range(edge_mlp_num_layers):
            self.dual_edge_mlp.append(
                nn.Linear(dual_edge_mlp_hidden_dim,dual_edge_mlp_hidden_dim,dtype=torch.float64))
        self.dual_edge_mlp.append(
            nn.Linear(dual_edge_mlp_hidden_dim,dual_edge_mlp_output_dim,dtype=torch.float64))
        self.dropout_rate = dropout_rate
        self.dual_edge_mlp_output_dim = dual_edge_mlp_output_dim
        if self.dual_edge_mlp_output_dim != 16 and self.dual_edge_mlp_output_dim != 6:
            raise RuntimeError('dual_edge_mlp_output_dim must be 16 or 6.')

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = data.num_nodes
        ud_edges  = data.ud_edges
        edge_map  = data.edge_map
        # Message passing
        for mp_layer in self.mp_convs:
            x = mp_layer(x,edge_index)
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout_rate,training=self.training)
        
        # Post message passing
        # Primal node
        vp = []
        for i in range(num_nodes):
            xi = x[i,:] # feature of i-th node
            for mlp_layer in self.primal_node_mlp:
                xi = mlp_layer(xi)
                xi = F.relu(xi)
                xi = F.dropout(xi,p=self.dropout_rate,training=self.training)
            vp.append(xi)
        vp = torch.stack(vp) # num_nodes x primal_node_mlp_output_dim
        # Dual node
        vd = []
        for i in range(num_nodes):
            xi = x[i,:]
            for mlp_layer in self.dual_node_mlp:
                xi = mlp_layer(xi)
                xi = F.relu(xi)
                xi = F.dropout(xi,p=self.dropout_rate,training=self.training)
            vd.append(xi)
        vd = torch.stack(vd) # num_nodes x dual_node_mlp_output_dim
        # Primal edge
        ep = []
        for edge in ud_edges:
            xi  = x[edge[0],:]
            xj  = x[edge[1],:]
            xij = xi + xj
            for mlp_layer in self.primal_edge_mlp:
                xij = mlp_layer(xij)
                xij = F.relu(xij)
                xij = F.dropout(xij,p=self.dropout_rate,training=self.training)
            ep.append(xij)
        ep = torch.stack(ep)
        # Dual edge
        ed = []
        for edge in ud_edges:
            xi  = x[edge[0],:]
            xj  = x[edge[1],:]
            xij = xi + xj
            for mlp_layer in self.dual_edge_mlp:
                xij = mlp_layer(xij)
                xij = F.relu(xij)
                xij = F.dropout(xij,p=self.dropout_rate,training=self.training)
            ed.append(xij)
        ed = torch.stack(ed)

        # Recover primal X
        X = self.recover_X(vp,ep,edge_map)
        # Recover dual S or Aty
        if self.dual_edge_mlp_output_dim == 16:
            S = self.recover_S(vd,ed,edge_map)
            Aty = None 
        elif self.dual_edge_mlp_output_dim == 6:
            Aty = self.recover_Aty(vd,ed,edge_map)
            S = None
        else:
            raise RuntimeError(f'dual_edge_mlp_output_dim = {dual_edge_mlp_output_dim} not supported.')

        return x, X, S, Aty

    def smat(self,x):
        # X = torch.tensor([[x[0],x[1],x[2],x[3]],
        #                   [x[1],x[4],x[5],x[6]], 
        #                   [x[2],x[5],x[7],x[8]], 
        #                   [x[3],x[6],x[8],x[9]]],dtype=torch.float64)
        if x.size(dim=0) != 10:
            raise RuntimeError('smat only accepts 10-D vector.')
        idx = torch.tensor([0,1,2,3,1,4,5,6,2,5,7,8,3,6,8,9],device=x.device)
        X   = x[idx].view(4,4) 
        return X

    def mat(self,x,n):
        if x.size(dim=0) != n**2:
            raise RuntimeError(f'mat only accepts {n**2}-D vector.')
        return x.view(n,n)

    def skewmat(self,x):
        if x.size(dim=0) != 6:
            raise RuntimeError(f'skewmat only accepts 6-D vector.')
        # X = torch.tensor([[0,x[0],x[1],x[2]],
        #                   [-x[0],0,x[3],x[4]], 
        #                   [-x[1],-x[3],0,x[5]], 
        #                   [-x[2],-x[4],-x[5],0]],dtype=torch.float64)
        zero = torch.tensor([0],dtype=torch.float64,device=x.device)
        nega = torch.tensor([-1],dtype=torch.float64,device=x.device)
        vec  = torch.cat([zero, 
                          nega*x[0:3],
                          x[0:1], 
                          zero, 
                          nega*x[3:5], 
                          x[1:2], 
                          x[3:4], 
                          zero, 
                          nega*x[5:], 
                          x[2:3], 
                          x[4:5], 
                          x[5:], 
                          zero])
        return vec.view(4,4)

    def recover_X(self,vp,ep,edge_map):
        N = vp.shape[0] # number of nodes
        rows = []
        for i in range(N):
            row = []
            for j in range(N):
                if i == j: # diagonal blocks, using node features vp
                    blk = self.smat(vp[i,:])
                else: # off-diagonal blocks, using edge features ep
                    edge_id = edge_map[i,j]
                    blk = self.smat(ep[edge_id,:])
                row.append(blk)
            row_mat = torch.cat(row,dim=1)
            rows.append(row_mat)
        X = torch.cat(rows,dim=0)
        return X

    def recover_S(self,vd,ed,edge_map):
        N = vd.shape[0] # number of nodes
        rows = []
        for i in range(N):
            row = []
            for j in range(N):
                if i == j:
                    blk = self.smat(vd[i,:])
                else:
                    edge_id = edge_map[i,j]
                    tmp = self.mat(ed[edge_id,:],4)
                    if i < j: # upper tri
                        blk = tmp
                    else: # lower tri
                        blk = tmp.t()
                row.append(blk)
            row_mat = torch.cat(row,dim=1)
            rows.append(row_mat)
        S = torch.cat(rows,dim=0)
        return S

    def recover_Aty(self,vd,ed,edge_map):
        N = vd.shape[0]
        rows = []
        for i in range(N):
            row = []
            for j in range(N):
                if i == j:
                    blk = self.smat(vd[i,:])
                else:
                    edge_id = edge_map[i,j]
                    tmp = self.skewmat(ed[edge_id,:])
                    if i < j:
                        blk = tmp
                    else:
                        blk = tmp.t()
                row.append(blk)
            row_mat = torch.cat(row,dim=1)
            rows.append(row_mat)
        Aty = torch.cat(rows,dim=0)
        return Aty

    def loss(self,data,X,S,Aty):
        y = data.y
        if self.dual_edge_mlp_output_dim == 16:
            loss = torch.norm(X - y[0,:,:],p='fro') + torch.norm(S - y[1,:,:],p='fro')
        elif self.dual_edge_mlp_output_dim == 6:
            loss = torch.norm(X - y[0,:,:],p='fro') + torch.norm(Aty - y[2,:,:],p='fro')
        return loss