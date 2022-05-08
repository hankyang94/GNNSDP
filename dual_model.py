import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch_geometric.nn as pyg_nn 
import torch_geometric.utils as pyg_utils

# Supervised Dual GNN model
class DualModel(nn.Module):
    def __init__(self, node_feature_mode=1,
                       gnn_type='GCN',
                       mp_hidden_dim=32,
                       mp_output_dim=64,
                       mp_num_layers=1, 
                       dual_node_mlp_hidden_dim=64,
                       dual_node_mlp_output_dim=10,
                       node_mlp_num_layers=1,
                       dual_edge_mlp_hidden_dim=64, 
                       dual_edge_mlp_output_dim=16, 
                       edge_mlp_num_layers=1, 
                       dropout_rate=0.2,
                       relu_slope=0.1):
        super(DualModel,self).__init__()
        if node_feature_mode == 1:
            mp_input_dim = 6 # use original point coordinates
        elif node_feature_mode == 2:
            mp_input_dim = 10 # use entries of cost matrix C
        elif node_feature_mode == 3:
            mp_input_dim = 16 # use both
        else:
            raise RuntimeError('node_feature_mode can only be 1, 2, or 3.')
        self.node_feature_mode = node_feature_mode
        print(f'Model: node_feature_mode = {node_feature_mode}, mp_input_dim = {mp_input_dim}, relu_slope = {relu_slope}. GNN type: {gnn_type}.')
        # Message passing
        self.mp_convs = nn.ModuleList()
        if gnn_type == 'GCN':
            self.mp_convs.append(pyg_nn.GCNConv(mp_input_dim,mp_hidden_dim,add_self_loops=False))
            for i in range(mp_num_layers):
                self.mp_convs.append(pyg_nn.GCNConv(mp_hidden_dim,mp_hidden_dim,add_self_loops=False))
            self.mp_convs.append(pyg_nn.GCNConv(mp_hidden_dim,mp_output_dim,add_self_loops=False))
        elif gnn_type == 'SAGE':
            self.mp_convs.append(pyg_nn.SAGEConv(mp_input_dim,mp_hidden_dim))
            for i in range(mp_num_layers):
                self.mp_convs.append(pyg_nn.SAGEConv(mp_hidden_dim,mp_hidden_dim))
            self.mp_convs.append(pyg_nn.SAGEConv(mp_hidden_dim,mp_output_dim))
        elif gnn_type == 'GraphConv':
            self.mp_convs.append(pyg_nn.GraphConv(mp_input_dim,mp_hidden_dim))
            for i in range(mp_num_layers):
                self.mp_convs.append(pyg_nn.GraphConv(mp_hidden_dim,mp_hidden_dim))
            self.mp_convs.append(pyg_nn.GraphConv(mp_hidden_dim,mp_output_dim))
        elif gnn_type == 'GATConv':
            self.mp_convs.append(pyg_nn.GATConv(mp_input_dim,mp_hidden_dim))
            for i in range(mp_num_layers):
                self.mp_convs.append(pyg_nn.GATConv(mp_hidden_dim,mp_hidden_dim))
            self.mp_convs.append(pyg_nn.GATConv(mp_hidden_dim,mp_output_dim))

        # Post message passing
        # Dual node MLP
        self.dual_node_mlp = nn.ModuleList()
        self.dual_node_mlp.append(
            nn.Linear(mp_output_dim,dual_node_mlp_hidden_dim,dtype=torch.float64))
        for i in range(node_mlp_num_layers):
            self.dual_node_mlp.append(
                nn.Linear(dual_node_mlp_hidden_dim,dual_node_mlp_hidden_dim,dtype=torch.float64))
        self.dual_node_mlp.append(
            nn.Linear(dual_node_mlp_hidden_dim,dual_node_mlp_output_dim,dtype=torch.float64))
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
        self.relu_slope = relu_slope
        if self.dual_edge_mlp_output_dim != 16 and self.dual_edge_mlp_output_dim != 6:
            raise RuntimeError('dual_edge_mlp_output_dim must be 16 or 6.')

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        ptr, edge_map, ud_edges = data.ptr, data.edge_map, data.ud_edges
        if self.node_feature_mode == 1:
            x = x[:,0:6] # use point coordinates
        elif self.node_feature_mode == 2:
            x = x[:,6:] # use cost matrix entries

        # Message passing
        for mp_layer in self.mp_convs[:-1]:
            x = mp_layer(x,edge_index)
            x = F.leaky_relu(x,negative_slope=self.relu_slope)
            x = F.dropout(x,p=self.dropout_rate,training=self.training)
        x = self.mp_convs[-1](x,edge_index) # last layer no relu and dropout
        
        # Post message passing
        num_graphs = data.num_graphs
        S = []
        Aty = []
        for k in range(num_graphs):
            node_id = torch.arange(ptr[k],ptr[k+1])
            Sk, Atyk = self.post_mp_one_graph(x[node_id,:],ud_edges[k],edge_map[k])
            S.append(Sk)
            Aty.append(Atyk)
        return x, S, Aty

    
    def post_mp_one_graph(self,x,ud_edges,edge_map):
        num_nodes = x.shape[0]
        # Dual node
        vd = []
        for i in range(num_nodes):
            xi = x[i,:]
            for mlp_layer in self.dual_node_mlp[:-1]:
                xi = mlp_layer(xi)
                xi = F.leaky_relu(xi,negative_slope=self.relu_slope)
                xi = F.dropout(xi,p=self.dropout_rate,training=self.training)
            xi = self.dual_node_mlp[-1](xi)
            vd.append(xi)
        vd = torch.stack(vd) # num_nodes x dual_node_mlp_output_dim
        # Dual edge
        ed = []
        for edge in ud_edges:
            xi  = x[edge[0],:]
            xj  = x[edge[1],:]
            xij = xi + xj
            for mlp_layer in self.dual_edge_mlp[:-1]:
                xij = mlp_layer(xij)
                xij = F.leaky_relu(xij,negative_slope=self.relu_slope)
                xij = F.dropout(xij,p=self.dropout_rate,training=self.training)
            xij = self.dual_edge_mlp[-1](xij)
            ed.append(xij)
        ed = torch.stack(ed)

        # Recover dual S or Aty
        if self.dual_edge_mlp_output_dim == 16:
            S = self.recover_S(vd,ed,edge_map)
            Aty = None 
        elif self.dual_edge_mlp_output_dim == 6:
            Aty = self.recover_Aty(vd,ed,edge_map)
            S = None
        else:
            raise RuntimeError(f'dual_edge_mlp_output_dim = {self.dual_edge_mlp_output_dim} not supported.')

        return S, Aty


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

    def loss(self,data,S,Aty):
        Sopt = data.S
        Atyopt = data.Aty
        if S[0] is not None:
            device = S[0].device
        else:
            device = Aty[0].device
        num_graphs = data.num_graphs
        dual_loss = []

        for i in range(num_graphs):
            if self.dual_edge_mlp_output_dim == 16:
                Sopti = torch.tensor(Sopt[i],dtype=torch.float64,device=device)
                dual_loss.append(
                    torch.div(
                        torch.norm(S[i] - Sopti,p='fro'),
                        torch.norm(Sopti,p='fro'))
                    )         
            elif self.dual_edge_mlp_output_dim == 6:
                Atyopti = torch.tensor(Atyopt[i],dtype=torch.float64,device=device)
                dual_loss.append(
                    torch.div(
                        torch.norm(Aty[i] - Atyopti,p='fro'),
                        torch.norm(Atyopti,p='fro'))
                    )
                    
        return torch.mean(torch.stack(dual_loss))