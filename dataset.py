import torch
import numpy as np 
from torch_geometric.data import Dataset, Data
import torch_geometric.utils as pyg_utils
import os.path as osp
import scipy.io as sio
import h5py

def gen_index_map(ud_edges,n):
    map = np.zeros((n,n),dtype=np.int16)
    for i, edge in enumerate(ud_edges):
        map[edge[0],edge[1]] = i
        map[edge[1],edge[0]] = i
    return map

def flatten_X(X,edge_map):
    n = X.shape[0]
    N = np.int(n/4)
    N_e = np.int( (N*(N-1))/2 )
    v = np.zeros((N,10))
    e = np.zeros((N_e,10))
    for i in range(N):
        for j in range(N):
            blk = X[4*i:4*i+4,4*j:4*j+4]
            tmp = np.array([blk[0,0],
                            blk[0,1],
                            blk[0,2],
                            blk[0,3],
                            blk[1,1],
                            blk[1,2],
                            blk[1,3],
                            blk[2,2],
                            blk[2,3],
                            blk[3,3]])
            if i == j:
                v[i,:] = tmp
            elif j > i:
                id = edge_map[i,j]
                e[id,:] = tmp
    return v, e

def flatten_S(S,edge_map):
    n = S.shape[0]
    N = np.int(n/4)
    N_e = np.int( (N*(N-1))/2 )
    v = np.zeros((N,10))
    e = np.zeros((N_e,16))
    for i in range(N):
        blk = S[4*i:4*i+4,4*i:4*i+4]
        tmp = np.array([blk[0,0],
                        blk[0,1],
                        blk[0,2],
                        blk[0,3],
                        blk[1,1],
                        blk[1,2],
                        blk[1,3],
                        blk[2,2],
                        blk[2,3],
                        blk[3,3]])
        v[i,:] = tmp
    for i in range(N):
        for j in range(N):
            if j > i:
                id = edge_map[i,j]
                blk = S[4*i:4*i+4,4*j:4*j+4]
                e[id,:] = np.reshape(blk,(16,))
    return v, e

def flatten_Aty(Aty,edge_map):
    n = Aty.shape[0]
    N = np.int(n/4)
    N_e = np.int( (N*(N-1))/2 )
    v = np.zeros((N,10))
    e = np.zeros((N_e,6))
    for i in range(N):
        blk = Aty[4*i:4*i+4,4*i:4*i+4]
        tmp = np.array([blk[0,0],
                        blk[0,1],
                        blk[0,2],
                        blk[0,3],
                        blk[1,1],
                        blk[1,2],
                        blk[1,3],
                        blk[2,2],
                        blk[2,3],
                        blk[3,3]])
        v[i,:] = tmp
    
    for i in range(N):
        for j in range(N):
            if j > i:
                id = edge_map[i,j]
                blk = Aty[4*i:4*i+4,4*j:4*j+4]
                tmp = np.array([
                    blk[0,1],
                    blk[0,2],
                    blk[0,3],
                    blk[1,2],
                    blk[1,3],
                    blk[2,3]
                ])
                e[id,:] = tmp
    return v, e

class QUASARDataset(Dataset):
    def __init__(self, root, num_graphs, graph_type=1, remove_self_loops=True, transform=None, pre_transform=None, pre_filter=None, matversion=1):
        self.remove_self_loops = remove_self_loops
        self.num_graphs = num_graphs
        self.graph_type = graph_type # 1 for fully-connected graph, 2 for star graph
        self.matversion = matversion
        if self.graph_type > 2:
            raise RuntimeError('Data graph type is either fully connected (1) or star (2).')
        print(f'Data graph type: {graph_type}.')
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.raw_dir = osp.join(root,'raw')
        # self.processed_dir = osp.join(root,'processed')

    @property
    def raw_file_names(self):
        return ['quasar_sol.mat']
    
    @property
    def processed_file_names(self):
        names = []
        for i in range(self.num_graphs):
            names.append(f'data_{i}.pt')
        return names

    def download(self):
        pass

    def process(self):
        if self.matversion == 2:
            print('Use h5py to read mat file.')
            count = 0 
            for raw_file_name in self.raw_file_names:
                fname = osp.join(self.raw_dir,raw_file_name) 
                file = h5py.File(fname,'r')
                Call = file['log_data']['C']
                Xall = file['log_data']['X']
                Sall = file['log_data']['S']
                pall = file['log_data']['points']
                qall = file['log_data']['q']
                thall = file['log_data']['theta']
                num_graphs = Call.shape[1]
                for i in range(num_graphs):
                    C               = np.array(file[Call[0][i]]) # (4N+4,4N+4)
                    Xopt            = np.array(file[Xall[0][i]]) # (4N+4,4N+4)
                    Sopt            = np.array(file[Sall[0][i]]) # (4N+4,4N+4)
                    points          = np.array(file[pall[0][i]]).T # (6,N) np array
                    q               = np.array(file[qall[0][i]]).flatten() # (4,)
                    theta           = np.array(file[thall[0][i]]).flatten() # (N,)
                    num_nodes       = theta.shape[0]
                    Aty             = C - Sopt

                    # Create PyG graph data
                    points_aug      = np.concatenate((
                                                    np.ones((6,1)),points),axis=1) # add extra one node
                    points_aug      = points_aug.T
                    Ctriu           = self.mytriu(C)
                    qsol            = self.myX(q,theta)
                    node_features   = np.concatenate((points_aug,Ctriu,qsol),axis=1)

                    node_features   = torch.tensor(node_features,dtype=torch.float64) # num_nodes x num_features

                    if self.graph_type == 1: # fully connected graph
                        edge_index  = torch.ones(num_nodes+1,num_nodes+1).nonzero().t().contiguous()
                    elif self.graph_type == 2: # star graph
                        first_row = torch.ones(1,num_nodes+1)
                        first_col = torch.ones(num_nodes,1)
                        adj_mat = torch.cat((first_row,torch.cat((first_col,torch.eye(num_nodes)),dim=1)),dim=0)
                        edge_index  = adj_mat.nonzero().t().contiguous()

                    if self.remove_self_loops:
                        edge_index  = pyg_utils.remove_self_loops(edge_index)[0]
                    
                    ud_edges        = torch.triu(
                        torch.ones(num_nodes+1,num_nodes+1) - torch.eye(num_nodes+1)).nonzero().t().contiguous()
                    ud_edges        = ud_edges.t().tolist()
                    edge_map        = gen_index_map(ud_edges,num_nodes+1)
                    # flatten
                    Aty_v, Aty_e = flatten_Aty(Aty,edge_map)
                    X_v, X_e = flatten_X(Xopt,edge_map)
                    S_v, S_e = flatten_S(Sopt,edge_map)

                    graph           = Data(x=node_features,
                                        edge_index=edge_index,
                                        ud_edges=ud_edges,
                                        edge_map=edge_map,
                                        X=Xopt,
                                        S=Sopt,
                                        Aty=Aty,
                                        C=C,
                                        Aty_v=Aty_v,
                                        Aty_e=Aty_e,
                                        X_v=X_v,
                                        X_e=X_e,
                                        S_v=S_v,
                                        S_e=S_e)

                    torch.save(graph,osp.join(self.processed_dir,f'data_{count}.pt'))
                    count += 1
            print(f'Expected # graphs: {self.num_graphs}. Actual # graphs: {count}.')
            if count != self.num_graphs:
                raise RuntimeError('Actual number of graphs different from expected.')


        elif self.matversion == 1:
            print('Use scipy io to read mat file.')
            count = 0
            for raw_file_name in self.raw_file_names:
                # load the .mat file
                file = osp.join(self.raw_dir,raw_file_name) 
                raw_data = sio.loadmat(file)['log_data']
                num_graphs = len(raw_data)
                for i in range(num_graphs):
                    # extract quasar solution from the problem data
                    prob            = raw_data[i][0]
                    points          = prob[0] # (6,N) np array
                    outlier_rate    = prob[1].flatten() # (1,)
                    barc2           = prob[2].flatten() # (1,)
                    Xopt            = prob[3] # (4N+4,4N+4)
                    Sopt            = prob[4] # (4N+4,4N+4)
                    optval          = prob[5].flatten() # (1,)
                    C               = prob[6] # (4N+4,4N+4)
                    Rgt             = prob[7] # (3,3)
                    q               = prob[8].flatten() # (4,)
                    theta           = prob[9].flatten() # (N,)
                    eta             = prob[10].flatten() # (4,)
                    
                    num_nodes       = theta.shape[0]
                    Aty             = C - Sopt

                    # Create PyG graph data
                    points_aug      = np.concatenate((
                                                    np.ones((6,1)),points),axis=1) # add extra one node
                    points_aug      = points_aug.T
                    Ctriu           = self.mytriu(C)
                    qsol            = self.myX(q,theta)
                    node_features   = np.concatenate((points_aug,Ctriu,qsol),axis=1)

                    node_features   = torch.tensor(node_features,dtype=torch.float64) # num_nodes x num_features

                    if self.graph_type == 1: # fully connected graph
                        edge_index  = torch.ones(num_nodes+1,num_nodes+1).nonzero().t().contiguous()
                    elif self.graph_type == 2: # star graph
                        first_row = torch.ones(1,num_nodes+1)
                        first_col = torch.ones(num_nodes,1)
                        adj_mat = torch.cat((first_row,torch.cat((first_col,torch.eye(num_nodes)),dim=1)),dim=0)
                        edge_index  = adj_mat.nonzero().t().contiguous()

                    if self.remove_self_loops:
                        edge_index  = pyg_utils.remove_self_loops(edge_index)[0]
                    
                    ud_edges        = torch.triu(
                        torch.ones(num_nodes+1,num_nodes+1) - torch.eye(num_nodes+1)).nonzero().t().contiguous()
                    ud_edges        = ud_edges.t().tolist()
                    edge_map        = gen_index_map(ud_edges,num_nodes+1)
                    # flatten
                    Aty_v, Aty_e = flatten_Aty(Aty,edge_map)
                    X_v, X_e = flatten_X(Xopt,edge_map)
                    S_v, S_e = flatten_S(Sopt,edge_map)

                    graph           = Data(x=node_features,
                                        edge_index=edge_index,
                                        ud_edges=ud_edges,
                                        edge_map=edge_map,
                                        X=Xopt,
                                        S=Sopt,
                                        Aty=Aty,
                                        C=C,
                                        Aty_v=Aty_v,
                                        Aty_e=Aty_e,
                                        X_v=X_v,
                                        X_e=X_e,
                                        S_v=S_v,
                                        S_e=S_e)

                    torch.save(graph,osp.join(self.processed_dir,f'data_{count}.pt'))
                    count += 1
            print(f'Expected # graphs: {self.num_graphs}. Actual # graphs: {count}.')
            if count != self.num_graphs:
                raise RuntimeError('Actual number of graphs different from expected.')
    
    def len(self):
        return len(self.processed_file_names)

    def get(self,idx):
        return torch.load(osp.join(self.processed_dir,f'data_{idx}.pt'))

    def mytriu(self,C):
        n = C.shape[0]
        N = np.int(n/4)
        x = np.zeros((N,10))
        for i in range(N):
            blk = C[0:4,:][:,4*i:4*i+4]
            x[i,:] = blk[np.triu_indices(4)]
        return x

    def myX(self,q,theta):
        N = theta.shape[0]
        q = np.reshape(q,(1,4))
        theta = np.reshape(theta,(N,1))
        X = np.kron(theta,q)
        X = np.concatenate((q,X),axis=0)
        return X

if __name__ == "__main__":
    dir        = '/Users/hankyang/Datasets/QUASAR'
    Dataset    =  QUASARDataset(dir)