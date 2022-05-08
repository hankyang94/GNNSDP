import torch
import numpy as np 
from torch_geometric.data import Dataset, Data
import torch_geometric.utils as pyg_utils
import os.path as osp
import scipy.io as sio

def gen_index_map(ud_edges,n):
    map = np.zeros((n,n),dtype=np.int16)
    for i, edge in enumerate(ud_edges):
        map[edge[0],edge[1]] = i
        map[edge[1],edge[0]] = i
    return map

class QUASARDataset(Dataset):
    def __init__(self, root, num_graphs, graph_type=1, remove_self_loops=True, transform=None, pre_transform=None, pre_filter=None):
        self.remove_self_loops = remove_self_loops
        self.num_graphs = num_graphs
        self.graph_type = graph_type # 1 for fully-connected graph, 2 for star graph
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
                node_features   = np.concatenate((points_aug,Ctriu),axis=1)

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
                # y               = torch.stack((torch.tensor(Xopt,dtype=torch.float64),
                #                                torch.tensor(Sopt,dtype=torch.float64),
                #                                torch.tensor(Aty,dtype=torch.float64)))

                graph           = Data(x=node_features,
                                       edge_index=edge_index,
                                       ud_edges=ud_edges,
                                       edge_map=edge_map,
                                       X=Xopt,
                                       S=Sopt,
                                       Aty=Aty,
                                       C=C)

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


    # def primal_vec(X):
    #     # expect X to be numpy array of size (4N+4,4N+4)
    #     n = X.shape[0]
    #     N = n/4
    #     node = []


    # def svec(S):
    #     # S is a symmetric matrix in numpy
    #     n = S.shape[0]
    #     si, sj = np.meshgrid(np.arange(n), np.arange(n))
    #     mask_tri = (si >= sj)
    #     si = si[mask_tri]
    #     sj = sj[mask_tri]

    #     mask_nondiag = (si > sj)
    #     data = S[si, sj]
    #     data[mask_nondiag] *= np.sqrt(2)

    #     return data




if __name__ == "__main__":
    dir        = '/Users/hankyang/Datasets/QUASAR'
    Dataset    =  QUASARDataset(dir)