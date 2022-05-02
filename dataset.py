import torch
import numpy as np 
from torch_geometric.data import Dataset, Data
import os.path as osp
import scipy.io as sio

def gen_index_map(ud_edges,n):
    map = np.zeros((n,n),dtype=np.int16)
    for i, edge in enumerate(ud_edges):
        map[edge[0],edge[1]] = i
        map[edge[1],edge[0]] = i
    return map

class QUASARDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.raw_dir = osp.join(root,'raw')
        # self.processed_dir = osp.join(root,'processed')

    @property
    def raw_file_names(self):
        return ['quasar_sol_0.mat']
    
    @property
    def processed_file_names(self):
        names = []
        for i in range(1000):
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
                                                  np.zeros((6,1)),points),axis=1) # add extra one node
                node_features   = torch.tensor(points_aug.T,dtype=torch.float64) # num_nodes x num_features
                edge_index      = torch.ones(num_nodes+1,num_nodes+1).nonzero().t().contiguous()
                ud_edges        = torch.triu(
                    torch.ones(num_nodes+1,num_nodes+1) - torch.eye(num_nodes+1)).nonzero().t().contiguous()
                ud_edges        = ud_edges.t().tolist()
                edge_map        = gen_index_map(ud_edges,num_nodes+1)
                y               = torch.stack((torch.tensor(Xopt,dtype=torch.float64),
                                               torch.tensor(Sopt,dtype=torch.float64),
                                               torch.tensor(Aty,dtype=torch.float64)))

                graph           = Data(x=node_features,
                                       edge_index=edge_index,
                                       y=y,
                                       ud_edges=ud_edges,
                                       edge_map=edge_map)

                torch.save(graph,osp.join(self.processed_dir,f'data_{count}.pt'))
                count += 1
    
    def len(self):
        return len(self.processed_file_names)

    def get(self,idx):
        return torch.load(osp.join(self.processed_dir,f'data_{idx}.pt'))


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