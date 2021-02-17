import os
import numpy as np
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point



class ModelNetDataLoader(Dataset):
    def __init__(self, root, tasks, labels, partition='train', npoint=2048, uniform=False, normal_channel=False, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.normal_channel = normal_channel
        self.data, self.label = load_data(root, tasks, labels)
        self.partition = partition
        print('The number of ' + partition + ' data: ' + str(self.data.shape[0]))

    def __len__(self):
        return self.data.shape[0]
    
    def _get_item(self, index):
        pointcloud = self.data[index][:self.npoints]
        label = self.label[index]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)

        return pointcloud, label

    def __getitem__(self, index):
        return self._get_item(index)
    
    
    
def load_data(root, tasks, labels):
    all_data = []
    all_label = []
    for i in range(len(tasks)):
        data = np.load(os.path.join(root, tasks[i]))
        label = np.load(os.path.join(root, labels[i]))
        
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    
    
    
    
    return all_data, all_label