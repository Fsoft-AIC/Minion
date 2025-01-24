import torch
import numpy as np
from sklearn.cluster import KMeans
import copy

class Config:
    def __init__(self):
        self.memory_size = 5
        self.hidden_size = 768  # Typical BERT hidden size

class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_samples):
        self.data = [i for i in range(num_samples)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input': torch.rand(1, 768, dtype=torch.bfloat16)}, 0, idx

def get_data_loader_BERT(config, dataset, shuffle, drop_last, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

class TestClass:
    def __init__(self):
        self.config = Config()

    def select_memory(self, dataset):
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, drop_last=False, batch_size=1)
        features = []
        
        for step, (instance, label, idx) in enumerate(data_loader):
            # Simulating encoder output with random bfloat16 tensor
            hidden = instance['input']  # This is already a random bfloat16 tensor
            print(hidden.dtype)
            fea = hidden.detach().cpu().float().numpy()  # Convert to float32 and then to numpy array
            features.append(fea)
        print(type(features[0]))
        features = np.concatenate(features)  # Now concatenation should work
        
        features = features.reshape(N, -1)  # Reshape to 2D: (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0)
        mem_feas = torch.from_numpy(mem_feas)
        
        features = torch.from_numpy(features)
        rel_proto = features.mean(0)

        return mem_set, mem_feas

# Test the function
test_instance = TestClass()
test_dataset = Dataset(10)  # Create a dataset with 10 samples
mem_set, mem_feas = test_instance.select_memory(test_dataset)

print(f"Number of samples in memory set: {len(mem_set)}")
print(f"Shape of memory features: {mem_feas.shape}")