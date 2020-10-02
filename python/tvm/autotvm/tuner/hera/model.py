import torch
from torch.utils.data import DataLoader, Dataset

def get_dataloader(train, target=None, batch_size=32, shuffle=False):
    if target is not None:
        return _get_train_dataloader(train, target, batch_size, shuffle)
    
    return _get_pred_dataloader(train, batch_size, shuffle)

def _get_train_dataloader(train, target, batch_size, shuffle):
    ds = WhateverDS(train, target)
    return _get_loader(ds, batch_size=batch_size, shuffle=shuffle)

def _get_pred_dataloader(train, batch_size, shuffle):
    ds = WhateverPredDs(train)
    return _get_loader(ds, batch_size=batch_size, shuffle=shuffle)

def _get_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class WhateverPredDs(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.df.iloc[idx].to_dict()
        return x

class WhateverDS(Dataset):
    def __init__(self, df, target):
        self.df = df
        self.target = target
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.df.iloc[idx].to_dict()
        y = self.target.iloc[idx]
        return x, y

class WhateverModel(torch.nn.Module):
    def __init__(self, cat_feats, padding_indexes, embedding_dims=10, hidden_dim=32):
        super(WhateverModel, self).__init__()
        self.cat_feats = cat_feats
        self.embs = {}
        total = 0
        for i, (feat, size) in enumerate(self.cat_feats.items()):
            embed = torch.nn.Embedding(size, embedding_dims, padding_idx=padding_indexes[feat])
            self.embs[feat] = embed
            total+=1
        # cat all
        self.linear = torch.nn.Linear((total*embedding_dims)+1, hidden_dim)
        self.linear2 = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU6()
        self.final = torch.nn.Linear(hidden_dim, 1)
    
    def get_features(self, x):
        feats = []
        for k,v in x.items():
            embed = self.embs.get(k, None)
            if embed is None:
                continue
            try:
                feats.append(embed(v.long().detach()))
            except IndexError as e:
                print(k)
                raise e
        feats = torch.cat(feats, axis=1)
        batch = x["flop"].size(0)
        flops = x["flop"].reshape(batch,1).float().detach()
        x2 = self.linear2(flops)
        x2 = self.relu(x2)
        allfeats = torch.cat((feats, x2), axis=1)
        x = self.linear(allfeats)
        return x

    def forward(self, x, feats_only=False):
        x = self.get_features(x)
        if feats_only:
            return x
        x = self.relu(x)
        return self.final(x)
        