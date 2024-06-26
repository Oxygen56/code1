import pickle
import time

dataset_name = 'Ciao' # dataset: Epinions, Ciao or yelp
dataset_path = '../dataset/' + dataset_name + '/'


# 20 表示0.2作为测试
with open(dataset_path + 'dataset_filter5.pkl', 'rb') as f:
    train_set = pickle.load(f)
    valid_set = pickle.load(f)
    test_set = pickle.load(f)


with open(dataset_path + 'list_filter5.pkl', 'rb') as f:
    u_items_list = pickle.load(f)
    u_users_list = pickle.load(f)
    u_users_items_list = pickle.load(f)
    i_users_list = pickle.load(f)
    (user_count, item_count, rate_count) = pickle.load(f)




from tranh import TransH
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class GRDataset(Dataset):
    def __init__(self, pos_data, neg_data):
        self.pos_data = pos_data
        self.neg_data = neg_data

    def __getitem__(self, index):
        return self.pos_data[index], self.neg_data[index]

    def __len__(self):
        return len(self.pos_data)



def collate_fn(batch_data):
    pos_list, neg_list = [],[]
    for pl, nl in batch_data:
        pos_list.append(pl)
        neg_list.append(nl)

    pos_data_pad = torch.LongTensor(pos_list)
    neg_data_pad = torch.LongTensor(neg_list)

    return pos_data_pad, neg_data_pad

# generated by cross_sampling.py
with open('%s/relation_train_set.pkl' %dataset_name, 'rb') as f:
    new_pos = pickle.load(f)
    new_neg = pickle.load(f)

train_data = GRDataset(new_pos, new_neg)
train_loader = DataLoader(train_data, batch_size=256, shuffle = True, collate_fn = collate_fn)


# embedding dim, it needs to be the same with embedding size in the main model
dim = 80

user_emb = nn.Embedding(user_count+1, dim, padding_idx = 0)
item_emb = nn.Embedding(item_count+1, dim, padding_idx = 0)


model = TransH(user_emb, item_emb, user_count, item_count, 5, dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

print(dim)

min_loss = 1e15
print('begin: ')
for epoch in range(20):
    corrent_loss, batch = 0,0
    for posX, negX in train_loader:
        posX = posX.cuda()
        negX = negX.cuda()

        model.normalizeEmbedding()
        loss = model(posX, negX)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        corrent_loss += loss.item()
        batch += 1
        if batch % 10000 == 0:
            print("epoch %d with batch %d" %(epoch, batch))

    if corrent_loss < min_loss:
        print('epoch, current loss, min_loss: ', epoch, corrent_loss, min_loss)
        with open('%s/relation_train_set_model_%s.pkl' %(dataset_name,dim), 'wb') as f:
            pickle.dump(model, f)
        min_loss = corrent_loss


# begin to store the pretrain embedding parameters
user_emb = model.HentityEmbedding.weight.detach().cpu().numpy()
item_emb = model.TentityEmbedding.weight.detach().cpu().numpy()
rating_emb = model.relationEmbedding.weight.detach().cpu().numpy()


with open('%s/relation_train_set_model_%s_weights.pkl' %(dataset_name, dim), 'wb') as f:
    pickle.dump(user_emb, f)
    pickle.dump(item_emb, f)
    pickle.dump(rating_emb, f)
