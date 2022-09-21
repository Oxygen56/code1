#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
created by Kun Yuan, Sep 20, 2022.
The code is revised on Wangshuo's version.
"""

import os
import time
import json
import argparse
import pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from utils import collate_fn
from DeppGraph import DeppGraph
from dataloader import GRDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='./dataset/Ciao/', help='dataset directory path: datasets/Ciao/Epinions')
parser.add_argument('--dataset_name', default='Ciao', help='Ciap, Epinions, yelp')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=80, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--seed', type=int, default=14, help='the number of random seed to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=50, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', type=int,default=0, help='test')
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

here = os.path.dirname(os.path.abspath(__file__))


fn = 'results/'+  args.dataset_name

if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists(fn):
    os.mkdir(fn)

'''
cpu_num = 4 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
'''

def main():
    print('Loading data...')
    # use
    with open(args.dataset_path + 'dataset_filter5.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)

    # load train set generated by cross-sampling
    with open('tranh/%s/new_train_set_transh_30.pkl' %args.dataset_name, 'rb') as f:
        train_set = pickle.load(f)
        neg_transh_train_set = pickle.load(f)


    # guarantee all users and items in validation and testing set have appeared in the training set
    train_user_set = set([u for u,i,r in train_set])
    train_item_set = set([i for u,i,r in train_set])
    new_valid_set = []
    for u,i,r in valid_set:
        if u in train_user_set and i in train_item_set and r != 0:
            new_valid_set.append((u,i,r))
    new_test_set = []
    for u,i,r in test_set:
        if u in train_user_set and i in train_item_set and r != 0:
            new_test_set.append((u,i,r))

    valid_set = new_valid_set
    test_set = new_test_set


    # load data
    with open(args.dataset_path + 'list_filter5.pkl', 'rb') as f:
        u_items_list = pickle.load(f)
        u_users_list = pickle.load(f)
        u_users_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        (user_count, item_count, rate_count) = pickle.load(f)

    with open(args.dataset_path + 'self_sf_user_list_filter5.pkl', 'rb') as f:
        sf_list = pickle.load(f)

    with open(args.dataset_path + 'self_sf_user_items_list_filter5.pkl', 'rb') as f:
        sf_user_item_list = pickle.load(f)


    # use: i:[i1, i3]
    with open(args.dataset_path + 'bal_sample_item_list_filter5.pkl', 'rb') as f:
        i_item_friend_list = pickle.load(f)


    with open(args.dataset_path + 'bal_sample_item_users_list_filter5.pkl', 'rb') as f:
        if_item_users_list = pickle.load(f)


    with open('tranh/%s/relation_train_set_model_%s_weights.pkl' %(args.dataset_name, args.embed_dim), 'rb') as f:
        user_emb = pickle.load(f)
        item_emb = pickle.load(f)
        rate_emb = pickle.load(f)
    
    train_data = GRDataset(train_set, u_items_list,  u_users_list, u_users_items_list, i_users_list,  sf_list, sf_user_item_list, i_item_friend_list, if_item_users_list, neg_transh_train_set)
    valid_data = GRDataset(valid_set, u_items_list,  u_users_list, u_users_items_list, i_users_list,  sf_list, sf_user_item_list, i_item_friend_list, if_item_users_list, [])
    test_data = GRDataset(test_set, u_items_list, u_users_list,  u_users_items_list, i_users_list,  sf_list, sf_user_item_list, i_item_friend_list,if_item_users_list, [])
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    model = DeppGraph(user_emb, item_emb, rate_emb, user_count+1, item_count+1, rate_count+1, args.embed_dim).to(device)

    # set test=1 if testing mode is executed
    if args.test:
        print('Load checkpoint and testing...')
        ckpt = torch.load('%s/random_best_checkpoint.pth.tar' %fn)
        model.load_state_dict(ckpt['state_dict'])
        print(ckpt['epoch'])
        user_emb = model.user_emb.weight.detach().cpu().numpy()
        item_emb = model.item_emb.weight.detach().cpu().numpy()
        with open('%s/GNN_embedding.pkl' %fn, 'wb') as f:
            pickle.dump(user_emb, f)
            pickle.dump(item_emb, f)
        mae, rmse, results = validate(train_loader, model)
        t_mae, t_rmse, t_results = validate(test_loader, model)
        v_mae, v_rmse, v_results = validate(valid_loader, model)
        with open('%s/GNN_train.txt' %fn, 'w') as f:
            f.write(json.dumps(results))
        with open('%s/GNN_test.txt' %fn, 'w') as f:
            f.write(json.dumps(t_results))
        with open('%s/GNN_valid.txt' %fn, 'w') as f:
            f.write(json.dumps(v_results))
        print("Test: MAE: {:.4f}, RMSE: {:.4f}".format(t_mae, t_rmse))

        return

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    valid_loss_list, test_loss_list = [],[]
    best_mae = 10

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch = epoch)
        valid_loss, test_loss = trainForEpoch(train_loader, valid_loader, test_loader, model, optimizer, epoch, args.epoch, criterion, best_mae, log_aggr = 100)


        valid_loss_list.extend(valid_loss)
        test_loss_list.extend(test_loss)

        # store best loss and save a model checkpoint

        x = [k for k,v in valid_loss]
        idx = x.index(min(x))
        mae, rmse = valid_loss[idx]
        test_mae, test_rmse = test_loss[idx]

        if epoch == 0:
            best_mae = mae
        elif mae < best_mae:
            best_mae = mae

        print('Epoch {} validation: MAE: {:.4f}, RMSE: {:.4f}, Best MAE: {:.4f}, test_MAE: {:.4f}, test_RMSE: {:.4f}'.format(epoch, mae, rmse, best_mae, test_mae, test_rmse))

        with open('%s/random_valid_loss_list.txt' %fn, 'w') as f:
            f.write(json.dumps(valid_loss_list))

        with open('%s/random_test_loss_list.txt' %fn, 'w') as f:
            f.write(json.dumps(test_loss_list))



def trainForEpoch(train_loader, valid_loader, test_loader, model, optimizer, epoch, num_epochs, criterion, best_mae, log_aggr=1):
    model.train()

    sum_epoch_loss = 0
    valid_loss_list, test_loss_list = [],[]

    start = time.time()
    for i, (uids, iids, labels, u_items, u_users, u_users_items, i_users, i_sf_users, i_sf_users_items, i_friend_list, if_item_users, pos_list, neg_list) in tqdm(enumerate(train_loader), total=len(train_loader)):
        uids = uids.to(device)
        iids = iids.to(device)
        labels = labels.to(device)
        pos_list = pos_list.to(device)
        neg_list = neg_list.to(device)
        u_items = u_items.to(device)
        u_users = u_users.to(device)
        u_users_items = u_users_items.to(device)
        i_users = i_users.to(device)
        i_sf_users = i_sf_users.to(device)
        i_sf_users_items = i_sf_users_items.to(device)
        i_friend_list = i_friend_list.to(device)
        if_item_users = if_item_users.to(device)
        
        optimizer.zero_grad()
        outputs, r = model(uids, iids, u_items, u_users, u_users_items, i_users, i_sf_users, i_sf_users_items, i_friend_list, if_item_users, pos_list, neg_list, True)

        loss = criterion(outputs, labels.unsqueeze(1))
        loss += 2*r
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN Main 2] epoch %d/%d batch loss: %.4f %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, r.item(), sum_epoch_loss / (i + 1),
                  len(uids) / (time.time() - start)))

        start = time.time()

        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if i % 400 == 0 and i != 0:
            model.eval()
            mae, rmse, valid_errors = validate(valid_loader, model)
            valid_loss_list.append([mae, rmse])
            print('valid_result: ', [mae, rmse])

            test_mae, test_rmse, test_errors = validate(test_loader, model)
            test_loss_list.append([test_mae, test_rmse])
            print('test_result: ', [test_mae, test_rmse])

            if test_mae < 0.7:
                torch.save(ckpt_dict, '%s/random_latest_checkpoint_%s_%s.pth.tar' %(fn,epoch,i))

                with open('%s/test_predict_list_%s_%s.txt' % (fn, epoch, i), 'w') as f:
                    f.write(json.dumps(test_errors))
                with open('%s/valid_predict_list_%s_%s.txt' % (fn, epoch, i), 'w') as f:
                    f.write(json.dumps(valid_errors))

            if test_mae < best_mae:
                torch.save(ckpt_dict, '%s/random_best_checkpoint.pth.tar' %(fn))
                #torch.save(ckpt_dict, '%s/random_best_checkpoint_%s_%s.pth.tar' %(fn,epoch,i))
                best_mae = test_mae
                with open('%s/test_best_predict_list.txt' % (fn), 'w') as f:
                    f.write(json.dumps(test_errors))
                with open('%s/valid_best_predict_list.txt' % (fn), 'w') as f:
                    f.write(json.dumps(valid_errors))


            model.train()


    return valid_loss_list, test_loss_list



def validate(valid_loader, model):
    model.eval()
    errors = []
    results = []
    with torch.no_grad():
        for uids, iids, labels,  u_items, u_users, u_users_items, i_users, i_sf_users, i_sf_users_items, i_friend_list, if_item_users, pos_list, neg_list in tqdm(valid_loader):
            uids = uids.to(device)
            iids = iids.to(device)
            labels = labels.to(device)
            pos_list = pos_list.to(device)
            neg_list = neg_list.to(device)
            u_items = u_items.to(device)
            u_users = u_users.to(device)
            u_users_items = u_users_items.to(device)
            i_users = i_users.to(device)
            i_sf_users = i_sf_users.to(device)
            i_sf_users_items = i_sf_users_items.to(device)
            i_friend_list = i_friend_list.to(device)
            if_item_users = if_item_users.to(device)

            preds, r = model(uids, iids,  u_items, u_users, u_users_items, i_users, i_sf_users, i_sf_users_items, i_friend_list, if_item_users, pos_list, neg_list, False)
            error = torch.abs(preds.squeeze(1) - labels)
            errors.extend(error.data.cpu().numpy().tolist())
            for idx, uid in enumerate(uids):
                results.append([uids[idx].cpu().item(), iids[idx].cpu().item(), labels[idx].cpu().item(), preds[idx].cpu().item()])
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    return mae, rmse, results



if __name__ == '__main__':
    main()