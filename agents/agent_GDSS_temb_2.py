import sys
sys.path.append('/home/zhiwen/architec/modify_door/')
#sys.path.append('D:/code/zhiwen/architec/modify_door/')
import argparse
import glob
import os
import warnings
import csv
import time
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.model_GDSS_time import ScoreNetworkX, ScoreNetworkA, time_embedding
from data_loader.load_dataset_no_edge_class import get_dataset
from utils.ema import ExponentialMovingAverage
from utils.losses_time import load_loss_fn

parser = argparse.ArgumentParser()
parser.add_argument('--exp_model', type=str, default='GDSS', help='name1 when saved')
parser.add_argument('--exp_model_detail', type=str, default='+temb_3', help='name2 when saved')

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=64, help='batch size original = 128')
parser.add_argument('--max_node_num', type=int, default=40, help='maximum number of node')
parser.add_argument('--max_feat_num', type=int, default=11, help='maximum number of feature')

parser.add_argument('--depth', type=int, default=5, help='depth of model')

parser.add_argument('--adim', type=int, default=32, help='')
parser.add_argument('--nhid', type=int, default=32, help='hidden size of model')
parser.add_argument('--num_layers', type=int, default=8, help='')
parser.add_argument('--num_linears', type=int, default=2, help='')

parser.add_argument('--c_init_x', type=int, default=2, help='')
parser.add_argument('--c_hid_x', type=int, default=8, help='')
parser.add_argument('--c_final_x', type=int, default=4, help='')

parser.add_argument('--c_init_adj', type=int, default=4, help='')
parser.add_argument('--c_hid_adj', type=int, default=16, help='')
parser.add_argument('--c_final_adj', type=int, default=8, help='')

parser.add_argument('--temb_size', type=int, default=8, help='')
parser.add_argument('--reduce_mean', type=bool, default=False, help='')
parser.add_argument('--beta_min_x', type=float, default=0.1, help='')
parser.add_argument('--beta_max_x', type=float, default=1.0, help='')
parser.add_argument('--num_scales_x', type=int, default=1000, help='')

parser.add_argument('--is_temb_x', type=bool, default='True', help='')
parser.add_argument('--is_temb_adj', type=bool, default='True', help='')

parser.add_argument('--beta_min_adj', type=float, default=0.2, help='')
parser.add_argument('--beta_max_adj', type=float, default=1.0, help='')
parser.add_argument('--num_scales_adj', type=int, default=1000, help='')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--ema', type=float, default=0.999, help='ema')
parser.add_argument('--lr_decay', type=float, default=0.999, help='')
parser.add_argument('--grad_norm', type=float, default=1.0, help='')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--eps', type=float, default=1.0e-5, help='')
args = parser.parse_args(args=[])


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

path = '/home/zhiwen/architec/modify_door/dataset/nx_file/'
#path = 'D:/code/zhiwen/architec/modify_door/dataset/nx_file/'
name = 'att_no_edge_class'
### prepare dataset
train_dataset, test_dataset = get_dataset(path, name, args.max_node_num, args.max_feat_num)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

### prepare model
model_x = ScoreNetworkX(args.max_feat_num, args.depth, args.nhid, args.num_linears,
                        args.c_init_x, args.c_hid_x, args.c_final_x, args.adim, temb_dim=args.temb_size*2, is_temb=args.is_temb_x).to(args.device)

model_adj = ScoreNetworkA(args.max_feat_num, args.max_node_num, args.nhid, args.num_layers, args.num_linears, 
                    args.c_init_adj, args.c_hid_adj, args.c_final_adj, args.adim, is_temb=args.is_temb_adj).to(args.device)
model_t = time_embedding(temb_size=args.temb_size).to(args.device)

x_params = list(model_x.parameters()) + list(model_t.parameters())
optimizer_x = torch.optim.Adam(model_x.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler_x = torch.optim.lr_scheduler.ExponentialLR(optimizer_x, gamma=args.lr_decay)
ema_x = ExponentialMovingAverage(model_x.parameters(), decay=args.ema)

adj_params = list(model_adj.parameters()) + list(model_t.parameters())
optimizer_adj = torch.optim.Adam(model_adj.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler_adj = torch.optim.lr_scheduler.ExponentialLR(optimizer_adj, gamma=args.lr_decay)
ema_adj = ExponentialMovingAverage(model_adj.parameters(), decay=args.ema)

loss_fn = load_loss_fn(args)

### create experiments
def make_dir(dirs):
    try:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    except Exception as err:
        print("create_dirs error!")
        print(dirs)
        exit()

experiments = '/home/zhiwen/architec/modify_door/experiments/' + args.exp_model + '/' + args.exp_model_detail + '/'
print(experiments)
#experiments = 'D:/code/zhiwen/architec/modify_door/experiments/' + args.exp_model + '/' + args.exp_model_detail + '/'
checkpoint_path = experiments + 'checkpoint/'
checkpoint_current = checkpoint_path + 'checkpoint.pth.tar'
checkpoint_best = checkpoint_path + 'checkpoint_best.pth.tar'
log_path_train = experiments + 'train_log.csv'
log_path_val = experiments + 'val_log.csv'

make_dir(checkpoint_path)

start_epoch = 0

### Load checkpoint.
if os.path.exists(checkpoint_current):
    print('==> Resuming from checkpoint..')
    print("=> loading checkpoint")
    checkpoint = torch.load(checkpoint_current)
    
    model_x.load_state_dict(checkpoint['model_x'])
    model_adj.load_state_dict(checkpoint['model_adj'])
    model_t.load_state_dict(checkpoint['model_t'])
    ema_x.load_state_dict(checkpoint['ema_x'])
    ema_adj.load_state_dict(checkpoint['ema_adj'])
    optimizer_x.load_state_dict(checkpoint['optimizer_x'])
    optimizer_adj.load_state_dict(checkpoint['optimizer_adj'])
    min_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch'] + 1
else:
    pass

def train(epoch, train_loader):
    model_x.train()
    model_adj.train()
    model_t.train()

    loss_x_train = 0.0
    loss_adj_train = 0.0
    total = 0
    print('=========================train===============================')

    for batch_idx, data in enumerate(train_loader):
        optimizer_x.zero_grad()
        optimizer_adj.zero_grad()
        
        x = data[0].to(args.device)
        adj = data[1].to(args.device)
        flags = data[2].to(args.device)
        
        total += len(x)
        
        loss_subject = (x, adj, flags)
        loss_x, loss_adj = loss_fn(model_x, model_adj, model_t, *loss_subject)
        
        loss_x_train += loss_x.item()
        loss_adj_train += loss_adj.item()
        
        loss_x.backward(retain_graph=True)
        loss_adj.backward(retain_graph=True)
        
        torch.nn.utils.clip_grad_norm_(model_x.parameters(), args.grad_norm)
        torch.nn.utils.clip_grad_norm_(model_adj.parameters(), args.grad_norm)

        optimizer_x.step()
        optimizer_adj.step()

        # -------- EMA update --------
        ema_x.update(model_x.parameters())
        ema_adj.update(model_adj.parameters())
        
        scheduler_x.step()
        scheduler_adj.step()
        
        if batch_idx % 4 == 0:
            print('Train Epoch: {} [{:04d}/{:04d} ({:.2%})] loss_x_train: {:.6f} | loss_adj_train: {:.6f}'.format(
                epoch, total, len(train_loader.dataset),
                total / len(train_loader.dataset),
                loss_x / len(x),
                loss_adj / len(x))) 
             
    return loss_x_train / len(train_loader.dataset), loss_adj_train / len(train_loader.dataset)

def test(loader):
    model_x.eval()
    model_adj.eval()    
    model_t.eval()  
    
    loss_x_test = 0.0
    loss_adj_test = 0.0
    
    for batch_idx, data in enumerate(loader):
        x = data[0].to(args.device)
        adj = data[1].to(args.device)
        flags = data[2].to(args.device)
        
        loss_subject = (x, adj, flags)
        
        with torch.no_grad():
            ema_x.store(model_x.parameters())
            ema_x.copy_to(model_x.parameters())
            ema_adj.store(model_adj.parameters())
            ema_adj.copy_to(model_adj.parameters())           
            
            loss_x, loss_adj = loss_fn(model_x, model_adj, model_t, *loss_subject)        
        
            loss_x_test += loss_x.item()
            loss_adj_test += loss_adj.item()
            
            ema_x.restore(model_x.parameters())
            ema_adj.restore(model_adj.parameters())
        
    return loss_x_test / len(loader.dataset), loss_adj_test / len(loader.dataset)

if __name__ == '__main__':
#def main():
    warnings.filterwarnings(action='ignore')
    
    min_loss = 1e10
    
    
    for epoch in range(start_epoch, args.epochs):
        t = time.time()
        
        ### train
        loss_x_train, loss_adj_train = train(epoch, train_loader)
        
        ### val
        loss_x_val, loss_adj_val = test(test_loader)
        total_loss = loss_x_val + loss_adj_val
        
        
        print('=============================================================')
        print('train',
            'Epoch: {:04d}'.format(epoch),
            'loss_x_train: {:.6f}'.format(loss_x_train),
            'loss_adj_train: {:.6f}'.format(loss_adj_train))  
        
        print('val',
            'Epoch: {:04d}'.format(epoch),
            'loss_x_val: {:.6f}'.format(loss_x_val),
            'loss_adj_val: {:.6f}'.format(loss_adj_val))       
        print('=============================================================')
    
        ### save train.csv
        if os.path.exists(log_path_train) == False:
            with open(log_path_train, 'w', newline='') as train_writer_csv:
                header_list = ['epoch', 'loss_x', 'loss_adj']
                train_writer = csv.DictWriter(train_writer_csv, fieldnames=header_list)
                train_writer.writeheader()
        with open(log_path_train, 'a', newline='') as train_writer_csv:
            train_writer = csv.writer(train_writer_csv)
            train_writer.writerow([epoch, str(loss_x_train), str(loss_adj_train)])
        
        ### sava val.csv 
        if os.path.exists(log_path_val) == False:
            with open(log_path_val, 'w', newline='') as val_writer_csv:
                header_list = ['epoch', 'loss_x', 'loss_adj']
                val_writer = csv.DictWriter(val_writer_csv, fieldnames=header_list)
                val_writer.writeheader()
        with open(log_path_val, 'a', newline='') as val_writer_csv:
            val_writer = csv.writer(val_writer_csv)
            val_writer.writerow([epoch, str(loss_x_val), str(loss_adj_val)])
        
        state = {'model_x': model_x.state_dict(),
                'model_adj': model_adj.state_dict(),
                'model_t': model_t.state_dict(),
                'ema_x': ema_x.state_dict(),
                'ema_adj': ema_adj.state_dict(),
                'loss': total_loss,
                'epoch': epoch,
                'optimizer_x': optimizer_x.state_dict(),
                'optimizer_adj': optimizer_adj.state_dict()}
                          
        ### save checkpoint
        torch.save(state, checkpoint_current)     
        
        if total_loss < min_loss :
            print('Saving..')
            torch.save(state, checkpoint_best)
            min_loss = total_loss   