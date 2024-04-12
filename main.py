import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import wandb

import torch
from model import CNN1D
from loader import Loader
from trainer import Trainer
from tester import Tester
from model import Discriminator, Regressor
import math

from torch.optim import lr_scheduler
import itertools

import argparse


class CustomLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer, last_epoch=-1)
    
    def get_lr(self):
        factor = 1.0 / (1.0 + 10 * self.last_epoch) ** 0.75
        return [base_lr * factor for base_lr in self.base_lrs]

# Example usage:
# Create your optimizer

# Create your custom learning rate scheduler





if __name__ == '__main__':
    
    random_seed = 1905
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', type=int, default=0, help='use wandb or not')
    parser.add_argument('--source', type=str, default='FD001', help='source domain')
    parser.add_argument('--target', type=str, default='FD002', help='target domain')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--train', type=int, default=1, help='train or not')
    parser.add_argument('--gpu', type=str, default='mps', help='gpu id')
    #print parser.parse_args()
    print(parser.parse_args())
    
    args = parser.parse_args()
    if args.wandb:
    # inir wandb
        wandb.init(project='DADA')
        # add key
        key = np.loadtxt('key.txt', dtype=str)
        wandb.login(key=key)
    
    # test if gpu is available
    if args.gpu == 'mps':
        device = 'mps'
    else:
        device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    
    seq_len = 30  
    batch_size = 32
    x0 = 0.001  
    early_rul = 130  
    
    # Load data
    source_data, source_labels, source_test_data, source_test_labels = Loader(name=args.source, seq_len=seq_len, shift=1, early_rul=early_rul).load_data()
    target_data, target_labels, target_test_data, target_test_labels = Loader(name=args.target, seq_len=seq_len, shift=1, early_rul=early_rul).load_data()
    
    # # Split source and target data
    # source_data_X_train, source_data_X_val, source_data_y_train, source_data_y_val = train_test_split(source_data, source_labels, test_size=0.2, random_state=random_seed)
    # target_data_X_test, target_data_X_val, target_data_y_test, target_data_y_val = train_test_split(target_data, target_labels, test_size=0.2, random_state=random_seed)
    
    # # Create dataloaders
    # source_data_train = torch.utils.data.TensorDataset(torch.Tensor(source_data_X_train), torch.Tensor(source_data_y_train))
    # source_data_val = torch.utils.data.TensorDataset(torch.Tensor(source_data_X_val), torch.Tensor(source_data_y_val))
    
    # target_data_test = torch.utils.data.TensorDataset(torch.Tensor(target_data_X_test), torch.Tensor(target_data_y_test))
    # target_data_val = torch.utils.data.TensorDataset(torch.Tensor(target_data_X_val), torch.Tensor(target_data_y_val))
    
    
    source_data_train = torch.utils.data.TensorDataset(torch.Tensor(source_data), torch.Tensor(source_labels))
    source_data_test = torch.utils.data.TensorDataset(torch.Tensor(source_test_data), torch.Tensor(source_test_labels))
    
    target_data_train = torch.utils.data.TensorDataset(torch.Tensor(target_data), torch.Tensor(target_labels))
    target_data_test = torch.utils.data.TensorDataset(torch.Tensor(target_test_data), torch.Tensor(target_test_labels))
    
    
    
    source_train = torch.utils.data.DataLoader(source_data_train, batch_size = 32, shuffle = True)
    source_test = torch.utils.data.DataLoader(source_data_test, batch_size = 1000, shuffle = True)   
    
    target_train = torch.utils.data.DataLoader(target_data_train, batch_size = 32, shuffle = True)
    target_test = torch.utils.data.DataLoader(target_data_test, batch_size = 1000, shuffle = True)
    
    
    
    
    
    dim = source_data.shape[2]
    
   
    
    
    input_data = torch.randn(batch_size, seq_len, dim)  # Example input data
    model = CNN1D(input_channels=dim)
    discriminator = Discriminator()
    regressor = Regressor()
    # optimize model and regressor
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), regressor.parameters()), lr=x0, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + 10 * epoch) ** 0.75)
    
    
  
    
    if args.train:
        trainer = Trainer(model=model,
                        regressor=regressor,
                        discriminator=discriminator,
                        model_optimizer=optimizer,
                        model_scheduler=lr_scheduler,
                        print_every=500,
                        epochs=args.epochs,
                        device=args.gpu,
                        prefix=f'{args.source}_{args.target}',
                        early_rul=early_rul,
                        wandb=args.wandb)
        best_score, best_RMSE = trainer.train(source_train, source_test, target_train, target_test)
        print('==== TRAINING FINISHED ====')
    
    #load model
    
    checkpoint = torch.load(f'checkpoints/{args.source}_{args.target}_best_score.pth')

    # Extract the model's state dict
   
    

    # Load the model state dict
    model.load_state_dict(checkpoint['fe_state_dict'])
    regressor.load_state_dict(checkpoint['reg_state_dict'])
    # discriminator.load_state_dict(checkpoint['dis_state_dict'])
    model.eval()
    regressor.eval
    # discriminator.eval()
    model.to(device)
    tester = Tester(model=model, regressor=regressor,early_rul=early_rul, device=device)
    source_score, source_RMSE = tester.test(source_test)
    target_score, target_RMSE = tester.test(target_test)
    source_res = [source_score, source_RMSE.cpu().numpy()]
    target_res = [target_score, target_RMSE.cpu().numpy()]
   
    
    #        | SOURCE | TARGET
    # ------------------------
    # Score  |        |  
    # RMSE   |        |  
    
    df = pd.DataFrame([source_res, target_res], columns=['Score', 'RMSE'], index=[args.source, args.target])
    print(f'======= {args.source} -> {args.target} =======')
    print(df)
    
        
        
        
