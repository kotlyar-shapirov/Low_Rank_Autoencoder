import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os

from tqdm import tqdm

# Upload model
# from models.R1AE import ConvLRAE, ConvVAE, ConvAE


import torchvision
from torchvision import transforms
from torch.utils.data import Subset, Dataset, DataLoader

import argparse


torch.manual_seed(0)

### argparser!
parser = argparse.ArgumentParser(description='Run AE-type models on training')

parser.add_argument('-m', '--model', type=str, default='LRAE',  help='model type')
parser.add_argument('-d', '--device', type=str, default='cuda:1', help='torch device name. E.g.: cpu, cuda:0, cuda:1')
args = parser.parse_args()



###### Setup script

## Main work parameters
DEVICE = args.device
MODEL_TYPE = args.model

# DEVICE = 'cuda:0' 
# MODEL_TYPE = 'LRAE'


# DATASET_TYPE = 'MNIST'
# DATASET_TYPE = 'CIFAR10'
DATASET_TYPE = 'CelebA'

# Upload model
if DATASET_TYPE in ['MNIST']:
    from models.NIPS_R1AE_MNIST import ConvLRAE, ConvVAE, ConvAE
    print("models were downloaded from 'models.R1AE'")
    IN_FEATURES = 256*2*2
    BOTTLENECK = 128
    OUT_FEATURES = 128*8*8
    MODEL_NAME_PREF = 'test1_NIPS__'
    SAVE_DIR = 'test_NIPS'
elif DATASET_TYPE in ['CIFAR10']:
    from models.NIPS_R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE
    print("models were downloaded from 'models.R1AE_CelebA'")
    IN_FEATURES = 1024*2*2
    BOTTLENECK = 512
    OUT_FEATURES = 1024*4*4
    # MODEL_NAME_PREF = 'test2__'
    # SAVE_DIR = 'test_2_save'
    MODEL_NAME_PREF = 'test_NIPS__'
    SAVE_DIR = 'test_NIPS'
    
elif DATASET_TYPE in ['CelebA', 'CELEBA']:
    from models.NIPS_R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE
    print("models were downloaded from 'models.R1AE_CelebA'")
    IN_FEATURES = 1024*4*4
    BOTTLENECK = 512
    OUT_FEATURES = 1024*8*8
    
    # MODEL_NAME_PREF = 'test2__'
    # SAVE_DIR = 'test_2_save'
    MODEL_NAME_PREF = 'test_NIPS__'
    SAVE_DIR = 'test_NIPS'
else:
   print("Warning! the default models will be uploaded!")
   from models.R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE
   print("models were downloaded from 'models.R1AE_CelebA'") 
   
    
    
    



# MODEL_NAME_PREF = 'test2__'
# SAVE_DIR = 'test_1_save'
# SAVE_DIR = ''

# IN_FEATURES = 256*2*2
# OUT_FEATURES = 128
    
# IN_FEATURES = 1024*4*4
# OUT_FEATURES = 512

# BOTTLENECK = 512
# OUT_FEATURES = 1024*8*8



#
# BATCH_SIZE = 64
# BATCH_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 200
# EPOCHS = 100


# LRAE parameters
N_BINS = 20
DROPOUT = 0.0
TEMP = 0.5
SAMPLING = 'gumbell'


TRAIN_SIZE = -1
TEST_SIZE = -1



ALPHA = 1e-1
# ALPHA = 1e-2
# EPOCH_SAVE = 25 # save and remain
EPOCH_SAVE = 50 # save and remain

EPOCH_SAVE_BACKUP = 5 # save and rewrite 
SHOW_LOSS_BACKUP = 5 # save and rewrite 
LEARNING_RATE = 1e-4

NONLINEARITY = nn.ReLU()







def print_params(param_list, param_names_list):
    for param_name, param in zip(param_names_list, param_list):
        print(f"{param_name}: {param}")
    print()
    

# Show input data
print('Input script data', '\n')
print('Main parameters:')
in_param_list = [SAVE_DIR, DEVICE, MODEL_TYPE, DATASET_TYPE, OUT_FEATURES, EPOCHS]
in_param__names_list = ['SAVE_DIR', 'DEVICE', 'MODEL_TYPE', 'DATASET_TYPE', 'OUT_FEATURES', 'EPOCHS']
print_params(in_param_list, in_param__names_list)
print()
print()


print('All model parameters:')
in_param_list = [OUT_FEATURES, NONLINEARITY, IN_FEATURES,  N_BINS, DROPOUT, TEMP, SAMPLING]
in_param__names_list = ['OUT_FEATURES', 'NONLINEARITY', 'IN_FEATURES',  'N_BINS', 'DROPOUT', 'TEMP', 'SAMPLING']
print_params(in_param_list, in_param__names_list)
print()

print('Training parameters:')
in_param_list = [BATCH_SIZE, LEARNING_RATE, ALPHA,  EPOCHS]
in_param__names_list = ['BATCH_SIZE', 'LEARNING_RATE', 'ALPHA', 'EPOCHS']
print_params(in_param_list, in_param__names_list)
print()

print('Dataset parameters:')
in_param_list = [DATASET_TYPE, TRAIN_SIZE, TEST_SIZE, BATCH_SIZE]
in_param__names_list = ['DATASET_TYPE', 'TRAIN_SIZE', 'TEST_SIZE', 'BATCH_SIZE']
print_params(in_param_list, in_param__names_list)
print()


## other parameters
NUM_WORKERS = 32

# Other parameters
print('Other parameters')
other_param_list = [NUM_WORKERS, EPOCH_SAVE, EPOCH_SAVE_BACKUP, SHOW_LOSS_BACKUP]
other_param_names_list = ['NUM_WORKERS', 'EPOCH_SAVE', 'EPOCH_SAVE_BACKUP', 'SHOW_LOSS_BACKUP']
for param_name, param in zip(other_param_names_list, other_param_list):
    print(f"{param_name}: {param}")
print()



# Service parameters
GOOD_DATASET_TYPE = ['MNIST', 'CIFAR10', 'CelebA', 'CELEBA']
GOOD_MODEL_TYPE = ['VAE', 'AE', 'LRAE']

# Checking parameters
assert MODEL_TYPE in GOOD_MODEL_TYPE, f"Error, bad model type, select from: {GOOD_MODEL_TYPE}"
assert DATASET_TYPE in GOOD_DATASET_TYPE, f"Error, bad dataset type, select from: {GOOD_DATASET_TYPE}"
#############




####### Dataset
dataset_type = DATASET_TYPE
print('\n\n')
print(f'Loading dataset: {dataset_type}')

        
# Torchvision dataset


if DATASET_TYPE in ['MNIST']:
    train_ds = torchvision.datasets.MNIST('./files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    transforms.Resize(32),
                                    torchvision.transforms.ToTensor(),
                                ]))
    test_ds = torchvision.datasets.MNIST('./files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    transforms.Resize(32),
                                    torchvision.transforms.ToTensor(),
                                ]))
    ds_train_size, df_test_size = 60000, 10000
    ds_in_channels = 1
    
    
elif DATASET_TYPE in ['CIFAR10']:
    train_ds = torchvision.datasets.CIFAR10('./files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    transforms.Resize(32),
                                    torchvision.transforms.ToTensor(),
                                ]))
    test_ds = torchvision.datasets.CIFAR10('./files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    transforms.Resize(32),
                                    torchvision.transforms.ToTensor(),
                                ]))
    ds_train_size, df_test_size = 50000, 10000
    ds_in_channels = 3
    
elif DATASET_TYPE in ['CelebA', 'CELEBA']:
    train_ds = torchvision.datasets.CelebA('./files/', split='train', target_type ='attr', download=True,
                                transform=torchvision.transforms.Compose([
                                    transforms.CenterCrop(148),
                                    transforms.Resize([64, 64]),
                                    torchvision.transforms.ToTensor(),
                                ]))
    test_ds = torchvision.datasets.CelebA('./files/', split='valid', target_type ='attr', download=True,
                                transform=torchvision.transforms.Compose([
                                    transforms.CenterCrop(148),
                                    transforms.Resize([64, 64]),
                                    torchvision.transforms.ToTensor(),
                                ]))
    ds_train_size, df_test_size = 162770, 19962  # used validation for test; True test_size = 19867
    ds_in_channels = 3
    
else:
    assert False, f"Error, bad dataset type, select from: {GOOD_DATASET_TYPE}"


num_workers = NUM_WORKERS

# dataset and dataloader
TRAIN_SIZE = ds_train_size if TRAIN_SIZE == -1 else TRAIN_SIZE
TEST_SIZE = df_test_size if TEST_SIZE == -1 else TEST_SIZE
# print(f"TRAIN_SIZE: {TRAIN_SIZE}({ds_train_size})")
# print(f"TEST_SIZE: {TEST_SIZE}({df_test_size})")


BATCH_SIZE = BATCH_SIZE
dl = DataLoader(train_ds, batch_size=BATCH_SIZE,     num_workers=num_workers)
dl_test = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=num_workers)

# #full dataset train
# FULL_TRAIN_SIZE = 2
# dl_full = DataLoader(train_ds, batch_size=FULL_TRAIN_SIZE)
# for x, y in dl_full:
#     X_full_train = x
#     targets = y
#     break

# #full dataset train
# FULL_TEST_SIZE = TEST_SIZE
# dl_full = DataLoader(test_ds, batch_size=FULL_TEST_SIZE)
# for x, y in dl_full:
#     X_full_test = x
#     targets_test = y
#     break

print("Dataset parameters:")
print(f"TRAIN_SIZE: {TRAIN_SIZE}({ds_train_size})")
print(f"TEST_SIZE: {TEST_SIZE}({df_test_size})")
for param, param_name in zip([BATCH_SIZE], ["BATCH_SIZE"] ):
    print(f"{param_name} = {param}")

print(f"{DATASET_TYPE} dataset logs:")
print("Img channel:", ds_in_channels)
# print(X_full_train.shape)
# print(torch.max(X_full_train))
# print(targets.unique(return_counts=True))

print('\n\n')

# ds_in_channels = X_full_train.shape[1]
###################



###################### Initialization of the model
device = DEVICE
model_name = MODEL_NAME_PREF + f"{DATASET_TYPE}__{MODEL_TYPE}__{BOTTLENECK}__{ALPHA}"

print('\n\n')


print("Initialization of the model")
print("model_name: ", model_name, '\n\n' )
if MODEL_TYPE == 'LRAE':
    GRID = torch.arange(1,N_BINS+1).to(device)/N_BINS
    model = ConvLRAE(IN_FEATURES, BOTTLENECK, OUT_FEATURES, N_BINS, GRID, dropout=DROPOUT, nonlinearity=NONLINEARITY,
                sampling=SAMPLING, temperature=TEMP, in_channels=ds_in_channels).to(device)
elif MODEL_TYPE == 'VAE':
    model = ConvVAE(IN_FEATURES, BOTTLENECK, OUT_FEATURES, nonlinearity=NONLINEARITY, in_channels=ds_in_channels).to(device)
elif MODEL_TYPE == 'AE':
    model = ConvAE(IN_FEATURES, BOTTLENECK, OUT_FEATURES, nonlinearity=NONLINEARITY, in_channels=ds_in_channels).to(device)
else:
    assert False, f"Error, bad model type, select from: {GOOD_MODEL_TYPE}"
      
print(f"{MODEL_TYPE} was initialized")
PATH = os.path.join(SAVE_DIR, model_name)
print('Save PATH:', PATH)
#################################



######## Training
print("Training of the model")


# setup training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


PATH = PATH
EPOCHS = EPOCHS


loss_list_train = []
loss_train_cum = 0

loss_list_test = []
loss_test_cum = 0
i = 0
loss = 0


alpha_kl = ALPHA
alpha_entropy = ALPHA
if MODEL_TYPE == 'LRAE':
    alpha_entropy *= (OUT_FEATURES/8)**0.5
    print(f"Alpha update = {(OUT_FEATURES/8)**0.5: .3f}")
    print(f"Entropy alpha = {alpha_entropy: .3e}")


epoch_save_backup = EPOCH_SAVE_BACKUP
epoch_save = EPOCH_SAVE
show_loss_backup = SHOW_LOSS_BACKUP


# Training
model.train()
optimizer.zero_grad()
torch.cuda.empty_cache()

for epoch in tqdm(range(EPOCHS)):
    # Forward pass: Compute predicted y by passing x to the model
        
    # Training
    model.train() # Model to train
    for x_batch, y_batch in dl:

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # model forward
        # 2d downsampling
        x_down = model.down(x_batch)
        B, C, H, W = x_down.shape
        x_flat = x_down.view(B,C*H*W)
        
        encoded_out_dim, factors_probability = model.low_rank.low_rank_pants(x_flat)
        decoded_1d = model.low_rank.decoder(encoded_out_dim)
        
        # print(B, C, H, W )
        # 2d upsampling
        if DATASET_TYPE in ['MNIST']:
            C, H, W = C//2, H*4, W*4
        elif DATASET_TYPE in ['CELEBA', 'CelebA', 'CIFAR10']:
            C, H, W = C, H*2, W*2
        else:
            assert 0, f'Error, Bad DATASET_TYPE={DATASET_TYPE}, should be in {GOOD_DATASET_TYPE}'
        
        # print(B, C, H, W )
        decoded_2d_small = decoded_1d.view(B, C, H, W)
        decoded_2d = model.up(decoded_2d_small)
        
        # loss

#         loss_entropy = torch.sum(torch.log(factors_probability+1e-9)*factors_probability,dim=-1)
        # factors_probability = nn.Softmax(dim=-1)(factors_probability)
        # loss_entropy = torch.sum(torch.log(factors_probability+1e-9)*factors_probability,dim=-1)
        loss = criterion(decoded_2d.view(-1), x_batch.view(-1)) 
        if MODEL_TYPE == 'VAE':
            loss = torch.nn.functional.binary_cross_entropy(decoded_2d, x_batch)
            loss -= alpha_kl*factors_probability.mean()  # KL loss
            
        if MODEL_TYPE == 'LRAE':
            loss = torch.nn.functional.binary_cross_entropy(decoded_2d, x_batch)
            factors_probability = nn.Softmax(dim=-1)(factors_probability)
            loss_entropy = torch.sum(torch.log(factors_probability+1e-9)*factors_probability,dim=-1)            
            loss += alpha_entropy*torch.mean(torch.exp(loss_entropy)) # entropy loss
            
            
          
            
            
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accumulate loss
        loss_train_cum += loss.item()
        
        # validation and saving
        i += 1
        if i % 100 == 0:
            loss_list_train.append(loss_train_cum/100)
            loss_train_cum = 0
            with torch.no_grad():
                model.eval() # put to eval
                
                for x_batch, y_batch in dl_test:
                    # model forward
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    x_decoded = model(x_batch)

                    loss_test = criterion(x_decoded.view(-1), x_batch.view(-1))
                    loss_test_cum += loss_test.item()
                    
            assert torch.isnan(x_decoded).sum() == 0, f"Error! Nan values ({torch.isnan(x_decoded).sum()}) in models output"
      
            # save to list
            loss_list_test.append(loss_test_cum/len(dl_test))
            loss_test_cum = 0
          
    # backup saving  
    if epoch%epoch_save == 0:
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'loss_list_train': loss_list_train,
            'loss_list_test': loss_list_test,
            
            }, PATH + f"__{epoch}.pth")
        epoch_previous = epoch
            
    # backup saving  
    if epoch%epoch_save_backup == 0:
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'loss_list_train': loss_list_train,
            'loss_list_test': loss_list_test,
            
            }, PATH + f"__backup.pth")
        epoch_previous = epoch
      
    # loss printing        
    if (epoch % show_loss_backup == (show_loss_backup-1)) or (epoch == EPOCHS -1):
        fig = plt.figure(figsize=(6,3))
        plt.plot(loss_list_train, alpha=0.5, label='train')
        plt.plot(loss_list_test, alpha=0.5, label='test')
        plt.legend()
        plt.savefig( PATH  + "_loss.jpg")
        pass
            
            
        



print("Finishing of the training...")
torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'loss_list_train': loss_list_train,
        'loss_list_test': loss_list_test,
        
        }, PATH + f"__{epoch}__end.pth")

print("Model training was successfully finished and saved!")





#######################
