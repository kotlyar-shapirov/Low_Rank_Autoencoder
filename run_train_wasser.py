import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os

from tqdm import tqdm



import torchvision
from torchvision import transforms
from torch.utils.data import Subset, Dataset, DataLoader

import argparse


from utils.script_utils import select_dataset, init_model, setup_dataset_training, save_checkpoint_from
from utils.loss import wasser_loss, entropy_loss, wasser_loss_diff
from main_utils import get_models_class_list, get_base_model_parameters


from timeit import default_timer as timer
timer_start = timer() # start timer

torch.manual_seed(0)
# Defaults
GOOD_DATASET_TYPE = ['MNIST', 'CIFAR10', 'CELEBA']
GOOD_MODEL_TYPE = ['VAE', 'AE', 'LRAE']
GOOD_ARCHITECTURE_TYPE = ['V1', 'NIPS']



### argparser!
parser = argparse.ArgumentParser(description='Run AE-type models on training')

parser.add_argument('-m', '--model', type=str, default='LRAE',  
                    help=f'model type: {GOOD_MODEL_TYPE}')
parser.add_argument('-a', '--architecture', type=str, default='NIPS',  
                    help=f'model architecture type: {GOOD_ARCHITECTURE_TYPE}')
parser.add_argument('-D', '--dataset', type=str, default='MNIST',  
                    help=f'dataset type: {GOOD_DATASET_TYPE}')

parser.add_argument('-A', '--alpha', type=str, default=0.1,  
                    help=f'alpha coeff')
parser.add_argument('-L', '--dist_loss', type=str, default='entropy',  
                    help=f'distribution loss type')
parser.add_argument('-b', '--batch_size', type=str, default=32,  
                    help=f'Batch size')

parser.add_argument('-d', '--device', type=str, default='cuda:1', 
                    help=f'torch device name. E.g.: cpu, cuda:0, cuda:1')
args = parser.parse_args()



###### Setup script

## Main work parameters
DEVICE = args.device
MODEL_TYPE = args.model

ARCHITECTURE_TYPE = args.architecture
DATASET_TYPE = args.dataset

ALPHA = float(args.alpha)
DISTRIBUTION_LOSS = args.dist_loss
assert DISTRIBUTION_LOSS in ['entropy', 'wasser'], f'Choose dist loss from: entropy, wasser'
BATCH_SIZE = int(args.batch_size)

EPOCHS = 201
LEARNING_RATE = 1e-4

#### setup runs
print("Setup runs")
if DATASET_TYPE.upper() in ['MNIST']:
    MODEL_NAME_PREF = f'test_bl_NIPS_{BATCH_SIZE}_{LEARNING_RATE}__'
    SAVE_DIR = 'test_NIPS'
    
elif DATASET_TYPE.upper() in ['CIFAR10']:
    MODEL_NAME_PREF = 'test_NIPS__'
    SAVE_DIR = 'test_NIPS'
    
elif DATASET_TYPE.upper() in ['CELEBA']: 
    MODEL_NAME_PREF = f'test_bl_NIPS_{BATCH_SIZE}_{LEARNING_RATE}__'
    SAVE_DIR = 'test_NIPS'
else:
   print("Warning! the default run setups was not setuped!")
   
# MODEL_NAME_PREF = 'test_NIPS__'
# SAVE_DIR = 'test_NIPS'
################### 



models_class_list = get_models_class_list(DATASET_TYPE, ARCHITECTURE_TYPE) 




 
   
   
### Model parameters

# setup model parameters

models_params = get_base_model_parameters(DATASET_TYPE, ARCHITECTURE_TYPE)
BOTTLENECK =  models_params['BOTTLENECK']      
   
# other Model parameters
NONLINEARITY = nn.ReLU()
models_params['NONLINEARITY'] = nn.ReLU()
###


# LRAE parameters
N_BINS, DROPOUT, TEMP, SAMPLING = 20, 0.0, 0.5, 'gumbell'
models_params = models_params | {'N_BINS': N_BINS, 'DROPOUT':DROPOUT, 'SAMPLING':SAMPLING, 'TEMP': TEMP}
##


TRAIN_SIZE = -1
TEST_SIZE = -1


# EPOCH_SAVE = 50 # save and remain
EPOCH_SAVE = 25 # save and remain

EPOCH_SAVE_BACKUP = 5 # save and rewrite 
SHOW_LOSS_BACKUP = 5 # save and rewrite 










def print_params(param_list, param_names_list):
    for param_name, param in zip(param_names_list, param_list):
        print(f"{param_name}: {param}")
    print()
    

# Show input data
print('Input script data', '\n')
print('Main parameters:')
in_param_list = [SAVE_DIR, DEVICE, MODEL_TYPE, DATASET_TYPE,  ARCHITECTURE_TYPE, BOTTLENECK, EPOCHS]
in_param__names_list = ['SAVE_DIR', 'DEVICE', 'MODEL_TYPE', 'DATASET_TYPE', 'ARCHITECTURE_TYPE', 'BOTTLENECK', 'EPOCHS']
print_params(in_param_list, in_param__names_list)
print()
print()


print('All model parameters:')
print_params(models_params.values(), models_params.keys())
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





# Checking parameters
assert MODEL_TYPE.upper() in GOOD_MODEL_TYPE, f"Error, bad model type, select from: {GOOD_MODEL_TYPE}"
assert DATASET_TYPE.upper() in GOOD_DATASET_TYPE, f"Error, bad dataset type, select from: {GOOD_DATASET_TYPE}"
assert ARCHITECTURE_TYPE.upper() in GOOD_ARCHITECTURE_TYPE, f"Error, bad model architecture type, select from: {GOOD_ARCHITECTURE_TYPE}"
#############




####### Dataset
dataset_type = DATASET_TYPE
print('\n\n')
print(f'Loading dataset: {dataset_type}', flush=True)


train_ds, test_ds, ds_train_size, df_test_size, ds_in_channels = select_dataset(DATASET_TYPE, GOOD_DATASET_TYPE)
dl, dl_test = setup_dataset_training(train_ds, test_ds, BATCH_SIZE, num_workers=NUM_WORKERS)  
models_params['DS_IN_CHANNELS'] = ds_in_channels


TRAIN_SIZE = ds_train_size if TRAIN_SIZE == -1 else TRAIN_SIZE
TEST_SIZE = df_test_size if TEST_SIZE == -1 else TEST_SIZE
print("Dataset parameters:")
print(f"TRAIN_SIZE: {TRAIN_SIZE}({ds_train_size})")
print(f"TEST_SIZE: {TEST_SIZE}({df_test_size})")
for param, param_name in zip([BATCH_SIZE], ["BATCH_SIZE"] ):
    print(f"{param_name} = {param}")

print(f"{DATASET_TYPE} dataset logs:")
print("Img channel:", ds_in_channels)

print('\n\n')
###################



###################### Initialization of the model

# DEVICE, 

device = DEVICE
model_name = MODEL_NAME_PREF + f"{DATASET_TYPE}__{MODEL_TYPE}__{BOTTLENECK}__{ALPHA}"

print('\n\n')
print("Initialization of the model")
print("model_name: ", model_name, '\n\n' )

model = init_model(MODEL_TYPE, GOOD_MODEL_TYPE,  models_class_list, models_params, device)

print(f"{MODEL_TYPE} was initialized")
PATH = os.path.join(SAVE_DIR, model_name)
print('Save PATH:', PATH, flush=True)
#################################





######## Training
print("Training of the model", flush=True)
device = DEVICE


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

#time
epoch_time_list = []
epoch_t1 = None


alpha_kl = ALPHA
alpha_entropy = ALPHA
# if MODEL_TYPE == 'LRAE':
#     alpha_entropy *= (OUT_FEATURES/8)**0.5
#     print(f"Alpha update = {(OUT_FEATURES/8)**0.5: .3f}")
#     print(f"Entropy alpha = {alpha_entropy: .3e}")


epoch_save_backup = EPOCH_SAVE_BACKUP
epoch_save = EPOCH_SAVE
show_loss_backup = SHOW_LOSS_BACKUP





# Training
model.train()
optimizer.zero_grad()
torch.cuda.empty_cache()

for epoch in tqdm(range(EPOCHS)):
    
    # time
    epoch_t2 = timer()
    if epoch_t1 is not None:
        epoch_time_list += [epoch_t2 - epoch_t1]
    epoch_t1 = epoch_t2
    
    
    # Forward pass: Compute predicted y by passing x to the model
        
    # Training
    model.train() # Model to train
    epoch_t1 = timer()
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
        loss = criterion(decoded_2d.view(-1), x_batch.view(-1)) 
        if MODEL_TYPE == 'VAE':
            loss = torch.nn.functional.binary_cross_entropy(decoded_2d, x_batch)
            loss -= alpha_kl*factors_probability.mean()  # KL loss
            
        if MODEL_TYPE == 'LRAE':
            # OLD MSE-LIKE LOSS
            # loss = torch.nn.functional.binary_cross_entropy(decoded_2d, x_batch)

            # NEW RECURSIVE WASSERSTEIN LOSS
            x_down_rec = model.down(decoded_2d)
            B, C, H, W = x_down_rec.shape
            x_flat_rec = x_down_rec.view(B,C*H*W)
            encoded_out_dim_rec, factors_probability_rec = model.low_rank.low_rank_pants(x_flat_rec)
            loss = wasser_loss_diff(factors_probability, factors_probability_rec)

            if DISTRIBUTION_LOSS == 'entropy':
                loss += alpha_entropy*entropy_loss(factors_probability)
            elif DISTRIBUTION_LOSS == 'wasser':
                loss += alpha_entropy*wasser_loss(factors_probability)
                      
            
            
            
          
            
            
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
    if epoch % epoch_save == 0:
        
        save_checkpoint_from(PATH + f"__backup__{epoch}.pth", model, optimizer,  epoch=epoch, loss=loss, 
                    loss_list_train=loss_list_train, loss_list_test=loss_list_test,
                    epoch_time_list=epoch_time_list)
        
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss,
        #     'loss_list_train': loss_list_train,
        #     'loss_list_test': loss_list_test,
        #     'epoch_time_list': epoch_time_list,
            
        #     }, PATH + f"__{epoch}.pth")
        epoch_previous = epoch
            
    # backup saving  
    if epoch%epoch_save_backup == 0:
        
        save_checkpoint_from(PATH + f"__backup.pth", model, optimizer,  epoch=epoch, loss=loss, 
                    loss_list_train=loss_list_train, loss_list_test=loss_list_test,
                    epoch_time_list=epoch_time_list)
        
        
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss,
        #     'loss_list_train': loss_list_train,
        #     'loss_list_test': loss_list_test,
        #     'epoch_time_list': epoch_time_list,
            
        #     }, PATH + f"__backup.pth")
        epoch_previous = epoch
      
    # loss printing        
    if (epoch % show_loss_backup == (show_loss_backup-1)) or (epoch == EPOCHS -1):
        fig = plt.figure(figsize=(6,3))
        plt.plot(loss_list_train, alpha=0.5, label='train')
        plt.plot(loss_list_test, alpha=0.5, label='test')
        plt.legend()
        plt.savefig( PATH  + "_loss.jpg")
        plt.close()
        pass
            
            
        



print("Finishing of the training...")

save_checkpoint_from(PATH + f"__{epoch}__end.pth", model, optimizer,  epoch=epoch, loss=loss, 
                    loss_list_train=loss_list_train, loss_list_test=loss_list_test,
                    epoch_time_list=epoch_time_list)


# torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#         'loss_list_train': loss_list_train,
#         'loss_list_test': loss_list_test,
#         'epoch_time_list': epoch_time_list,
        
#         }, PATH + f"__{epoch}__end.pth")


#######################


timer_end = timer()
print(f"Elapsed time: {timer_end - timer_start:.2f} second") # Time in seconds, e.g. 5.38091952400282
print("Mean full epoch time:", f"{np.asarray(epoch_time_list).mean().item():.1f}")
print(f"Model training for {model_name} was successfully finished and saved!")
print('\n\n\n\n\n')

