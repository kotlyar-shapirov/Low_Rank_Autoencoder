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

from models.evaluation import inf_by_layers, check_reconstruction, gen_idx_for_batches, display_datasets
from models.evaluation import gen_gm_dataset, update_FID_class, ManualFID, prepare_to_FID, get_inf_by_layers
from torchmetrics.image.fid import FrechetInceptionDistance as tm_FrechetInceptionDistance
from torcheval.metrics import FrechetInceptionDistance

from sklearn.mixture import GaussianMixture

from timeit import default_timer as timer

timer_start = timer() # start timer
### argparser!
parser = argparse.ArgumentParser(description='Run AE-type models on training')

parser.add_argument('-l', '--load_path', type=str, help='Load model checkpoint path [.pth]')
parser.add_argument('-o', '--out', type=str, default='evaluation', help='Output dir. If -1, out dir is a dir from load_path')
# parser.add_argument('-m', '--model', type=str, default='LRAE',  help='model type')
parser.add_argument('-d', '--device', type=str, default='cuda:1', help='torch device name. E.g.: cpu, cuda:0, cuda:1')
args = parser.parse_args()



###### Setup script


## Main work parameters
DEVICE = args.device
# MODEL_TYPE = args.model
LOAD_PATH = args.load_path
OUT_DIR = args.out if str(-1) != args.out else os.path.dirname(LOAD_PATH)





load_path = LOAD_PATH
model_name = os.path.basename(load_path)
model_dir = os.path.dirname(load_path)
MODEL_DIR = model_dir
MODEL_NAME = model_name

model_name_in_list = model_name.split('__')

DATASET_TYPE = model_name_in_list[1]
MODEL_TYPE = model_name_in_list[2]
OUT_FEATURES = int(model_name_in_list[3])
ALPHA = float(model_name_in_list[4])
EPOCHS = model_name_in_list[5]


# Upload model
if DATASET_TYPE in ['MNIST']:
    # from models.R1AE import ConvLRAE, ConvVAE, ConvAE
    from models.R1AE import ConvLRAE, ConvVAE, ConvAE
    print("models were downloaded from 'models.R1AE'")
    IN_FEATURES = 256*2*2
    C_H_W = [256, 2, 2]
    DS_IN_CHANNELS = 1

elif DATASET_TYPE in ['CIFAR10']:
    from models.R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE
    print("models were downloaded from 'models.R1AE_CelebA'")
    IN_FEATURES = 1024*2*2
    C_H_W = [1024, 2, 2]
    DS_IN_CHANNELS = 3

    
elif DATASET_TYPE in ['CelebA', 'CELEBA']:
    from models.R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE
    print("models were downloaded from 'models.R1AE_CelebA'")
    IN_FEATURES = 1024*4*4
    C_H_W = [1024, 4, 4]
    DS_IN_CHANNELS = 3
else:
   print("Warning! the default models will be uploaded!")
   from models.R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE
   print("models were downloaded from 'models.R1AE_CelebA'") 
   
    
NONLINEARITY = nn.ReLU()

TRAIN_SIZE, TEST_SIZE = -1, -1


# LRAE parameters
N_BINS = 20
DROPOUT = 0.0
TEMP = 0.5
SAMPLING = 'gumbell'


# Testing parameters
N_FAKE_SAMPLES = 50000
TEST_BATCH_SIZE_BIG = 2*1024
TEST_BATCH_SIZE_SMALL = 512




def print_params(param_list, param_names_list):
    for param_name, param in zip(param_names_list, param_list):
        print(f"{param_name}: {param}")
    print()
    
    
    

# Show input data
print('Input script data', '\n')
print('Main parameters:')
in_param_list = [LOAD_PATH, DEVICE, MODEL_TYPE, DATASET_TYPE, OUT_FEATURES, MODEL_DIR, MODEL_NAME]
in_param__names_list = ['LOAD_PATH', 'DEVICE', 'MODEL_TYPE', 'DATASET_TYPE', 'OUT_FEATURES', 'MODEL_DIR', 'MODEL_NAME']
print_params(in_param_list, in_param__names_list)
print()
print()


print('All model parameters:')
in_param_list = [OUT_FEATURES, NONLINEARITY, IN_FEATURES,  N_BINS, DROPOUT, TEMP, SAMPLING]
in_param__names_list = ['OUT_FEATURES', 'NONLINEARITY', 'IN_FEATURES',  'N_BINS', 'DROPOUT', 'TEMP', 'SAMPLING']
print_params(in_param_list, in_param__names_list)
print()

print('Testing parameters:')
in_param_list = [N_FAKE_SAMPLES, TEST_BATCH_SIZE_BIG, TEST_BATCH_SIZE_SMALL]
in_param__names_list = ['N_FAKE_SAMPLES', 'TEST_BATCH_SIZE_BIG', 'TEST_BATCH_SIZE_SMALL']
print_params(in_param_list, in_param__names_list)
print()

print('Dataset parameters:')
in_param_list = [DATASET_TYPE, TRAIN_SIZE, TEST_SIZE]
in_param__names_list = ['DATASET_TYPE', 'TRAIN_SIZE', 'TEST_SIZE']
print_params(in_param_list, in_param__names_list)
print()


## other parameters
NUM_WORKERS = 10

# Other parameters
print('Other parameters')
other_param_list = [NUM_WORKERS, ]
other_param_names_list = ['NUM_WORKERS']
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






################# Initialization of the model
device = DEVICE
print(DEVICE)


print("Initialization of the model")
print("model_name: ", model_name, '\n\n' )
if MODEL_TYPE == 'LRAE':
    GRID = torch.arange(1,N_BINS+1).to(device)/N_BINS
    model = ConvLRAE(IN_FEATURES, OUT_FEATURES, N_BINS, GRID, dropout=DROPOUT, nonlinearity=NONLINEARITY,
                sampling=SAMPLING, temperature=TEMP, in_channels=DS_IN_CHANNELS).to(device)
elif MODEL_TYPE == 'VAE':
    # print("!!!!!!!!!!!!!!!")
    print(IN_FEATURES, OUT_FEATURES, NONLINEARITY, DS_IN_CHANNELS)
    model = ConvVAE(IN_FEATURES, OUT_FEATURES, nonlinearity=NONLINEARITY, in_channels=DS_IN_CHANNELS).to(device)
    # print("22222222222222222")
elif MODEL_TYPE == 'AE':
    model = ConvAE(IN_FEATURES, OUT_FEATURES, nonlinearity=NONLINEARITY, in_channels=DS_IN_CHANNELS).to(device)
else:
    assert False, f"Error, bad model type, select from: {GOOD_MODEL_TYPE}"
      
print(f"{MODEL_TYPE} was initialized")





## model weights upload
model_path = os.path.join(model_dir, model_name + '.pth')
PATH = load_path

print('Load PATH:', PATH)
out_path_name = os.path.join(OUT_DIR, model_name)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
loss_list_train = checkpoint['loss_list_train']
loss_list_test = checkpoint['loss_list_test']
print("Loaded epoch:", epoch)


##########################################



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
                                    transforms.Resize([64, 64]),
                                    torchvision.transforms.ToTensor(),
                                ]))
    test_ds = torchvision.datasets.CelebA('./files/', split='valid', target_type ='attr', download=True,
                                transform=torchvision.transforms.Compose([
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


# BATCH_SIZE = BATCH_SIZE
# dl = DataLoader(train_ds, batch_size=BATCH_SIZE,     num_workers=num_workers)
# dl_test = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=num_workers)

# #full dataset train
FULL_TRAIN_SIZE = TRAIN_SIZE
dl_full_train = DataLoader(train_ds, batch_size=FULL_TRAIN_SIZE)
for x, y in dl_full_train:
    X_full_train = x
    targets = y
    break

# #full dataset train
FULL_TEST_SIZE = TEST_SIZE
dl_full_test = DataLoader(test_ds, batch_size=FULL_TEST_SIZE)
for x, y in dl_full_test:
    X_full_test = x
    targets_test = y
    break

print("Dataset parameters:")
print(f"TRAIN_SIZE: {TRAIN_SIZE}({ds_train_size})")
print(f"TEST_SIZE: {TEST_SIZE}({df_test_size})")
# for param, param_name in zip([BATCH_SIZE], ["BATCH_SIZE"] ):
#     print(f"{param_name} = {param}")

print(f"{DATASET_TYPE} dataset logs:")
print("Img channel:", ds_in_channels)
# print(X_full_train.shape)
# print(torch.max(X_full_train))
# print(targets.unique(return_counts=True))

print('\n\n')

# ds_in_channels = X_full_train.shape[1]
###################






###################### EVALUATION


##vLoss function
plt.figure(figsize=(6,3))
plt.plot(loss_list_train, alpha=0.5, label='train')
plt.plot(loss_list_test, alpha=0.5, label='test')
# plt.yscale('log')
plt.legend()
plt.savefig(f"{out_path_name}__loss.jpg")
# plt.show()

## Data calculation
with torch.no_grad():
    model.eval()    
    decoded_2d1, encoded_out_dim1, factors_probability1 = get_inf_by_layers(model, X_full_train,
                                                                            batch_size=TEST_BATCH_SIZE_BIG, device=device)    
    decoded_2d2, encoded_out_dim2, factors_probability2 = get_inf_by_layers(model, X_full_test,
                                                                            batch_size=TEST_BATCH_SIZE_BIG, device=device)


## RECONSTRUCTION
### Check reconstruction
plt.figure()
check_reconstruction(model, test_ds, device)
plt.savefig(f"{out_path_name}__rec_test.jpg")
# plt.show()
plt.figure()
check_reconstruction(model, train_ds, device)
plt.savefig(f"{out_path_name}__rec_train.jpg")
# plt.show()

### MSE and PSNR scores
## Calculate reconstruction
print()
# MSE
mse_train = torch.nn.MSELoss()(decoded_2d1.cpu().detach(), X_full_train.cpu().detach())
mse_test = torch.nn.MSELoss()(decoded_2d2.cpu().detach(), X_full_test.cpu().detach())
print(f"MSE: {mse_test.item():.4f}({mse_train.item():.4f})")

# PSNR
psnr_train = 10*torch.log10(1 / (mse_train + 1e-20))
psnr_test = 10*torch.log10(1 / (mse_test + 1e-20))

print(f"PSNR: {psnr_test.item():.2f}({psnr_train.item():.2f})")

print()

### FID reconstruction train and test
print()
# test
imgs_real_r = X_full_test
torch.cuda.empty_cache()
r_fid = ManualFID(device=device)
r_fid.update_full(imgs_real_r, True, batch_size=512, transform=prepare_to_FID)
del imgs_real_r


imgs_fake_r = decoded_2d2
r_fid.clear_part(is_real=False)
r_fid.update_full(imgs_fake_r, False, batch_size=512, transform=prepare_to_FID)
r_fid_value = r_fid.compute()
# print("Test reconstruction fid:", r_fid_value, '\n')
del imgs_fake_r
print(f"Rec FID test: {r_fid_value :.2f}")


del r_fid

#train
# FID reconstruction train and test
# train
imgs_real_r_train = X_full_train
torch.cuda.empty_cache()
r_fid_train = ManualFID(device=device)
r_fid_train.update_full(imgs_real_r_train, True, batch_size=512)
del imgs_real_r_train

imgs_fake_r_train = decoded_2d1
r_fid_train.clear_part(is_real=False)
r_fid_train.update_full(imgs_fake_r_train, False, batch_size=512)
r_fid_value_train = r_fid_train.compute()
# print("Test reconstruction fid:", r_fid_value_train, '\n')
del imgs_fake_r_train

print(f"Rec FID train: {r_fid_value_train :.2f}")
print()
print(f"Rec FID: {r_fid_value:.2f}({r_fid_value_train:.2f})")



## GENERATION 
print("\n\nGeneration test:")
torch.cuda.empty_cache()

# setup distributions
device_fid = device
model = model.to(device)
# Setup generating

dataset_list = []
dataset_names = []
ground_truth = X_full_train.detach().cpu()
# N_samples = ground_truth.shape[0]
N_samples = N_FAKE_SAMPLES
print("N_samples =", N_samples)

# Generating samples Truth
dataset_truth = ground_truth # full dataset
dataset_list +=[dataset_truth]
dataset_names += ['Truth']
print(f"{dataset_names[-1]} samples = ", dataset_list[-1].shape[0])

# Generating samples from autoencoders
model_dataset = gen_gm_dataset(model, encoded_out_dim1, device, n_components=1, total_size=N_samples,
                               batch_size=TEST_BATCH_SIZE_SMALL, C_H_W=C_H_W, max_iter=1000)
dataset_list +=[model_dataset]
dataset_names += [MODEL_TYPE + '_gm1']
print(f"{dataset_names[-1]} samples = ", dataset_list[-1].shape[0])

# Generating samples from autoencoders
model_dataset = gen_gm_dataset(model, encoded_out_dim1, device, n_components=4, total_size=N_samples,
                               batch_size=TEST_BATCH_SIZE_SMALL, C_H_W=C_H_W, max_iter=1000)
dataset_list +=[model_dataset]
dataset_names += [MODEL_TYPE + '_gm4']
print(f"{dataset_names[-1]} samples = ", dataset_list[-1].shape[0])

#### display 
print('\nDisplaying generation')
display_datasets(dataset_list, dataset_names)
plt.savefig(f"{out_path_name}__gen.jpg")


#### FID calculation
print("\n\n")
print("Generation FID  calculations:")
torch.cuda.empty_cache()

try:
    m_fid = r_fid_train
    print("m_fid <--- r_fid_train ")
    m_fid.to(device_fid)
except:
    print("Init m_fid")
    imgs_real = prepare_to_FID(dataset_list[0])
    m_fid = ManualFID(device=device_fid)
    m_fid.update_full(imgs_real, True, batch_size=512)
    del imgs_real
        

imgs_fake = prepare_to_FID(dataset_list[1])
m_fid.clear_part(is_real=False)
m_fid.update_full(imgs_fake, False, batch_size=512)
m_fid_value = m_fid.compute()
print("fake:", m_fid_value.item(), '\n')
del imgs_fake

imgs_fake_gm4 = prepare_to_FID(dataset_list[2])
m_fid.clear_part(is_real=False)
m_fid.update_full(imgs_fake_gm4, False, batch_size=512)
m_fid_value_gm4 = m_fid.compute()
print("fake gm4:", m_fid_value_gm4.item())
del imgs_fake_gm4

print(f"FID gm1: {m_fid_value.item() :.2f} FID gm4: {m_fid_value_gm4.item() :.2f}")
print()


#########################
timer_end = timer()
print("Evaluation was successfully finished!")
print(f"Elapsed time: {timer_end - timer_start:.2f} second") # Time in seconds, e.g. 5.38091952400282

