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
from models.evaluation import get_MSE_PSNR_score, calculate_FID

# from torchmetrics.image.fid import FrechetInceptionDistance as tm_FrechetInceptionDistance
# from torcheval.metrics import FrechetInceptionDistance

# from sklearn.mixture import GaussianMixture

from timeit import default_timer as timer

from utils.script_utils import setup_dataset_eval, update_out_file, plot_loss, select_dataset, init_model
from main_utils import get_models_class_list

timer_start = timer() # start timer

#default
GOOD_DATASET_TYPE = ['MNIST', 'CIFAR10', 'CELEBA']
GOOD_MODEL_TYPE = ['VAE', 'AE', 'LRAE']
GOOD_ARCHITECTURE_TYPE = ['V1', 'NISP']




### argparser!
parser = argparse.ArgumentParser(description='Run AE-type models on training')

parser.add_argument('-l', '--load_path', type=str, help='Load model checkpoint path [.pth]')
parser.add_argument('-o', '--out', type=str, default='evaluation', help='Output dir. If -1, out dir is a dir from load_path')

parser.add_argument('-d', '--device', type=str, default='cuda:1', help='torch device name. E.g.: cpu, cuda:0, cuda:1')
parser.add_argument('-a', '--architecture', type=str, default='NIPS',  
                    help=f'model architecture type: {GOOD_ARCHITECTURE_TYPE}')
args = parser.parse_args()



###### Setup script


## Main work parameters
DEVICE = args.device
# MODEL_TYPE = args.model
LOAD_PATH = args.load_path
OUT_DIR = args.out if str(-1) != args.out else os.path.dirname(LOAD_PATH)

ARCHITECTURE_TYPE = args.architecture





load_path = LOAD_PATH
# model_name = os.path.basename(load_path)
model_name = os.path.splitext(os.path.basename(load_path))[0]
model_dir = os.path.dirname(load_path)
MODEL_DIR = model_dir
MODEL_NAME = model_name

model_name_in_list = model_name.split('__')
DATASET_TYPE = model_name_in_list[1]
MODEL_TYPE = model_name_in_list[2]
OUT_FEATURES = int(model_name_in_list[3])
ALPHA = float(model_name_in_list[4])
EPOCHS = model_name_in_list[5]

print(f"Eval {load_path} was started!")



models_class_list = get_models_class_list(DATASET_TYPE, ARCHITECTURE_TYPE) 

# Upload model
if DATASET_TYPE in ['MNIST']:

    IN_FEATURES = 256*2*2
    BOTTLENECK = 128
    C_H_W = [128, 8, 8]
    OUT_FEATURES = C_H_W[0]*C_H_W[1]*C_H_W[2]
    DS_IN_CHANNELS = 1
    N_GM_COMPONENTS = 4

elif DATASET_TYPE in ['CIFAR10']:
    IN_FEATURES = 1024*2*2
    C_H_W = [1024, 8, 8]
    BOTTLENECK = 512
    OUT_FEATURES = C_H_W[0]*C_H_W[1]*C_H_W[2]
    DS_IN_CHANNELS = 3
    N_GM_COMPONENTS = 10

    
elif DATASET_TYPE in ['CelebA', 'CELEBA']:
    IN_FEATURES = 1024*4*4
    C_H_W = [1024, 8, 8]
    BOTTLENECK = 512
    OUT_FEATURES = C_H_W[0]*C_H_W[1]*C_H_W[2]
    DS_IN_CHANNELS = 3
    N_GM_COMPONENTS = 10
else:
   print("Warning! the default run setups was not setuped!")
   
    
NONLINEARITY = nn.ReLU()

# TRAIN_SIZE, TEST_SIZE = -1, -1


# LRAE parameters
N_BINS = 20
DROPOUT = 0.0
TEMP = 0.5
SAMPLING = 'gumbell'


# Testing parameters
N_FAKE_SAMPLES = 50000
TRAIN_SIZE, TEST_SIZE = -1, -1

# N_FAKE_SAMPLES = 1000
# TRAIN_SIZE, TEST_SIZE = 1000, 1000



models_params = {'IN_FEATURES': IN_FEATURES, 'BOTTLENECK': BOTTLENECK, 'OUT_FEATURES': OUT_FEATURES,
                 'NONLINEARITY': NONLINEARITY, 'DS_IN_CHANNELS': DS_IN_CHANNELS,
                 'N_BINS': N_BINS, 'DROPOUT':DROPOUT, 'SAMPLING':SAMPLING, 'TEMP': TEMP}




TEST_BATCH_SIZE_BIG = 512
TEST_BATCH_SIZE_SMALL = 128
if DEVICE in ['cuda:2', 'cuda:3', 'cuda:4']:
    TEST_BATCH_SIZE_BIG = 512
    TEST_BATCH_SIZE_SMALL = 128

if DEVICE in ['cuda:0', 'cuda:1']:
    TEST_BATCH_SIZE_BIG = 1024
    TEST_BATCH_SIZE_SMALL = 512




def print_params(param_list, param_names_list):
    for param_name, param in zip(param_names_list, param_list):
        print(f"{param_name}: {param}")
    print()
    
    
    

# Show input data
print('Input script data', '\n')
print('Main parameters:')


in_param_list = [LOAD_PATH, DEVICE, MODEL_TYPE, DATASET_TYPE, ARCHITECTURE_TYPE, BOTTLENECK, MODEL_DIR, MODEL_NAME]
in_param__names_list = ['LOAD_PATH', 'DEVICE', 'MODEL_TYPE', 'DATASET_TYPE', 'ARCHITECTURE_TYPE', 'BOTTLENECK', 'MODEL_DIR', 'MODEL_NAME']
print_params(in_param_list, in_param__names_list)
print()
print()


print('All model parameters:')
# in_param_list = [BOTTLENECK, OUT_FEATURES, NONLINEARITY, IN_FEATURES,  N_BINS, DROPOUT, TEMP, SAMPLING]
# in_param__names_list = ['BOTTLENECK', 'OUT_FEATURES', 'NONLINEARITY', 'IN_FEATURES',  'N_BINS', 'DROPOUT', 'TEMP', 'SAMPLING']
print_params(in_param_list, in_param__names_list)
print_params(models_params.values(), models_params.keys())
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

model = init_model(MODEL_TYPE, GOOD_MODEL_TYPE,  models_class_list, models_params, device)

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


train_ds, test_ds, ds_train_size, df_test_size, ds_in_channels = select_dataset(DATASET_TYPE, GOOD_DATASET_TYPE)

# dataset and dataloader
TRAIN_SIZE = ds_train_size if TRAIN_SIZE == -1 else TRAIN_SIZE
TEST_SIZE = df_test_size if TEST_SIZE == -1 else TEST_SIZE

X_full_train, X_full_test, targets, targets_test = setup_dataset_eval(train_ds, test_ds, TRAIN_SIZE,
                                                                      TEST_SIZE,  num_workers=NUM_WORKERS)  


print("Dataset parameters:")
print(f"TRAIN_SIZE: {TRAIN_SIZE}({ds_train_size})")
print(f"TEST_SIZE: {TEST_SIZE}({df_test_size})")
# for param, param_name in zip([BATCH_SIZE], ["BATCH_SIZE"] ):
#     print(f"{param_name} = {param}")

print(f"{DATASET_TYPE} dataset logs:")
print("Img channel:", ds_in_channels)
print(X_full_train.shape)
print(torch.max(X_full_train))
print(targets.unique(return_counts=True))

print('\n\n')
###################





##### EVALUATION
print('\n\nEvaluation:')
out_file_path = f"{out_path_name}__metrics.txt"
update_out_file(f'Out: {load_path}', out_file_path, rewrite=True)



## Loss function
save_path_str = f"{out_path_name}__loss.jpg"

plot_loss(loss_list_train, loss_list_test, save_path=save_path_str)
   
plt.close()   
print("Figure was saved:", save_path_str)
##########



## Data calculation
with torch.no_grad():
    model.eval()    
    decoded_2d1, encoded_out_dim1, factors_probability1 = get_inf_by_layers(model, X_full_train,
                                                                            batch_size=TEST_BATCH_SIZE_BIG, device=device, C_H_W=C_H_W)    
    decoded_2d2, encoded_out_dim2, factors_probability2 = get_inf_by_layers(model, X_full_test,
                                                                            batch_size=TEST_BATCH_SIZE_BIG, device=device, C_H_W=C_H_W)

############

## RECONSTRUCTION


### Check reconstruction
# test
plt.figure()
save_path_str = f"{out_path_name}__rec_test.jpg"
check_reconstruction(model, test_ds, device, C_H_W=C_H_W)

plt.savefig(save_path_str), print("Figure was saved:", save_path_str), plt.close()
# plt.savefig(save_path_str)
# print("Figure was saved:", save_path_str)
# plt.close()
# train
plt.figure()
check_reconstruction(model, train_ds, device, C_H_W=C_H_W)
plt.savefig(f"{out_path_name}__rec_train.jpg")
print("Figure was saved:", f"{out_path_name}__rec_train.jpg")
plt.close()
#################


### MSE and PSNR scores
## Calculate reconstruction
# MSE and PSNR
mse_test, psnr_test = get_MSE_PSNR_score(decoded_2d2, X_full_test)
mse_train, psnr_train = get_MSE_PSNR_score(decoded_2d1, X_full_train)

score_str = f"MSE: {mse_test:.4f} ({mse_train:.4f});   " + f"PSNR: {psnr_test:.2f} ({psnr_train:.2f});   "
update_out_file(score_str, out_file_path, print_=True)
#########################

   
### FID reconstruction train and test
print()
# test
r_fid_value = calculate_FID(X_full_test, decoded_2d2, TEST_BATCH_SIZE_SMALL, device, fid_class=None, transform=prepare_to_FID)
print(f"Rec FID test: {r_fid_value :.2f}")

#train
r_fid_train = ManualFID(device=device)
r_fid_value_train = calculate_FID(X_full_train, decoded_2d1, TEST_BATCH_SIZE_SMALL, device, fid_class=r_fid_train, transform=prepare_to_FID)
print(f"Rec FID train: {r_fid_value_train :.2f}")

score_str = f"Rec FID: {r_fid_value:.2f} ({r_fid_value_train:.2f})"
update_out_file(score_str, out_file_path, print_=True)

######################


## GENERATION 
print("\n\nGeneration test:")
torch.cuda.empty_cache()

# setup distributions
device_fid = DEVICE
model = model.to(device)
# Setup generating

dataset_list = []
dataset_names = []
ground_truth = X_full_train.detach().cpu()
# N_samples = ground_truth.shape[0]
N_samples = N_FAKE_SAMPLES
print("N_samples =", N_samples)


###########################
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
model_dataset = gen_gm_dataset(model, encoded_out_dim1, device, n_components=N_GM_COMPONENTS, total_size=N_samples,
                               batch_size=TEST_BATCH_SIZE_SMALL, C_H_W=C_H_W, max_iter=1000)
dataset_list +=[model_dataset]
dataset_names += [MODEL_TYPE + f'_gm{N_GM_COMPONENTS}']
print(f"{dataset_names[-1]} samples = ", dataset_list[-1].shape[0])
###################


#### display 
print('\nDisplaying generation')
display_datasets(dataset_list, dataset_names)
save_path_str = f"{out_path_name}__gen.jpg"
plt.savefig(save_path_str), print("Figure was saved:", save_path_str), plt.close()
#################



#### FID calculation
print("\n\n")
print("Generation FID  calculations:")
torch.cuda.empty_cache()

try:
    m_fid = r_fid_train
    print("m_fid <--- r_fid_train ")
    m_fid.to(device_fid)
    imgs_real = None
except:
    print("Init m_fid")
    m_fid = ManualFID(device=device_fid)
    imgs_real = dataset_list[0]
     
# GM1
imgs_fake = dataset_list[1]
m_fid_value = calculate_FID(imgs_real, imgs_fake, TEST_BATCH_SIZE_SMALL, device_fid, fid_class=m_fid, transform=prepare_to_FID)
print("fake:", m_fid_value, '\n')

#GM{N_GM_COMPONENTS}
imgs_fake = dataset_list[2]
m_fid_value_gm = calculate_FID(None, imgs_fake, TEST_BATCH_SIZE_SMALL, device_fid, fid_class=m_fid, transform=prepare_to_FID)
print(f"fake gm{N_GM_COMPONENTS}:", m_fid_value_gm, '\n')


score_str = f"FID gm1: {m_fid_value :.2f};   FID gm{N_GM_COMPONENTS}: {m_fid_value_gm :.2f}"
# score_str = f"FID gm1: {m_fid_value :.2f} \nFID gm{N_GM_COMPONENTS}: {m_fid_value_gm :.2f}"
update_out_file(score_str, out_file_path, print_=True)

#########################



timer_end = timer()
print(f"Elapsed time: {timer_end - timer_start:.2f} second") # Time in seconds, e.g. 5.38091952400282
print(f"Evaluation {load_path} was successfully finished!")
print('\n\n\n\n\n')

