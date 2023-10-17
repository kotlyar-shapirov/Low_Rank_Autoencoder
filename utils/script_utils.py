import torchvision
from torchvision import transforms
from torch.utils.data import Subset, Dataset, DataLoader
import torch
import matplotlib.pyplot as plt




def select_dataset(DATASET_TYPE, GOOD_DATASET_TYPE): 
    if DATASET_TYPE.upper() in ['MNIST']:
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
        
        
    elif DATASET_TYPE.upper() in ['CIFAR10']:
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
        
    elif DATASET_TYPE.upper() in ['CELEBA']:
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
        
        
    return train_ds, test_ds, ds_train_size, df_test_size, ds_in_channels
 
 
 
def setup_dataset_training(train_ds, test_ds, BATCH_SIZE, num_workers=10):    
    BATCH_SIZE = BATCH_SIZE
    dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=num_workers)
    dl_test = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=num_workers)
    return dl, dl_test


def setup_dataset_eval(train_ds, test_ds, ds_train_size, df_test_size,  num_workers=10):    

    # #full dataset train
    FULL_TRAIN_SIZE = ds_train_size
    dl_full_train = DataLoader(train_ds, batch_size=FULL_TRAIN_SIZE, num_workers=num_workers)
    for x, y in dl_full_train:
        X_full_train = x
        targets = y
        break

    # #full dataset train
    FULL_TEST_SIZE = df_test_size
    dl_full_test = DataLoader(test_ds, batch_size=FULL_TEST_SIZE, num_workers=num_workers)
    for x, y in dl_full_test:
        X_full_test = x
        targets_test = y
        break
    
    
    return X_full_train, X_full_test, targets, targets_test





def init_model(MODEL_TYPE, GOOD_MODEL_TYPE,  models_class_list, models_params, device):
    
    IN_FEATURES = models_params['IN_FEATURES']
    BOTTLENECK = models_params['BOTTLENECK']
    OUT_FEATURES = models_params['OUT_FEATURES']
    NONLINEARITY = models_params['NONLINEARITY']
    ds_in_channels = models_params['DS_IN_CHANNELS']
    N_BINS = models_params['N_BINS']
    DROPOUT = models_params['DROPOUT']
    SAMPLING = models_params['SAMPLING']
    TEMP = models_params['TEMP']
    
    
    
    ConvAE, ConvVAE, ConvLRAE = models_class_list
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
        
    return model
    
    
    
    
    
    
def update_out_file(update_str, file_path, rewrite=False, end='\n', print_=True):    
    flag = 'a' if not rewrite else 'w'
    with open(file_path, flag) as f:
        f.write(update_str + end)
        
    if print_:
        print('\n', update_str, '\n', sep='')
        
        
        
def plot_loss(loss_list_train, loss_list_test, save_path=None):
    plt.figure(figsize=(6,3))
    plt.plot(loss_list_train, alpha=0.5, label='train')
    plt.plot(loss_list_test, alpha=0.5, label='test')
    # plt.yscale('log')
    plt.legend()
    
    if (save_path is not None) and (save_path != ''):
        plt.savefig(save_path)
    
    
