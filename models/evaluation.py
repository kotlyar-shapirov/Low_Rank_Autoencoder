from copy import deepcopy

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm
from sklearn.mixture import GaussianMixture




def inf_by_layers(model, x_batch):
    # forward pass with intermediate layers
    x_down = model.down(x_batch)
    B, C, H, W = x_down.shape
    x_flat = x_down.view(B,C*H*W)
    encoded_inter_dim = x_flat
    encoded_out_dim, factors_probability = model.low_rank.low_rank_pants(encoded_inter_dim)
    # decoded_inter_dim =                    model.low_rank.intermediate_decoder(encoded_out_dim)
    decoded_inter_dim = encoded_out_dim
    decoded_1d =                           model.low_rank.decoder(decoded_inter_dim)
    decoded_2d_small = decoded_1d.view(B, C, H, W)
    decoded_2d = model.up(decoded_2d_small)

    return decoded_2d, encoded_out_dim, factors_probability


def get_inf_by_layers(model, X_full, batch_size, device):
    model.eval()    
    idx_list = gen_idx_for_batches(batch_size=batch_size, total_size=X_full.shape[0])
    decoded_2d_list, encoded_out_dim_list, factors_probability_list = [], [], []
    with torch.no_grad():
        for ii in tqdm(range(len(idx_list) -1)):
            X_eval = X_full[idx_list[ii]:idx_list[ii+1]]
            decoded_2d_, encoded_out_dim_, factors_probability_ = inf_by_layers(model, X_eval.to(device))
            decoded_2d_list += [decoded_2d_.detach().cpu()]
            encoded_out_dim_list += [encoded_out_dim_.detach().cpu()]
            
            if factors_probability_ is None:
                factors_probability_list += []
            else:
                factors_probability_list += [factors_probability_.detach().cpu()]
                
        decoded_2d, encoded_out_dim = torch.cat(decoded_2d_list, dim=0), torch.cat(encoded_out_dim_list, dim=0)
        if len(factors_probability_list) > 0:
            factors_probability = torch.cat(factors_probability_list, dim=0)
        else:
            factors_probability = None
    return decoded_2d, encoded_out_dim, factors_probability



def check_reconstruction(model, X_test):
    device = X_test.device
    fig, axs = plt.subplots(3,10, figsize=(11,4))
    for i in range(0,10):
        indx=i+30
        #true
        
        #pred
        with torch.no_grad():
            model.eval()
            x_batch = X_test[indx].unsqueeze(0).to(device)
            
            # plotting original images
            # print(x_batch.shape, x_batch.cpu().detach().permute(0, 2, 3, 1).numpy().shape)
            axs[0,i].imshow(x_batch[0].cpu().detach().permute(1, 2, 0).numpy()) # dataset
            

            # forward pass with intermediate layers
            decoded_2d, encoded_out_dim, factors_probability = inf_by_layers(model, x_batch)
            
        axs[1,i].imshow(decoded_2d[0].cpu().detach().permute(1, 2, 0).numpy()) # output
        mse = torch.nn.MSELoss()(decoded_2d[0].cpu().detach(), x_batch[0].cpu().detach())
        axs[0,i].set_title(f"{mse: .4f}")
        # axs[0,i].set_title(f"{mse*100: .2f}%")
        # axs[1,i].set_title(f"{10*torch.log10(1 / (mse + 1e-20)):.1f}")
        axs[1,i].set_title(f"{10*torch.log10(1 / (mse + 1e-20)):.2f}")


        
        # 1d probabilities
        if factors_probability is not None:
            if len(factors_probability.shape) > 2: 
                for j in range(factors_probability.shape[1]):
                    axs[2,i].plot(factors_probability[0,j,::].cpu().detach().numpy())
                    axs[2,i].set_ylim(-0.1,1.1)
            
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])
        axs[2,i].set_xticks([])
        axs[2,i].set_yticks([])

    return



def gen_idx_for_batches(total_size, batch_size=512, verbose=1):
    B = total_size
    idx_ranges = B // batch_size
    idx_list = [0] + [batch_size*(ii+1) for ii in range(idx_ranges)]
    
    di = 0
    if B % batch_size != 0:
        idx_list += [B]
        di = 1
        
    if verbose:
        print(batch_size, 'x', len(idx_list)-1-di, '+', di*(idx_list[-1] - idx_list[-2]),  f"== {B}({(len(idx_list)-1)*batch_size})")
    return idx_list


def display_datasets(dataset_list, dataset_names=[], W_n=5, H_n = 5):

    n_datasets = len(dataset_list)

    j_title = W_n//2 
    def_name = "-"

    plt.figure(figsize=[1*(W_n+1)*n_datasets, 1*H_n])

    for idx_dataset, dataset_ in enumerate(dataset_list):
        dataset = dataset_[torch.randperm(dataset_.shape[0])]
        
        B_, C_, H_, W_ = dataset.shape
        imgs = dataset.view(B_, C_, -1)
        imgs = imgs - imgs.min(-1, keepdims=True)[0]
        imgs = imgs/imgs.max(-1, keepdims=True)[0]
        dataset = imgs.reshape(B_, C_, H_, W_)
        dataset = dataset.expand(B_, 3, H_, W_)

        
        
        for i in range(H_n):
            for j in range(W_n+1):
                plt.subplot(H_n, (W_n+1)*n_datasets, i*(W_n+1)*n_datasets + j+1 + idx_dataset*(W_n+1))
                # print(i, j, idx_dataset, i*(W_n+1)*n_datasets + j+1 + idx_dataset*(W_n+1))
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                
                if j == W_n:
                    continue

                plt.imshow(dataset[i*H_n + j].permute(1, 2, 0))
                
                # plt.box(False)
                
                if j == j_title and i == 0:
                    plt.title(dataset_names[idx_dataset] if len(dataset_names) > idx_dataset else def_name)
                    
    
                
def gen_gm_dataset(model, encoded, device, n_components=1, total_size=50000, batch_size=1024, C_H_W = [256, 2, 2], max_iter=500):
    C, H, W = C_H_W
    global gen_idx_for_batches
    # device = encoded.device

    gm = GaussianMixture(n_components=n_components, max_iter=max_iter)
    gm.fit(encoded.detach().cpu().numpy())

    idx_list = gen_idx_for_batches(total_size, batch_size, verbose=True)

    batch_list = []
    for ii in tqdm(range(len(idx_list)-1)):
        B = idx_list[ii+1] - idx_list[ii]
        # print(B)
            
        rand = gm.sample(B)
        
        rand = torch.as_tensor(rand[0]).to(torch.float32).to(device)
        
        decoded_1d = model.low_rank.decoder(rand.to(device))
        decoded_2d_small = decoded_1d.view(B, C, H, W)
        decoded_2d = model.up(decoded_2d_small)
        batch_dataset = decoded_2d.detach().cpu()
        batch_list += [batch_dataset]
        
    model_dataset = torch.cat(batch_list, dim=0)
    return model_dataset    


def update_FID_class(fid, prepared_img, is_real, batch_size=512, update_list=None):
    device = fid.device
    total_size = prepared_img.shape[0]
    idx_list = gen_idx_for_batches(total_size, batch_size, verbose=True)
    for ii in tqdm(range(len(idx_list)-1)):
        torch.cuda.empty_cache()
        batch_img = prepared_img[idx_list[ii]:idx_list[ii+1]]
        fid.update(batch_img.to(device), is_real)
        if update_list is not None:
            update_list += [fid.compute().detach().cpu()]
        
    return fid



class ManualFID():
    def __init__(self, device='cpu'):
        inception_v3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception_v3.eval()
        
        self.device = device
        
        for param in inception_v3.parameters():
            param.request_grad = False
            
        self.inception_v3 = inception_v3
        
        
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = deepcopy(output.detach())
                # activation[name+ "_in"] = input
                # activation[name+ "_model"] = model
            return hook

        self.inception_v3.avgpool.register_forward_hook(get_activation('avgpool'))
        
        self.inception_v3.to(self.device)
        
        self.pool_tensor_real = None
        self.pool_tensor_fake = None
        
        
    def clear(self,):
        self.pool_tensor_real = None
        self.pool_tensor_fake = None
        
    def clear_part(self, is_real):
        if is_real is True:
            self.pool_tensor_real =  None
            print('Real was cleared!') 
        else:
            self.pool_tensor_fake =  None
            print('Fake was cleared!')

        
        
    def update_full(self, prepared_img_full, is_real, batch_size=512):
        device = self.device
        global gen_idx_for_batches
        idx_list = gen_idx_for_batches(prepared_img_full.shape[0], batch_size, verbose=True)
        
        
        images_full = prepared_img_full
        x_full = images_full
        
        pool_list = []
        for idx_start, ids_end in tqdm(zip(idx_list[-2::-1], idx_list[-1:0:-1]), total=len(idx_list)-1):
            with torch.no_grad():
                self.inception_v3.to(device)
                self.inception_v3(x_full[idx_start:ids_end].to(device))
                pool = self.activation['avgpool'].detach().squeeze()
                pool_list += [pool]
        pool_tensor= torch.cat(pool_list, dim=0)
        
        if is_real:
            self.pool_tensor_real =  pool_tensor
            print('Real is done!') 
        else:
            self.pool_tensor_fake =  pool_tensor
            print('Fake is done!')
            
     
            
    # def _matrix_sqrt_real(self, matrix):
    #         L, V= torch.linalg.eig(matrix)
            
    #         matrix_sqrt = V @ torch.diag(torch.sqrt(L)) @ torch.inverse(V)
    #         return matrix_sqrt.real
        
    def _trace_of_matsqrt(self, matrix):
        # L, V = torch.linalg.eig(matrix)
        L = torch.linalg.eigvals(matrix)
        matrix_diag_sqrt = torch.sqrt(L).real
        return matrix_diag_sqrt.sum()
        
            
            
    def compute_old(self, ):
        # matrix_sqrt = self._matrix_sqrt_real
        # def matrix_sqrt(matrix):
        #     U, S, Vh = torch.linalg.svd(matrix)
        #     matrix_sqrt = U @ torch.diag(torch.sqrt(S)) @ Vh
        #     return matrix_sqrt
        
        matrix_sqrt = self._matrix_sqrt_real
        
        if self.pool_tensor_real is None or self.pool_tensor_fake is None:
            print("real and fake should be updated!")
            return 0
        
        pool_tensor_real = self.pool_tensor_real
        pool_tensor_fake = self.pool_tensor_fake
        
        
        mu = pool_tensor_real.mean(0)
        sigma =  torch.cov(pool_tensor_real.T)
        sigma_sqrt = matrix_sqrt(sigma)

        mu_w = pool_tensor_fake.mean(0)
        sigma_w =  torch.cov(pool_tensor_fake.T)

        fid_per_dim = torch.norm(mu - mu_w)**2 + torch.trace(sigma) + torch.trace(sigma_w) \
                    - 2*torch.trace(matrix_sqrt(sigma_sqrt @ sigma_w @ sigma_sqrt))
        fid_manual = fid_per_dim.sum()
        
        return fid_manual.detach().cpu()
    
    
    def compute(self, ):
        # matrix_sqrt = self._matrix_sqrt_real
        # def matrix_sqrt(matrix):
        #     U, S, Vh = torch.linalg.svd(matrix)
        #     L, V= torch.linalg.eig(matrix)
            
        #     matrix_sqrt = V @ torch.diag(torch.sqrt(L)) @ V.T
        #     return matrix_sqrt
        
        
        # def matrix_sqrt(matrix):
        #     U, S, Vh = torch.linalg.svd(matrix)
        #     matrix_sqrt = U @ torch.diag(torch.sqrt(S)) @ Vh
        #     return matrix_sqrt
        
        if self.pool_tensor_real is None or self.pool_tensor_fake is None:
            print("real and fake should be updated!")
            return 0
        
        pool_tensor_real = self.pool_tensor_real
        pool_tensor_fake = self.pool_tensor_fake
        
        
        mu = pool_tensor_real.mean(0)
        # self.mu = mu
        sigma =  torch.cov(pool_tensor_real.T)
        # self.sigma = sigma
        # sigma_sqrt = matrix_sqrt(sigma)

        mu_w = pool_tensor_fake.mean(0)
        # self.mu_w = mu_w
        sigma_w =  torch.cov(pool_tensor_fake.T)
        # self.sigma_w = sigma_w

        fid_manual = torch.sum((mu - mu_w)**2) + sigma.trace() + sigma_w.trace() - 2*self._trace_of_matsqrt(sigma @ sigma_w)
        # fid_manual += -2*torch.trace(matrix_sqrt(sigma @ sigma_w))
        # fid_manual += -2*self._trace_of_matsqrt(sigma @ sigma_w)
        
        # print(fid_per_dim.shape)
        # fid_manual = fid_per_dim.sum()
        
        return fid_manual.detach().cpu()
    
    
    
def prepare_to_FID(dataset):

    B_, C_, H_, W_ = dataset.shape
    

    preprocess = transforms.Compose([transforms.Resize(299)
                                    #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ,])

    imgs = dataset
    imgs = imgs.view(B_, C_, -1)
    imgs = imgs - imgs.min(-1, keepdims=True)[0]
    imgs = imgs/imgs.max(-1, keepdims=True)[0]
    imgs = imgs.reshape(B_, C_, H_, W_)

    C_ = 3 
    imgs = preprocess(imgs.expand(B_, C_, H_, W_))

    return imgs
            
     
   
        
        

