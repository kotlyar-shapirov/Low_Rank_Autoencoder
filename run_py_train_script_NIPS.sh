
# nohup python3 run_train.py -m IRMAE -a NIPS -D CELEBA -d cuda:1 -A 0.001 -b 32 > test_NIPS/out_CELEBA_IRMAE.out 2>&1  &
# 17878




# nohup python3 run_train.py -m VAE -a NIPS -D CELEBA -d cuda:1 -A 0.001 -b 32 > test_NIPS/out_CELEBA_VAE1.out 2>&1  &


# test_dir_name="test_NIPS"
# for N in  8 16 32 64 128 2 4 256 512 1024



{
# for n_bins in  10 20 30 40 50 
#     do
#         nohup python3 run_train.py  -m LRAE    -D MNIST    -d cuda:0   -A 0.1  -b 256  -n 128  --n_bins $n_bins
#     done

# for n_bins in  10 20 30 40 50 
#     do
#         nohup python3 run_train.py  -m LRAE    -D MNIST    -d cuda:0   -A 0.1  -b 256  -n 16 --n_bins $n_bins
#     done

for n_bins in  10 20 30 40 50 
    do
        nohup python3 run_eval.py -l test_NIPS/data_lrae/test_NIPS_0.5_${n_bins}__MNIST__LRAE__128__0.1__100.pth \
                        -o eval_NIPS/data_lrae/ -d cuda:0 -a NIPS   --n_bins $n_bins\
                        --out_file test_NIPS_0.5_B__MNIST__LRAE__128__0.1__100__metrics.txt 
    done

for n_bins in  10 20 30 40 50 
    do
        nohup python3 run_eval.py -l test_NIPS/data_lrae/test_NIPS_0.5_${n_bins}__MNIST__LRAE__16__0.1__100.pth \
                        -o eval_NIPS/data_lrae/ -d cuda:0 -a NIPS   --n_bins $n_bins\
                        --out_file test_NIPS_0.5_B__MNIST__LRAE__16__0.1__100__metrics.txt 
    done


} > eval_NIPS/lrae_MNIST_LRAE_g0.5_B_n128_16.out 2>&1  &


{
# for n_bins in  10 20 30 40 50 
#     do
#         nohup python3 run_train.py  -m LRAE    -D MNIST    -d cuda:1   -A 0.1  -b 256  -n 8  --n_bins $n_bins
#         nohup python3 run_train.py  -m LRAE    -D MNIST    -d cuda:1   -A 0.1  -b 256  -n 32 --n_bins $n_bins
#     done


for n_bins in  10 20 30 40 50 
    do
        nohup python3 run_eval.py -l test_NIPS/data_lrae/test_NIPS_0.5_${n_bins}__MNIST__LRAE__8__0.1__100.pth \
                        -o eval_NIPS/data_lrae/ -d cuda:1 -a NIPS   --n_bins $n_bins\
                        --out_file test_NIPS_0.5_B__MNIST__LRAE__8__0.1__100__metrics.txt
        nohup python3 run_eval.py -l test_NIPS/data_lrae/test_NIPS_0.5_${n_bins}__MNIST__LRAE__32__0.1__100.pth \
                        -o eval_NIPS/data_lrae/ -d cuda:1 -a NIPS   --n_bins $n_bins\
                        --out_file test_NIPS_0.5_B__MNIST__LRAE__32__0.1__100__metrics.txt
    done
} > eval_NIPS/lrae_MNIST_LRAE_g0.5_B_n8_32.out 2>&1  &



{
# for gumbel_temp in  0.5 0.1 0.05 0.01 1.5 1  
#     do
#         nohup python3 run_train.py  -m LRAE    -D MNIST    -d cuda:2   -A 0.1  -b 256  -n 128  --n_bins 20 --gumbel_temp $gumbel_temp
#         nohup python3 run_train.py  -m LRAE    -D MNIST    -d cuda:2   -A 0.1  -b 256  -n 16 --n_bins 20 --gumbel_temp $gumbel_temp
#     done


for gumbel_temp in  0.5 0.1 0.05 0.01 1.5 1 
    do
        nohup python3 run_eval.py -l test_NIPS/data_lrae/test_NIPS_${gumbel_temp}_20__MNIST__LRAE__128__0.1__100.pth \
                        -o eval_NIPS/data_lrae/ -d cuda:2 -a NIPS   --gumbel_temp $gumbel_temp \
                        --out_file test_NIPS_G_20__MNIST__LRAE__128__0.1__100__metrics.txt
        nohup python3 run_eval.py -l test_NIPS/data_lrae/test_NIPS_${gumbel_temp}_20__MNIST__LRAE__16__0.1__100.pth \
                        -o eval_NIPS/data_lrae/ -d cuda:2 -a NIPS   --gumbel_temp $gumbel_temp \
                        --out_file test_NIPS_G_20__MNIST__LRAE__16__0.1__100__metrics.txt
    done
} > eval_NIPS/lrae_MNIST_LRAE_G_b20_n128_16.out 2>&1  &





# {

# } > test_NIPS/lrae_MNIST_LRAE_nbins_N16.out 2>&1  &


# {
# for n_bins in  5 10 15 20 25 30 35 40 45 50 
#     do
#         nohup python3 run_train.py  -m LRAE    -D MNIST    -d cuda:0   -A 0.1  -b 256  -n 8 --n_bins $n_bins
#     done
# } > test_NIPS/lrae_MNIST_LRAE_nbins_N8.out 2>&1  &

# {
# for n_bins in  5 10 15 20 25 30 35 40 45 50 
#     do
#         nohup python3 run_train.py  -m LRAE    -D MNIST    -d cuda:0   -A 0.1  -b 256  -n 32 --n_bins $n_bins
#     done
# } > test_NIPS/lrae_MNIST_LRAE_nbins_N32.out 2>&1  &








# {
# for N in  128 2 4 256 512 1024
#     do
#         nohup python3 run_train.py  -m LRAE    -D FMNIST    -d cuda:3   -A 0.1  -b 256  -n $N 
#     done
# } > test_NIPS/n_FMNIST_LRAE_N1.out 2>&1  &


# {
# for N in 8 16 32 64 128 2 4 256 512 1024
#     do
#         nohup python3 run_train.py  -m AE    -D FMNIST    -d cuda:0   -A 0  -b 256  -n $N 
#     done
# } > test_NIPS/n_FMNIST_AE_N.out 2>&1  &


# {
# for N in 8 16 32 64 128 2 4 256 512 1024
#     do
#         nohup python3 run_train.py  -m VAE    -D FMNIST    -d cuda:4   -A 0.001  -b 256  -n $N 
#     done
# } > test_NIPS/n_FMNIST_VAE_N.out 2>&1  &


# {
# for N in 8 16 32 64 128 2 4 256 512 1024
#     do
#         nohup python3 run_train.py  -m IRMAE    -D FMNIST    -d cuda:0   -A 0  -b 256  -n $N 
#     done
# } > test_NIPS/n_FMNIST_IRMAE_N.out 2>&1  &


# {
#     nohup python3 run_train.py   -m IRMAE  -a NIPS        -D MNIST    -d cuda:0   -A 0.001  -b 32   
#     nohup python3 run_train.py   -m VAE    -a NIPS        -D FMNIST   -d cuda:0   -A 0.001  -b 32
#     nohup python3 run_train.py   -m LRAE   -a NIPS        -D FMNIST   -d cuda:0   -A 0.1    -b 32
# } > test_NIPS/out_F_MNIST_IRMAE_LRAE_VAE.out 2>&1  &

# {
#     nohup python3 run_train.py -m IRMAE -a NIPS -D FMNIST -d cuda:3 -A 0.001 -b 32
#     nohup python3 run_train.py -m AE -a NIPS -D FMNIST -d cuda:3 -A 0.001 -b 32
# } > test_NIPS/out_FMNIST_IRMAE_AE.out 2>&1  &



# -m, -d, -b, -A out_file
# -D, -n, --n_bins
# -a

# test_dir_name="test_NIPS"
# nohup python3 run_train.py -m LRAE -a NIPS -D CELEBA -d cuda:0 -A 0.001 -b 1024 > $test_dir_name/bl_CELEBA_VAE_1024.out 2>&1  &
# # nohup python3 run_train.py -m VAE -a NIPS -D CELEBA -d cuda:0 -A 0.001 -b 1024 > $test_dir_name/bl_MNIST_VAE_1024.out 2>&1  &



# {
# nohup python3 run_train.py -m LRAE -a NIPS -D MNIST -d cuda:1 -A 0.5
# nohup python3 run_train.py -m LRAE -a NIPS -D MNIST -d cuda:1 -A 1
# # nohup python3 run_train.py -m LRAE -a NIPS -D MNIST -d cuda:1 -A 0.0001
# # nohup python3 run_train.py -m LRAE -a NIPS -D MNIST -d cuda:1 -A 0.1  
# } > $test_dir_name/o_MNIST_LRAE.out 2>&1  &



#nohup python3 run_train.py -m LRAE -a NIPS -D MNIST -d cuda:1 -A 0.5 > $test_dir_name/o_MNIST_LRAE_0.5.out 2>&1  &
# nohup python3 run_train.py -m LRAE -a NIPS -D MNIST -d cuda:0 -A 1 > $test_dir_name/o_MNIST_LRAE_1.out 2>&1  &



# nohup python3 run_train.py -m LRAE -a NIPS -D MNIST -d cuda:1 -A 0.5 > $test_dir_name/o_MNIST_LRAE_0.5.out 2>&1  &
# nohup python3 run_train.py -m LRAE -a NIPS -D MNIST -d cuda:0 -A 1 > $test_dir_name/o_MNIST_LRAE_1.out 2>&1  &
# 4101724, 4106058

# nohup python3 run_train_NIPS.py -m VAE -d cuda:3  > $test_dir_name/out2_MNIST_VAE.out 2>&1 &
# nohup python3 run_train_NIPS.py -m AE -d cuda:4  > $test_dir_name/out_MNIST_AE.out 2>&1 &
# nohup python3 run_train_NIPS.py -m LRAE -d cuda:2  > $test_dir_name/out_MNIST_LRAE.out 2>&1 &

# nohup python3 run_train_NIPS.py -m VAE -d cuda:0  > $test_dir_name/out2_CELEBA_VAE.out 2>&1 &
# # nohup python3 run_train_NIPS.py -m AE -d cuda:4  > $test_dir_name/out_MNIST_AE.out 2>&1 &
# nohup python3 run_train_NIPS.py -m LRAE -d cuda:1  > $test_dir_name/out_CELEBA_LRAE.out 2>&1 &

# # nohup python3 run_train.py -m AE -d cuda:0  > $test_3_save/out_CELEBA_0_AE.out 2>&1 & 
# # nohup python3 run_train.py -m LRAE -d cuda:1  > test_1_save/out_LRAE.out 2>&1 &


# # 732107


# python3 run_train_NIPS.py -m LRAE -a NIPS -D MNIST -d cuda:0