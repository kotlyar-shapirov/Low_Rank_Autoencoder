
# nohup python3 run_train.py -m IRMAE -a NIPS -D CELEBA -d cuda:1 -A 0.001 -b 32 > test_NIPS/out_CELEBA_IRMAE.out 2>&1  &
# 17878


# test_dir_name="test_NIPS"

{
    nohup python3 run_train.py   -m IRMAE  -a NIPS        -D MNIST    -d cuda:0   -A 0.001  -b 32   
    nohup python3 run_train.py   -m VAE    -a NIPS        -D FMNIST   -d cuda:0   -A 0.001  -b 32
    nohup python3 run_train.py   -m LRAE   -a NIPS        -D FMNIST   -d cuda:0   -A 0.1    -b 32
} > test_NIPS/out_F_MNIST_IRMAE_LRAE_VAE.out 2>&1  &

{
    nohup python3 run_train.py -m IRMAE -a NIPS -D FMNIST -d cuda:3 -A 0.001 -b 32
    nohup python3 run_train.py -m AE -a NIPS -D FMNIST -d cuda:3 -A 0.001 -b 32
} > test_NIPS/out_FMNIST_IRMAE_AE.out 2>&1  &



# -m, -d, -b, -A out_file
# -D 
# -a

# AIRAT SCRIPTS
# nohup python3 run_train_wasser.py -m LRAE -a NIPS -D MNIST -d cuda:0 -A 0.001 -L wasser -b 1024 > wasser_test/bl_MNIST_LRAE_1024.out 2>&1  &

# nohup python3 run_train.py -m VAE -a NIPS -D FMNIST -d cuda:0 -A 0.001 -b 256 > fmnist_test/bl_FMNIST_VAE_256.out 2>&1  &

nohup python3 run_train.py -m LRAE -a NIPS -D FMNIST -d cuda:1 -A 0.1 -b 256 > fmnist_test/bl_FMNIST_LRAE_256.out 2>&1  &

nohup python3 run_train.py -m IRMAE -a NIPS -D FMNIST -d cuda:2 -A 0.001 -b 256 > fmnist_test/bl_FMNIST_IRMAE_256.out 2>&1  &

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