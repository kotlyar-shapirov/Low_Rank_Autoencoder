test_dir_name="test_NIPS"

{
nohup python3 run_train.py -m VAE -a NIPS -D MNIST -d cuda:0 -A 0.001 -b 1024
nohup python3 run_train.py -m VAE -a NIPS -D MNIST -d cuda:0 -A 0.001 -b 128
} > $test_dir_name/o_MNIST_VAE_b1.out 2>&1  &

{
nohup python3 run_train.py -m VAE -a NIPS -D MNIST -d cuda:1 -A 0.001 -b 512
nohup python3 run_train.py -m VAE -a NIPS -D MNIST -d cuda:1 -A 0.001 -b 256
} > $test_dir_name/o_MNIST_VAE_b2.out 2>&1  &



# -m, -d, -b, -A 
# -D 
# -a

test_dir_name="test_NIPS"
nohup python3 run_train.py -m LRAE -a NIPS -D CELEBA -d cuda:0 -A 0.001 -b 1024 > $test_dir_name/bl_CELEBA_VAE_1024.out 2>&1  &
# nohup python3 run_train.py -m VAE -a NIPS -D CELEBA -d cuda:0 -A 0.001 -b 1024 > $test_dir_name/bl_MNIST_VAE_1024.out 2>&1  &



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