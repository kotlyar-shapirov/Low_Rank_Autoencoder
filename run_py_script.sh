test_dir_name="test_3_save"
nohup python3 run_train.py -m VAE -d cuda:2  > $test_dir_name/out_CIFAR10_VAE.out 2>&1 &
nohup python3 run_train.py -m AE -d cuda:4  > $test_dir_name/out_CIFAR10_AE.out 2>&1 

nohup python3 run_train.py -m LRAE -d cuda:2  > $test_dir_name/out_CIFAR10_LRAE.out 2>&1 &

# nohup python3 run_train.py -m AE -d cuda:0  > $test_3_save/out_CELEBA_0_AE.out 2>&1 & 
# nohup python3 run_train.py -m LRAE -d cuda:1  > test_1_save/out_LRAE.out 2>&1 &


# 2855389
# 2858810