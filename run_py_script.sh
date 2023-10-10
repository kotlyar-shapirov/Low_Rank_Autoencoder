test_dir_name="test_1_save"
nohup python3 run_train.py -m AE -d cuda:1  > $test_dir_name/out_AE.out 2>&1 &
nohup python3 run_train.py -m VAE -d cuda:0  > $test_dir_name/out_VAE.out 2>&1 &
# nohup python3 run_train.py -m LRAE -d cuda:1  > test_1_save/out_LRAE.out 2>&1 &


# 2125423