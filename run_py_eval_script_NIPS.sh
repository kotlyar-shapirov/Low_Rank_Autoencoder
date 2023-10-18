# test_dir_name="test_NIPS"
# eval_dir_name="eval_NIPS"
# GPU="cuda:0"


eval_dir_name="eval_NIPS"
# {
#     nohup python3 run_eval.py -l test_NIPS/data_o/test_o_NIPS__MNIST__VAE__128__0.01__100.pth -o eval_NIPS/data_o -d cuda:4 -a NIPS
#     nohup python3 run_eval.py -l test_NIPS/data_o/test_o_NIPS__MNIST__VAE__128__0.001__100.pth -o eval_NIPS/data_o -d cuda:4 -a NIPS
#     nohup python3 run_eval.py -l test_NIPS/data_o/test_o_NIPS__MNIST__VAE__128__0.0001__100.pth -o eval_NIPS/data_o -d cuda:4 -a NIPS
# } > $eval_dir_name/o_VAE_e100.out 2>&1 &

# {
#     nohup python3 run_eval.py -l test_NIPS/data_o/test_o_NIPS__MNIST__LRAE__128__0.01__100.pth -o eval_NIPS/data_o -d cuda:2 -a NIPS
#     nohup python3 run_eval.py -l test_NIPS/data_o/test_o_NIPS__MNIST__LRAE__128__0.001__100.pth -o eval_NIPS/data_o -d cuda:2 -a NIPS
#     nohup python3 run_eval.py -l test_NIPS/data_o/test_o_NIPS__MNIST__LRAE__128__0.0001__100.pth -o eval_NIPS/data_o -d cuda:2 -a NIPS
#     nohup python3 run_eval.py -l test_NIPS/data_o/test_o_NIPS__MNIST__LRAE__128__0.1__100.pth -o eval_NIPS/data_o -d cuda:2 -a NIPS
# } > $eval_dir_name/o_LRAE_e100.out 2>&1 &


nohup python3 run_eval.py -l test_NIPS/test_b_NIPS_1024__MNIST__VAE__128__0.001__50.pth -o eval_NIPS/data_o -d cuda:0 -a NIPS \
        > $eval_dir_name/o_VAE_b1024.out 2>&1 &
nohup python3 run_eval.py -l test_NIPS/test_b_NIPS_512__MNIST__VAE__128__0.001__50.pth  -o eval_NIPS/data_o -d cuda:1 -a NIPS \
        > $eval_dir_name/o_VAE_b512.out 2>&1 &
nohup python3 run_eval.py -l test_NIPS/test_b_NIPS_256__MNIST__VAE__128__0.001__50.pth  -o eval_NIPS/data_o -d cuda:2 -a NIPS \
        > $eval_dir_name/o_VAE_b256.out 2>&1 &
nohup python3 run_eval.py -l test_NIPS/test_b_NIPS_128__MNIST__VAE__128__0.001__50.pth  -o eval_NIPS/data_o -d cuda:4 -a NIPS \
        > $eval_dir_name/o_VAE_b128.out 2>&1 &



# python3  run_eval.py -l test_NIPS/test1_NIPS__MNIST__AE__128__0.1__50.pth -o eval_NIPS -d cuda:2 -a NIPS

# # {
# # nohup python3  run_eval_NIPS.py -l $test_dir_name/test1_NIPS__MNIST__AE__128__0.1__50.pth -o $eval_dir_name -d $GPU
# # nohup python3  run_eval_NIPS.py -l $test_dir_name/test1_NIPS__MNIST__VAE__128__0.1__50.pth -o $eval_dir_name -d $GPU
# # nohup python3  run_eval_NIPS.py -l $test_dir_name/test1_NIPS__MNIST__LRAE__128__0.1__50.pth -o $eval_dir_name -d $GPU 
# # } > $eval_dir_name/out_1.out 2>&1 &



# nohup python3  run_eval_NIPS.py -l $test_dir_name/test_NIPS__CelebA__VAE__512__0.1__100.pth -o $eval_dir_name -d $GPU > $eval_dir_name/out_1.out 2>&1 &



# nohup python3  run_eval_NIPS.py -l test_NIPS/test_NIPS__CelebA__VAE__512__0.1__100.pth -o eval_NIPS -d cuda:0 > eval_NIPS/out_1.out 2>&1 &

# nohup python3 run_train_NIPS.py -m VAE -d cuda:3  > $test_dir_name/out2_MNIST_VAE.out 2>&1 &
# nohup python3 run_train_NIPS.py -m AE -d cuda:4  > $test_dir_name/out_MNIST_AE.out 2>&1 &
# nohup python3 run_train_NIPS.py -m LRAE -d cuda:2  > $test_dir_name/out_MNIST_LRAE.out 2>&1 &


# python3 run_eval_NIPS.py -l test_NIPS/test1_NIPS__MNIST__AE__128__0.1_100.pth
# python3 run_eval_NIPS.py -l test_NIPS/test1_NIPS__MNIST__AE__128__0.1_100.pth






# nohup python3 run_eval_NIPS.py -l      VAE -d cuda:0  > $eval_dir_name/out_eval_script.out 2>&1


# nohup python3 run_train_NIPS.py -m VAE -d cuda:0  > $test_dir_name/out2_CELEBA_VAE.out 2>&1 &
# # nohup python3 run_train_NIPS.py -m AE -d cuda:4  > $test_dir_name/out_MNIST_AE.out 2>&1 &
# nohup python3 run_train_NIPS.py -m LRAE -d cuda:1  > $test_dir_name/out_CELEBA_LRAE.out 2>&1 &

# nohup python3 run_train.py -m AE -d cuda:0  > $test_3_save/out_CELEBA_0_AE.out 2>&1 & 
# nohup python3 run_train.py -m LRAE -d cuda:1  > test_1_save/out_LRAE.out 2>&1 &


# 732107