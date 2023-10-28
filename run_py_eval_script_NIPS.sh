# test_dir_name="test_NIPS"
# eval_dir_name="eval_NIPS"
# GPU="cuda:0"


# eval_dir_name="eval_NIPS"
# test_bl_NIPS_1024_0.0001__MNIST__VAE__128__0.001__50.pth



# -l -d  out_file_name - to change
# -o - change in different runs
# - a - constant 



# nohup python3 run_eval.py -l test_NIPS/test_NIPS__CELEBA__IRMAE__512__0.001__100.pth \
#                         -o eval_NIPS/ -d cuda:1 -a NIPS > eval_NIPS/out_CELEBA_IRMAE.out  2>&1 &



# nohup python3 run_eval.py -l test_NIPS/test1_NIPS__CELEBA__VAE__512__0.1__100.pth \
#                         -o eval_NIPS    -d cuda:1   -a NIPS > eval_NIPS/out_CELEBA_VAE.out 2>&1  &



{
for N in  2 4 8 16 32 64 128 256 512 1024
    do
        nohup python3 run_eval.py -l test_NIPS/data_n/test_NIPS__FMNIST__LRAE__${N}__0.1__50.pth \
                        -o eval_NIPS/data_n/ -d cuda:0 -a NIPS  --out_file test_NIPS__FMNIST__LRAE__N__0.1__50__metrics.txt
    done
} > eval_NIPS/n_FMNIST_LRAE_N_50.out 2>&1  &


{
for N in 2 4 8 16 32 64 128 256 512 1024
    do
        nohup python3 run_eval.py -l test_NIPS/data_n/test_NIPS__FMNIST__AE__${N}__0.0__50.pth \
                        -o eval_NIPS/data_n/ -d cuda:3 -a NIPS --out_file test_NIPS__FMNIST__AE__N__0.0__50__metrics.txt 
    done
} > eval_NIPS/n_FMNIST_AE_N_50.out 2>&1  &


{
for N in 2 4 8 16 32 64 128 256 512 1024
    do
        nohup python3 run_eval.py -l test_NIPS/data_n/test_NIPS__FMNIST__VAE__${N}__0.001__50.pth \
                        -o eval_NIPS/data_n/ -d cuda:1 -a NIPS --out_file test_NIPS__FMNIST__VAE__N__0.001__50__metrics.txt
    done
} > eval_NIPS/n_FMNIST_VAE_N_50.out 2>&1  &


{
for N in  2 4 8 16 32 64 128 256 512 1024
    do
        nohup python3 run_eval.py -l test_NIPS/data_n/test_NIPS__FMNIST__IRMAE__${N}__0.0__50.pth \
                        -o eval_NIPS/data_n/ -d cuda:2 -a NIPS --out_file test_NIPS__FMNIST__IRMAE__N__0.0__50__metrics.txt
    done
} > eval_NIPS/n_FMNIST_IRMAE_N_50.out 2>&1  &









# {
#     nohup python3 run_eval.py -l test_NIPS/test_NIPS__MNIST__IRMAE__128__0.001__50.pth \
#                         -o eval_NIPS/ -d cuda:0 -a NIPS
#     nohup python3 run_eval.py -l test_NIPS/test_NIPS__FMNIST__IRMAE__128__0.001__50.pth \
#                         -o eval_NIPS/ -d cuda:0 -a NIPS
# } > eval_NIPS/out_IRMAE_F_MNIST.out  2>&1     &

# {
#     nohup python3 run_eval.py -l test_NIPS/test_NIPS__FMNIST__AE__128__0.001__50.pth \
#                         -o eval_NIPS/ -d cuda:2 -a NIPS
#     nohup python3 run_eval.py -l test_NIPS/test_NIPS__FMNIST__VAE__128__0.001__50.pth \
#                         -o eval_NIPS/ -d cuda:2 -a NIPS
# } > eval_NIPS/out_AE_VAE_FMNIST.out  2>&1     &

# nohup python3 run_eval.py -l test_NIPS/test_NIPS__FMNIST__LRAE__128__0.1__50.pth \
#                         -o eval_NIPS/ -d cuda:4 -a NIPS > eval_NIPS/out_LRAE_FMNIST.out  2>&1   &

# python3 run_train.py -m VAE -A 0.001 -b 1024 -d cuda:0 

# echo "run nohup!!!!!!!!!!!!!!!"

# nohup python3 run_eval.py        -l test_NIPS/test_bl_NIPS_512_0.0001__CELEBA__VAE__512__0.001__50.pth \
#                         -o eval_NIPS/data_b -d cuda:0 -a NIPS > eval_NIPS/bl_CELEBA_0.0001_VAE_e50.out 2>&1

# {
# for EPOCH in 50
#     do
#         nohup python3 run_eval.py        -l test_NIPS/test_bl_NIPS_512_0.0001__CELEBA__VAE__512__0.001__${EPOCH}.pth \
#                         -o eval_NIPS/data_b -d cuda:0 -a NIPS 
#     done
# } > eval_NIPS/bl_CELEBA_0.0001_VAE_e50_75_100.out 2>&1      &




# {
# for EPOCH in 25 125
#     do
#         nohup python3 run_eval.py        -l test_NIPS/test_bl_NIPS_512_0.0001__CELEBA__VAE__512__0.001__${EPOCH}.pth \
#                         -o eval_NIPS/data_b -d cuda:2 -a NIPS
#     done
# } > eval_NIPS/bl_CELEBA_0.0001_VAE_e25_125.out 2>&1     &

# {
# for EPOCH in 200 0
#     do
#         nohup python3 run_eval.py        -l test_NIPS/test_bl_NIPS_512_0.0001__CELEBA__VAE__512__0.001__${EPOCH}.pth \
#                         -o eval_NIPS/data_b -d cuda:3 -a NIPS
#     done
# } > eval_NIPS/bl_CELEBA_0.0001_VAE_e200_0.out 2>&1      &

# ALPHA=0.00057
# ALPHA1=0.0032
# {
#     for EPOCH in 25 50 75 100 125 150 175 200
#         do
#             nohup python3 run_eval.py        -l test_NIPS/test_bl_NIPS_512_0.0001__CELEBA__VAE__128__0.001__${EPOCH}.pth \
#                             -o eval_NIPS/data_b -d cuda:2 -a NIPS > eval_NIPS/bl_0.0001_VAE_e${EPOCH}.out 2>&1 &
#         done
# } 


# {
#     for EPOCH in 25 50 75 100 125 150 175 200
#         do
#             nohup python3 run_eval.py        -l test_NIPS/data_b/test_bl_NIPS_1024_${ALPHA1}__MNIST__VAE__128__0.001__${EPOCH}.pth \
#                             -o eval_NIPS/data_b -d cuda:3 -a NIPS
#         done
# } > eval_NIPS/o_bl_${ALPHA1}_VAE_200.out 2>&1 &





# nohup python3 run_eval.py        -l test_NIPS/data_b/test_bl_NIPS_1024_0.0001__MNIST__VAE__128__0.001__175.pth \
#                          -o eval_NIPS/data_b -d cuda:2 -a NIPS > eval_NIPS/o_bl_0.0001_VAE_175.out 2>&1 &
# nohup python3 run_eval.py        -l test_NIPS/data_b/test_bl_NIPS_1024_0.0001__MNIST__VAE__128__0.001__200.pth \
#                          -o eval_NIPS/data_b -d cuda:3 -a NIPS > eval_NIPS/o_bl_0.0001_VAE_200.out 2>&1 &
# nohup python3 run_eval.py        -l test_NIPS/data_b/test_bl_NIPS_1024_0.0001__MNIST__VAE__128__0.001__125.pth \
#                          -o eval_NIPS/data_b -d cuda:2 -a NIPS > eval_NIPS/o_bl_0.0001_VAE_125.out 2>&1 &
# nohup python3 run_eval.py        -l test_NIPS/data_b/test_bl_NIPS_1024_0.0001__MNIST__VAE__128__0.001__150.pth \
#                          -o eval_NIPS/data_b -d cuda:3 -a NIPS > eval_NIPS/o_bl_0.0001_VAE_150.out 2>&1 &
# nohup python3 run_eval.py        -l test_NIPS/data_b/test_bl_NIPS_1024_0.0001__MNIST__VAE__128__0.001__50.pth \
#                          -o eval_NIPS/data_b -d cuda:4 -a NIPS > eval_NIPS/o_bl_0.0001_VAE_50.out 2>&1 &


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


# nohup python3 run_eval.py -l test_NIPS/test_b_NIPS_1024__MNIST__VAE__128__0.001__50.pth -o eval_NIPS/data_o -d cuda:0 -a NIPS \
#         > $eval_dir_name/o_VAE_b1024.out 2>&1 &
# nohup python3 run_eval.py -l test_NIPS/test_b_NIPS_512__MNIST__VAE__128__0.001__50.pth  -o eval_NIPS/data_o -d cuda:1 -a NIPS \
#         > $eval_dir_name/o_VAE_b512.out 2>&1 &
# nohup python3 run_eval.py -l test_NIPS/test_b_NIPS_256__MNIST__VAE__128__0.001__50.pth  -o eval_NIPS/data_o -d cuda:2 -a NIPS \
#         > $eval_dir_name/o_VAE_b256.out 2>&1 &
# nohup python3 run_eval.py -l test_NIPS/test_b_NIPS_128__MNIST__VAE__128__0.001__50.pth  -o eval_NIPS/data_o -d cuda:4 -a NIPS \
#         > $eval_dir_name/o_VAE_b128.out 2>&1 &



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