

def get_models_class_list(DATASET_TYPE, ARCHITECTURE_TYPE):
    ARCHITECTURE_TYPE = ARCHITECTURE_TYPE.upper()
    DATASET_TYPE = DATASET_TYPE.upper() 
    
    
    # need to add IRMAE in V1!!!
    # (ARCHITECTURE_TYPE.upper() in ['V1'])
    if (ARCHITECTURE_TYPE in ['V1']) and (DATASET_TYPE in ['MNIST', 'FMNIST', 'FASHIONMNIST']):
        from models.R1AE import ConvLRAE, ConvVAE, ConvAE
        print("models were downloaded from 'models.R1AE'")
        
    elif (ARCHITECTURE_TYPE in ['V1']) and (DATASET_TYPE in ['CIFAR10', 'CELEBA']):
        from models.R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE
        print("models were downloaded from 'models.R1AE_CelebA'")
        
    # (ARCHITECTURE_TYPE.upper() in ['NISP'])
    if (ARCHITECTURE_TYPE in ['NIPS']) and (DATASET_TYPE in ['MNIST', 'FMNIST', 'FASHIONMNIST']):
        from models.NIPS_R1AE_MNIST import ConvLRAE, ConvVAE, ConvAE, ConvIRMAE
        print("models were downloaded from 'models.NIPS_R1AE_MNIST'")
        
    elif (ARCHITECTURE_TYPE in ['NIPS']) and (DATASET_TYPE in ['CIFAR10', 'CELEBA']):
        from models.NIPS_R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE, ConvIRMAE
        print("models were downloaded from 'models.NIPS_R1AE_CelebA'")

    else:
        assert False, 'Error, bed combination of DATASET_TYPE and ARCHITECTURE_TYPE'
        
    return ConvLRAE, ConvAE, ConvVAE, ConvIRMAE





def get_eval_parameters(DATASET_TYPE):
    eval_params = {}

    if DATASET_TYPE.upper() in ['MNIST', 'FMNIST', 'FASHIONMNIST']:        
        eval_params['N_GM_COMPONENTS'] = 4   
    elif DATASET_TYPE.upper() in ['CIFAR10', 'CELEBA']:
        eval_params['N_GM_COMPONENTS'] = 10
    else:
        print("Warning! the default eval parameter will be used!")
        eval_params['N_GM_COMPONENTS'] = 10
        
    return eval_params
    
 
def get_base_model_NIPS_parameters(DATASET_TYPE):
    # models_params = {}
    # setup model parameters
    if DATASET_TYPE.upper() in ['MNIST', 'FMNIST', 'FASHIONMNIST']:        
        IN_FEATURES = 256*2*2
        BOTTLENECK = 128
        C_H_W = [128, 8, 8]
        OUT_FEATURES = C_H_W[0]*C_H_W[1]*C_H_W[2]
        DS_IN_CHANNELS = 1
        MIDDLE_MATRIXES = 4       
        
    elif DATASET_TYPE.upper() in ['CIFAR10']:
        IN_FEATURES = 1024*2*2
        BOTTLENECK = 512
        C_H_W = [1024, 8, 8]
        OUT_FEATURES = C_H_W[0]*C_H_W[1]*C_H_W[2] 
        DS_IN_CHANNELS = 3
        MIDDLE_MATRIXES = 4 

    elif DATASET_TYPE.upper() in ['CELEBA']:        
        IN_FEATURES = 1024*4*4
        BOTTLENECK = 512
        C_H_W = [1024, 8, 8]
        OUT_FEATURES = C_H_W[0]*C_H_W[1]*C_H_W[2]
        DS_IN_CHANNELS = 3
        MIDDLE_MATRIXES = 4 

    else:
        print("Warning! the module parameter setups was not setuped!")
           
    models_params = {'IN_FEATURES': IN_FEATURES, 'BOTTLENECK': BOTTLENECK, 'OUT_FEATURES': OUT_FEATURES,
                     'DS_IN_CHANNELS': DS_IN_CHANNELS, 'C_H_W': C_H_W, 'MIDDLE_MATRIXES': MIDDLE_MATRIXES}
            
    return models_params

def get_base_model_V1_parameters(DATASET_TYPE):
    # models_params = {}
    
    if DATASET_TYPE in ['MNIST', 'FMNIST', 'FASHIONMNIST']:
        IN_FEATURES = 256*2*2
        BOTTLENECK = 128
        C_H_W = [256, 2, 2]
        OUT_FEATURES = C_H_W[0]*C_H_W[1]*C_H_W[2]
        DS_IN_CHANNELS = 1
        MIDDLE_MATRIXES = 8 

    elif DATASET_TYPE in ['CIFAR10']:
        IN_FEATURES = 1024*2*2
        BOTTLENECK = 512 
        C_H_W = [1024, 2, 2]
        OUT_FEATURES = C_H_W[0]*C_H_W[1]*C_H_W[2]
        DS_IN_CHANNELS = 3
        MIDDLE_MATRIXES = 4 
    elif DATASET_TYPE in ['CelebA', 'CELEBA']:
        IN_FEATURES = 1024*4*4
        BOTTLENECK = 512
        C_H_W = [1024, 4, 4]
        OUT_FEATURES = C_H_W[0]*C_H_W[1]*C_H_W[2]
        DS_IN_CHANNELS = 3
        MIDDLE_MATRIXES = 4 

    else:
        print("Warning! the module parameter setups was not setuped!")
        
    models_params = {'IN_FEATURES': IN_FEATURES, 'BOTTLENECK': BOTTLENECK, 'OUT_FEATURES': OUT_FEATURES,
                     'DS_IN_CHANNELS': DS_IN_CHANNELS, 'C_H_W': C_H_W, 'MIDDLE_MATRIXES': MIDDLE_MATRIXES}
    
    return models_params

   
    
def  get_base_model_parameters(DATASET_TYPE, ARCHITECTURE_TYPE):
    ARCHITECTURE_TYPE = ARCHITECTURE_TYPE.upper()
    DATASET_TYPE = DATASET_TYPE.upper()    
    if ARCHITECTURE_TYPE.upper() == 'NIPS':
        models_params = get_base_model_NIPS_parameters(DATASET_TYPE)
    elif ARCHITECTURE_TYPE == 'V1':
        models_params = get_base_model_V1_parameters(DATASET_TYPE)
    else:
        assert 0, f"Error, models_params for ARCHITECTURE_TYPE={ARCHITECTURE_TYPE} are not defined"
    return models_params
    


