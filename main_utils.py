

def get_models_class_list(DATASET_TYPE, ARCHITECTURE_TYPE):
    # (ARCHITECTURE_TYPE.upper() in ['V1'])
    if (ARCHITECTURE_TYPE.upper() in ['V1']) and (DATASET_TYPE in ['MNIST']):
        from models.R1AE import ConvLRAE, ConvVAE, ConvAE
        print("models were downloaded from 'models.R1AE'")
        
    elif (ARCHITECTURE_TYPE.upper() in ['V1']) and (DATASET_TYPE in ['CIFAR10', 'CELEBA']):
        from models.R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE
        print("models were downloaded from 'models.R1AE_CelebA'")
        
    # (ARCHITECTURE_TYPE.upper() in ['NISP'])
    
    if (ARCHITECTURE_TYPE.upper() in ['NIPS']) and (DATASET_TYPE in ['MNIST']):
        from models.NIPS_R1AE_MNIST import ConvLRAE, ConvVAE, ConvAE
        print("models were downloaded from 'models.NIPS_R1AE_MNIST'")
        
    elif (ARCHITECTURE_TYPE.upper() in ['NIPS']) and (DATASET_TYPE in ['CIFAR10', 'CELEBA']):
        from models.NIPS_R1AE_CelebA import ConvLRAE, ConvVAE, ConvAE
        print("models were downloaded from 'models.NIPS_R1AE_CelebA'")

    else:
        assert False, 'Error, bed combination of DATASET_TYPE and ARCHITECTURE_TYPE'
        
    return ConvAE, ConvVAE, ConvLRAE
    