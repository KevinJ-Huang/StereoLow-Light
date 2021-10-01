import torch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.EnhanceN_arch as EnhanceN_arch
import models.archs.EnhanceN_arch1 as EnhanceN_arch1
import models.archs.EnhanceN_arch2 as EnhanceN_arch2
import models.archs.EnhanceN_arch3 as EnhanceN_arch3
import models.archs.EnhanceN_arch4 as EnhanceN_arch4
import models.archs.EnhanceN_arch5 as EnhanceN_arch5
import models.archs.EnhanceN_arch6 as EnhanceN_arch6



# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    # image restoration
    if which_model == 'STEN':
        netG = EnhanceN_arch.Net()
    elif which_model == 'STEN1':
        netG = EnhanceN_arch1.Net()
    elif which_model == 'STEN2':
        netG = EnhanceN_arch2.Net()
    elif which_model == 'STEN3':
        netG = EnhanceN_arch3.Net()
    elif which_model == 'STEN4':
        netG = EnhanceN_arch4.Net()
    elif which_model == 'STEN5':
        netG = EnhanceN_arch5.Net()
    elif which_model == 'STEN6':
        netG = EnhanceN_arch6.Net()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
