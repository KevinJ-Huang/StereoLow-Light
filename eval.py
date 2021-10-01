'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''
import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import argparse

import utils.util as util
import data.util as data_util
import models.archs.EnhanceN_arch as EnhanceN_arch

def dataload_test(img_path):
    img_numpy = cv2.imread(img_path,cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
    # img_numpy = cv2.resize(img_numpy,(512,512))
    img_numpy = img_numpy[:, :, [2, 1, 0]]
    img_numpy = torch.from_numpy(np.ascontiguousarray(np.transpose(img_numpy, (2, 0, 1)))).float()
    img_numpy = img_numpy.unsqueeze(0)
    return img_numpy


def forward_eval(model,img_left):
    with torch.no_grad():
        model_output = model(img_left)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output


def main(root, save_folder, imageLists, modelPath):
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ############################################################################
    #### model
    model_path =  modelPath
    model = EnhanceN_arch.DRBN()

    #### dataset
    test_dataset_folder = root
    save_imgs = True
    util.mkdirs(save_folder)
    # util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    # logger = logging.getLogger('base')

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    print("load model successfully")
    model.eval()
    model = model.to(device)

    groups = [line.rstrip() for line in open(os.path.join(imageLists))]
    image_filenames = [group.split('|') for group in groups]

    # process each image
    j = 0
    for img_idx in range(len(image_filenames)):

        img_name = osp.splitext(osp.basename(image_filenames[img_idx][0]))[0]
        img_left = dataload_test(image_filenames[img_idx][0]).to(device)
        # img_right = dataload_test(image_filenames[img_idx][2]).to(device)

        # for tx in range(2):
        #     img_left = img_left_ori[:, :, :, max(0, 736 * tx - 32):min(736 * (tx + 1) + 32, img_left_ori.shape[3])]
        #     img_right = img_right_ori[:, :, :, max(0, 736 * tx - 32):min(736 * (tx + 1) + 32, img_right_ori.shape[3])]

        output = forward_eval(model,img_left)
        output = util.tensor2img(output.squeeze(0))

        if save_imgs:
            cv2.imwrite(osp.join(save_folder, '{}.png'.format(img_name)), output)

        j = j + 1
        print("process %d th image" % j)

    print('################ Finish Testing ################')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/1760921465/Stereo_enhancement/Midllery_small/test/low',  metavar='PATH', help='validation dataset root dir')
    parser.add_argument('--imageLists', type=str, default='/code/STEN/data/groups_test_small.txt', metavar='FILE', help='record video ids')
    parser.add_argument('--save_folder', type=str, default='/output/enhance/', metavar='PATH', help='save results')
    parser.add_argument('--modelPath', type=str, default='/model/1760921465/NewWork2021/DRBN/Midllery_small.pt', help='Model path')

    args = parser.parse_args()

    root = args.root
    imageLists = args.imageLists
    save_folder = args.save_folder
    modelPath = args.modelPath

    main(root, save_folder, imageLists, modelPath)
