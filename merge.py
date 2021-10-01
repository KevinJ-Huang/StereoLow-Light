import cv2
import numpy as np
import os

def calWeight(d, k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def imgFusion(img1, img2, overlap, left_right=True):
    '''
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    '''
    # 这里先暂时考虑平行向融合
    w = calWeight(overlap, 0.05)  # k=5 这里是超参

    if left_right:  # 左右融合
        row,  col, channels = img1.shape
        row1, col1, channels1 = img2.shape
        img_res =  np.zeros((row, col + col1 - overlap,3))
        for c in range(channels):
            img_new = np.zeros((row, col + col1 - overlap))
            img_new[:, :col] = img1[:,:,c]
            w_expand = np.tile(w, (row, 1))  # 权重扩增
            img_new[:, col - overlap:col] = (1 - w_expand) * img1[:, col - overlap:col, c] + w_expand * img2[:, :overlap,c]
            img_new[:, col:] = img2[:, overlap:,c]
            img_res[:,:,c] = img_new

    else:  # 上下融合
        row, col, channels = img1.shape
        img_res = np.zeros((row, 2 * col - overlap, 3))
        for c in range(channels):
            img_new = np.zeros((2 * row - overlap, col))
            img_new[:row, :] = img1[:,:,c]
            w = np.reshape(w, (overlap, 1))
            w_expand = np.tile(w, (1, col))
            img_new[row - overlap:row, :] = (1 - w_expand) * img1[row - overlap:row, :,c] + w_expand * img2[:overlap, :,c]
            img_new[row:, :] = img2[overlap:, :,c]
            img_res[:, :, c] = img_new

    return img_res



if __name__ =="__main__":

    if not os.path.exists('/media/ustc-ee-huangjie/KESU/Datasets/stereodataset/Midllery_small_result/Stereo/left/'):
        os.makedirs('/media/ustc-ee-huangjie/KESU/Datasets/stereodataset/Midllery_small_result/Stereo/left/')

    files = sorted(os.listdir('/media/ustc-ee-huangjie/KESU/Datasets/stereodataset/Midllery_small_result/Stereo/results1'))
    file_list = []
    for file in files:

        filename =file[:-6]
        if not filename in file_list:

            img1 = cv2.imread("/media/ustc-ee-huangjie/KESU/Datasets/stereodataset/Midllery_small_result/Stereo/results1/"+filename+'_0.png',cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread("/media/ustc-ee-huangjie/KESU/Datasets/stereodataset/Midllery_small_result/Stereo/results1/"+filename+'_1.png',cv2.IMREAD_UNCHANGED)
            img1 = (img1 - img1.min())/img1.ptp()
            img2 = (img2 - img2.min())/img2.ptp()
            img_new1 = imgFusion(img1,img2,overlap=64,left_right=True)

            # img3 = cv2.imread(
            #     "/media/ustc-ee-huangjie/KESU/Datasets/stereodataset/Midllery_result/MIRNet/results1/" + filename + '_2.png',
            #     cv2.IMREAD_UNCHANGED)
            # img4 = cv2.imread(
            #     "/media/ustc-ee-huangjie/KESU/Datasets/stereodataset/Midllery_result/MIRNet/results1/" + filename + '_3.png',
            #     cv2.IMREAD_UNCHANGED)
            # img3 = (img3 - img3.min()) / img3.ptp()
            # img4 = (img4 - img4.min()) / img4.ptp()
            # img_new2 = imgFusion(img3, img4, overlap=64, left_right=True)
            # 
            # img_new1 = (img_new1 - img_new1.min()) / img_new1.ptp()
            # img_new2 = (img_new2 - img_new2.min()) / img_new2.ptp()
            # img_new = imgFusion(img_new1, img_new2, overlap=64, left_right=True)


            cv2.imwrite('/media/ustc-ee-huangjie/KESU/Datasets/stereodataset/Midllery_small_result/Stereo/left/'+filename+'.png',np.uint8(img_new1*255.0))

        file_list.append(filename)