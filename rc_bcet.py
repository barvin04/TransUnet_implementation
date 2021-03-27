import os
import cv2
import numpy as np


def BCET(min_out_img, max_out_img, mean_out_img, in_img):
    """
    Obtain and apply BCET function for given input image and target
    output params.
    Translated from MATLAB code at
    (https://www.imageeprocessing.com/2017/11/balance-contrast
    -enhancement-technique.html)
    Args:
       min_out_img (float): min value of target image
       max_out_img (float): max value of target image
       mean_out_img (float): mean value of target image
       in_img (np.array): input image to be transformed
    Returns:
       out_img (np.array): transformed output image
    """
    in_img = in_img.astype('float32')  # INPUT IMAGE
    Lmin = np.min(in_img)  # MINIMUM OF INPUT IMAGE
    Lmax = np.max(in_img)  # MAXIMUM OF INPUT IMAGE
    Lmean = np.mean(in_img)  # MEAN OF INPUT IMAGE
    LMssum = np.mean(in_img**2)  # MEAN SQUARE SUM OF INPUT IMAGE

    bnum = ((Lmax**2)*(mean_out_img - min_out_img)
            - LMssum*(max_out_img - min_out_img)
            + (Lmin**2)*(max_out_img - mean_out_img))
    bden = (2*(Lmax * (mean_out_img - min_out_img)
               - Lmean*(max_out_img - min_out_img)
               + Lmin * (max_out_img - mean_out_img)))

    b = bnum/bden

    a = (max_out_img-min_out_img)/((Lmax-Lmin)*(Lmax+Lmin-2*b))

    c = min_out_img - a * (Lmin-b)**2

    out_img = a * ((in_img-b)**2) + c  # PARABOLIC FUNCTION

    return out_img


if __name__ == '__main__':
    in_path = '/home/ruchi/lung_seg_dataset/all_data/images/'
    out_path = '/home/ruchi/lung_seg_dataset/all_data/bcet_images'
    for fname in os.listdir(in_path):
        img = cv2.imread(os.path.join(in_path, fname), cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, (512, 512), cv2.INTER_AREA)
        img = BCET(0, 255, 86, img)
        img = (img - np.mean(img)) / np.std(img)
        np.save(os.path.join(out_path, fname.rsplit('.', 1)[0]+'.npy'), img)
