import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (
    Resize, RandomHorizontalFlip, RandomVerticalFlip)
import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


BuildingFlooded = np.array([255, 0, 0])     # label 0
BuildingNonFlooded = np.array([0, 255, 0])  # label 1
RoadFlooded = np.array([0, 255, 120])       # label 2
RoadNonFlooded = np.array([0, 0, 255])      # label 3
Water = np.array([255, 0, 255])             # label 4
Tree = np.array([70, 70, 220])              # label 5
Vehicle = np.array([102, 102, 156])         # label 6
Pool = np.array([190, 153, 153])            # label 7
Grass = np.array([180, 165, 180])           # label 8
Background = np.array([0, 0, 0])            # label 9

num_classes = 9


# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser(
        description='Split huge RS image to small patches')
    parser.add_argument("--img-dir", default="data/floodnet/train_images")
    parser.add_argument("--mask-dir", default="data/floodnet/train_masks")
    parser.add_argument("--output-img-dir",
                        default="data/floodnet/train/images_1024")
    parser.add_argument("--output-mask-dir",
                        default="data/floodnet/train/masks_1024")
    parser.add_argument("--eroded", action='store_true')
    parser.add_argument("--gt", action='store_true')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--val-scale", type=float, default=1.0)
    parser.add_argument("--split-size", type=int, default=1024,
                        help='clipped size of image after preparation')
    parser.add_argument("--stride", type=int, default=512,
                        help='stride of clipping original images')

    return parser.parse_args()


def get_img_mask_padded(image, mask, patch_size):
    img, mask = np.array(image), np.array(mask)
    oh, ow = img.shape[0], img.shape[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh
    h, w = oh + height_pad, ow + width_pad

    pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                               border_mode=cv2.BORDER_CONSTANT, value=0)(image=img)

    pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                                border_mode=cv2.BORDER_CONSTANT, value=6)(image=mask)

    img_pad, mask_pad = pad_img['image'], pad_mask['image']
    img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
    mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)

    return img_pad, mask_pad


# Now replace RGB to integer values to be used as labels.
# Find pixels with combination of RGB for the above defined arrays...
# If matches then replace all values in that pixel with a specific integer
def rgb_to_2D_label(_label):
    """
    Supply our lable masks as input in RGB format.
    Replace pixels with specific RGB values ...
    """
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)

    label_seg[np.all(_label.transpose([1, 2, 0]) ==
                     BuildingFlooded, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) ==
                     BuildingNonFlooded, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == RoadFlooded, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) ==
                     RoadNonFlooded, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Water, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 5
    label_seg[np.all(_label.transpose([1, 2, 0]) == Vehicle, axis=-1)] = 6
    label_seg[np.all(_label.transpose([1, 2, 0]) == Pool, axis=-1)] = 7
    label_seg[np.all(_label.transpose([1, 2, 0]) == Grass, axis=-1)] = 8
    label_seg[np.all(_label.transpose([1, 2, 0]) == Background, axis=-1)] = 9

    return label_seg


def image_augment(image, mask, patch_size, mode='train', val_scale=1.0):
    image_list = []
    mask_list = []
    image_width, image_height = image.size[1], image.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]

    assert image_height == mask_height and image_width == mask_width

    if mode == 'train':
        h_vlip = RandomHorizontalFlip(p=1.0)
        v_vlip = RandomVerticalFlip(p=1.0)
        image_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(mask.copy())
        image_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(mask.copy())

        image_list_train = [image, image_h_vlip, image_v_vlip]
        mask_list_train = [mask, mask_h_vlip, mask_v_vlip]

        for i in range(len(image_list_train)):
            image_tmp, mask_tmp = get_img_mask_padded(
                image_list_train[i], mask_list_train[i], patch_size)
            mask_tmp = rgb_to_2D_label(mask_tmp.copy())
            image_list.append(image_tmp)
            mask_list.append(mask_tmp)
    else:
        rescale = Resize(size=(int(image_width * val_scale),
                         int(image_height * val_scale)))
        image, mask = rescale(image.copy()), rescale(mask.copy())
        image, mask = get_img_mask_padded(
            image.copy(), mask.copy(), patch_size)
        mask = rgb_to_2D_label(mask.copy())
        image_list.append(image)
        mask_list.append(mask)

    return image_list, mask_list


def randomsizedcrop(image, mask):
    # assert image.shape[:2] == mask.shape
    h, w = image.shape[0], image.shape[1]
    crop = albu.RandomSizedCrop(min_max_height=(
        int(3*h//8), int(h//2)), width=h, height=w)(image=image.copy(), mask=mask.copy())

    img_crop, mask_crop = crop['image'], crop['mask']

    return img_crop, mask_crop


def car_aug(image, mask):
    assert image.shape[:2] == mask.shape

    v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    rotate_90 = albu.RandomRotate90(p=1.0)(
        image=image.copy(), mask=mask.copy())

    image_vflip, mask_vflip = v_flip['image'], v_flip['mask']
    image_hflip, mask_hflip = h_flip['image'], h_flip['mask']
    image_rotate, mask_rotate = rotate_90['image'], rotate_90['mask']

    image_list = [image, image_vflip, image_hflip, image_rotate]
    mask_list = [mask, mask_vflip, mask_hflip, mask_rotate]

    return image_list, mask_list


def floodnet_format(inp):
    """
    Original image of FloodNet dataset is very large, thus pre-processing
    of them is adopted. Given fixed split size and stride size to generate
    splitted image, the intersection of width and height is determined.
    For example, given one 4000 x 3000 original image, the split size is
    1024 and stride size is 1024, thus it would generate 4 x 3 = 12 images
    whose size are all 1024 x 1024.
    """

    (img_path, mask_path, imgs_output_dir, masks_output_dir,
     mode, val_scale, split_size, stride) = inp
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]

    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')

    # img and mask shape: WxHxC
    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(
    ), patch_size=split_size, mode=mode, val_scale=val_scale)

    assert img_filename == mask_filename and len(image_list) == len(mask_list)

    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]

        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                img_tile = img[y:y + split_size, x:x + split_size]
                mask_tile = mask[y:y + split_size, x:x + split_size]

                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size:
                    image_crop, mask_crop = randomsizedcrop(
                        img_tile, mask_tile)
                    bins = np.array(range(num_classes + 1))
                    class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)
                    cf = class_pixel_counts / \
                        (mask_crop.shape[0] * mask_crop.shape[1])

                    if cf[4] > 0.1 and mode == 'train':
                        car_imgs, car_masks = car_aug(image_crop, mask_crop)

                        for i in range(len(car_imgs)):
                            out_img_path = os.path.join(
                                imgs_output_dir, "{}_{}_{}_{}.jpg".format(img_filename, m, k, i))
                            cv2.imwrite(out_img_path, car_imgs[i])

                            out_mask_path = os.path.join(
                                masks_output_dir, "{}_{}_{}_{}.png".format(mask_filename, m, k, i))
                            cv2.imwrite(out_mask_path, car_masks[i])

                    else:
                        out_img_path = os.path.join(
                            imgs_output_dir, "{}_{}_{}.jpg".format(img_filename, m, k))
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(
                            masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)

                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()

    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride

    img_paths = glob.glob(os.path.join(imgs_dir, "*.jpg"))
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))

    img_paths.sort()
    mask_paths.sort()

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, mode, val_scale,
            split_size, stride) for img_path, mask_path in zip(img_paths, mask_paths)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(floodnet_format, inp)
    t1 = time.time()
    split_time = t1 - t0

    print('images spliting spends: {} s'.format(split_time))
