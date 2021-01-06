import os
import cv2
import imageio
import numpy as np
from PIL import ImageEnhance
from PIL import Image
import albumentations as A
augment_level = 8

def get_files_in_directory(path):
    if not path.endswith('/'):
        path = path + "/"
    file_list = [path + f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and not f.startswith('.'))]
    return file_list

def load_image(filename, width=0, height=0, channels=0, alignment=0, print_console=True):
    image = np.atleast_3d(imageio.imread(filename))
    # if there is alpha plane, cut it
    if image.shape[2] >= 4:
        image = image[:, :, 0:3]

    return image

def save_image(filename, image, print_console=True):
    image = image.astype(np.uint8)
    imageio.imwrite(filename, image)

def change_contrast(name, factor):
    img = Image.open(name)
    enhancer = ImageEnhance.Contrast(img)
    im_output = enhancer.enhance(factor)
    return np.array(im_output)

def change_brightness(name, factor):
    img = Image.open(name)
    enhancer = ImageEnhance.Brightness(img)
    im_output = enhancer.enhance(factor)
    return np.array(im_output)

def centerCrop(name):
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = A.CenterCrop(height = int(img_rgb.shape[0]*0.7), width = int(img_rgb.shape[1]*0.7), always_apply=False, p=1.0)(image = img_rgb)['image']
    return img

def gaussianNoise(name):
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = A.GaussNoise(var_limit=(10.0, 60.0), always_apply=False, p=1.0)(image = img_rgb)['image']
    return img

training_filenames = get_files_in_directory('/Users/ryan/Desktop/Visual Recognition/2020fall_DLCV/HW4/train' )
target_dir = '/Users/ryan/Desktop/Visual Recognition/2020fall_DLCV/HW4/train_aug/'
if not os.path.exists(target_dir):
        os.makedirs(target_dir)

counter = 1
for file_path in training_filenames:
        org_image = load_image(file_path)

        filename = os.path.basename(file_path)
        filename, extension = os.path.splitext(filename)

        new_filename = target_dir + filename

        save_image(new_filename + extension, org_image)

        if augment_level >= 2:
            ud_image = np.flipud(org_image)
            save_image(new_filename + "_v" + extension, ud_image)
        if augment_level >= 3:
            lr_image = np.fliplr(org_image)
            save_image(new_filename + "_h" + extension, lr_image)
        if augment_level >= 4:
            lr_image = np.fliplr(org_image)
            lrud_image = np.flipud(lr_image)
            save_image(new_filename + "_hv" + extension, lrud_image)

        if augment_level >= 5:
            rotated_image1 = np.rot90(org_image)
            save_image(new_filename + "_r1" + extension, rotated_image1)
        if augment_level >= 6:
            rotated_image2 = np.rot90(org_image, -1)
            save_image(new_filename + "_r2" + extension, rotated_image2)

        if augment_level >= 7:
            rotated_image1 = np.rot90(org_image)
            ud_image = np.flipud(rotated_image1)
            save_image(new_filename + "_r1_v" + extension, ud_image)
        if augment_level >= 8:
            rotated_image2 = np.rot90(org_image, -1)
            ud_image = np.flipud(rotated_image2)
            save_image(new_filename + "_r2_v" + extension, ud_image)

        if augment_level >= 9:
            contrast_img = change_contrast(file_path, factor=2.0)
            contrast_img = np.rot90(contrast_img)
            cv2.imwrite(new_filename + "_c" + extension, contrast_img)
        if augment_level >= 10:
            contrast_img = change_contrast(file_path, factor=2.0)
            cv2.imwrite(new_filename + "_c_r1" + extension, contrast_img)
        if augment_level >= 11:
            contrast_img = change_contrast(file_path, factor=2.0)
            contrast_img = np.rot90(contrast_img, -1)
            cv2.imwrite(new_filename + "_c_r2" + extension, contrast_img)
        if augment_level >= 12:
            contrast_img = change_contrast(file_path, factor=2.0)
            contrast_img = np.rot90(contrast_img, 1)
            contrast_img = np.flipud(contrast_img)
            cv2.imwrite(new_filename + "_c_r1_v" + extension, contrast_img)
        if augment_level >= 13:
            contrast_img = change_contrast(file_path, factor=2.0)
            contrast_img = np.rot90(contrast_img, -1)
            contrast_img = np.flipud(contrast_img)
            cv2.imwrite(new_filename + "_c_r2_v" + extension, contrast_img)

        if augment_level >= 14:
            bright_img = change_brightness(file_path, factor=1.5)
            cv2.imwrite(new_filename + "_b" + extension, bright_img)
        if augment_level >= 15:
            dark_img = change_brightness(file_path, factor=0.5)
            cv2.imwrite(new_filename + "_d" + extension, dark_img)
        if augment_level >= 16:
            bright_img = change_brightness(file_path, factor=1.5)
            bright_img = np.rot90(bright_img)
            cv2.imwrite(new_filename + "_b_r1" + extension, bright_img)
        if augment_level >= 17:
            bright_img = change_brightness(file_path, factor=1.5)
            bright_img = np.rot90(bright_img, -1)
            cv2.imwrite(new_filename + "_b_r2" + extension, bright_img)
        if augment_level >= 18:
            dark_img = change_brightness(file_path, factor=0.5)
            dark_img = np.rot90(dark_img)
            cv2.imwrite(new_filename + "_d_r1" + extension, dark_img)
        if augment_level >= 19:
            dark_img = change_brightness(file_path, factor=0.5)
            dark_img = np.rot90(dark_img, -1)
            cv2.imwrite(new_filename + "_d_r2" + extension, dark_img)
        if augment_level >= 20:
            crop_img = centerCrop(file_path)
            cv2.imwrite(new_filename + "_crop" + extension, crop_img)
        if augment_level >= 21:
            noise_img = gaussianNoise(file_path)
            cv2.imwrite(new_filename + "_noise" + extension, noise_img)

        print(counter)
        counter+=1
