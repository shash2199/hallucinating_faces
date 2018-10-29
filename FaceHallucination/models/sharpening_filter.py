# from scipy import signal
# from scipy import ndimage
import os
import numpy as np
import cv2

def sharpen_image(path_to_aligned_images, image_name, path_to_results):
    box_filter = (-1*np.ones((3, 3))) / 9
    # print(box_filter)
    sharp_filter = np.zeros((3, 3))
    sharp_filter[1, 1] = 1
    # print(sharp_filter)

    image = cv2.imread(os.path.join(path_to_aligned_images, image_name))
    image = cv2.resize(image, (96, 128))
    sharp_result = cv2.filter2D(image, -1, sharp_filter)
    box_result = cv2.filter2D(image, -1, box_filter)
    result = sharp_result - box_result
    if path_to_results[-1] != '/':
        path_to_results += '/'
    img_split = image_name.split(".")
    image_name = img_split[0] + "_sharpening_filter.png"
    # result = cv2.resize(result, (96, 128))
    cv2.imwrite(path_to_results + image_name, result)



if __name__=="__main__":
    path_to_aligned_images = '../aligned_data/profile_pictures_aligned_low_res'
    path_to_results = '../results/profile_pictures_sharpen'
    for filename in os.listdir(path_to_aligned_images):
        print(filename)
        sharpen_image(path_to_aligned_images, filename, path_to_results)