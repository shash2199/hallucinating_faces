import os, os.path
import numpy as np
from PIL import Image
from copy import deepcopy
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt

# Creates low resolution training images along with corresponding high resolution training_images
def create_data_from_images(
  data_path,
  low_res_image_path,
  low_res_training_path,
  high_res_training_path,
  low_res_testing_path,
  high_res_testing_path,
  split_factor,
  sigma,
  downsample_factor,
  levels):
  high_res_images = os.listdir(data_path)
  num_samples = len(high_res_images)

  sample_image = scipy.misc.imread(os.path.join(data_path, high_res_images[0]), mode='L')
  width, height = sample_image.shape
  sample_pyramid = generate_gaussian_pyramid(
      im=sample_image,
      downsample_factor=downsample_factor,
      sigma=sigma,
      levels=levels)

  low_res_width, low_res_height = sample_pyramid[-1].shape
  del sample_image
  del sample_pyramid
  high_res_data = np.zeros((width * height, num_samples))
  low_res_data = np.zeros((low_res_width * low_res_height, num_samples))

  for i in range(num_samples):
    high_res_image = scipy.misc.imread(os.path.join(data_path, high_res_images[i]), mode='L')
    pyramid = generate_gaussian_pyramid(
      im=high_res_image,
      downsample_factor=downsample_factor,
      sigma=sigma,
      levels=levels
    )
    high_res_image = pyramid[0]
    low_res_image = pyramid[-1]
    # Saving the low resolution images
    scipy.misc.imsave(os.path.join(low_res_image_path, 'low_' + high_res_images[i]), low_res_image)
    # Flattening images into a format that eigentransformation can use.
    high_res_image = high_res_image.flatten()
    low_res_image = low_res_image.flatten()
    # According to eigentransformpaper, we could/need to add zero mean Gaussian Noise with relatively small std such as 0.03 and 0.05
    #low_res_image += np.random.normal(0, 0.05, len(low_res_image))
    high_res_data[:, i] = high_res_image
    low_res_data[:, i] = low_res_image
    print ("Finished processing image " + str(i))

  # Saving training
  num_training_samples = round(num_samples * (1.0 - split_factor))

  training_low_res_data = low_res_data[:, :num_training_samples]
  training_high_res_data = high_res_data[:, :num_training_samples]
  testing_low_res_data = low_res_data[:, num_training_samples:]
  testing_high_res_data = high_res_data[:, num_training_samples:]
  # Saving testing
  np.save(low_res_training_path, training_low_res_data)
  print("Saving low resolution training data")
  np.save(high_res_training_path, training_high_res_data)
  print("Saving high resolution training data")
  np.save(low_res_testing_path, testing_low_res_data)
  print("Saving low resolution testing data")
  np.save(high_res_testing_path, testing_high_res_data)
  print("Saving high resolution training data")
  print("Done")

# Downsample a numpy ndarray image by an integer factor
def down_sample(im, factor):
  width, height = im.shape
  if (factor > width or factor > height):
    raise ValueError("Subsample factor is bigger than image dimensions")

  im = im[::factor, ::factor]
  return im

# Normalize the pixel values from [0, 255] to [0, 1]
def normalize(im):
  return im / 255

# Generate a guassian pyramid and returns it as a list of images.
def generate_gaussian_pyramid(im, downsample_factor, sigma, levels):
  if (levels < 1):
    raise ValueError("Pyramid should have more than 1 level")

  width, height = im.shape
  im = deepcopy(normalize(im))
  gaussian_pyramid = [im]

  for level in range(1, levels):
    im = deepcopy(im)
    im = ndimage.filters.gaussian_filter(im, sigma)
    im = down_sample(im, downsample_factor)
    gaussian_pyramid.append(im)

  return gaussian_pyramid

IMAGE_PATH = '../aligned_data/profile_pictures_aligned/'
TRAINING_HIGH_RES_PATH = 'training/training_high_res'
TRAINING_LOW_RES_PATH = 'training/training_low_res'
TESTING_HIGH_RES_PATH = 'testing/testing_high_res.npy'
TESTING_LOW_RES_PATH = 'testing/testing_low_res.npy'
LOW_RES_IMAGE_PATH = '../aligned_data/profile_pictures_aligned_low_res/'
SIGMA = 2 # Sigma for gaussian filter.
DOWNSAMPLE_FACTOR = 2 # Downsample factor for gaussian pyramid
PYRAMID_LEVELS = 3

# Run this to generate the training low resolution and high resolution images
create_data_from_images(
  data_path=IMAGE_PATH,
  low_res_image_path=LOW_RES_IMAGE_PATH,
  low_res_training_path=TRAINING_LOW_RES_PATH,
  high_res_training_path=TRAINING_HIGH_RES_PATH,
  low_res_testing_path=TESTING_LOW_RES_PATH,
  high_res_testing_path=TESTING_HIGH_RES_PATH,
  split_factor=0.1,
  sigma=SIGMA,
  downsample_factor=DOWNSAMPLE_FACTOR,
  levels=PYRAMID_LEVELS
)

# Testing individual image.
im = scipy.misc.imread("result/original/arvind4.png", mode='L')
pyramid = generate_gaussian_pyramid(im, 2, 2, 3)
scipy.misc.imsave('result/high_res/arvind_high_res4.png', pyramid[0])
scipy.misc.imsave('result/low_res/arvind_low_res4.png', pyramid[-1])
plt.imshow(pyramid[-1], cmap="gray")
plt.show()