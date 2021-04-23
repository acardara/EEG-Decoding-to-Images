import numpy as np

# load refactored image dataset containing 40 classes of 900(64x64x3) images. Output shape is (36000, 64, 64, 3)
# each image class is separated by 900 indices. class_0 = images[0:900], class_1 = images[900:1800], ...
def load_images():
  images = np.load("/content/data/images/images.npy")
  image_labels = np.load("/content/data/images/image_labels.npy")
  return images, image_labels
  
# flatten image data from (36000, 64, 64, 3) to (36000, 12288)
def flatten_images(data):
  return data.reshape((data.shape[0], 12288))

# reshape image data from (36000, 12288) to (36000, 64, 64, 3) 
def reshape_images(data):
  return data.reshape((data.shape[0],64,64,3))

"""
 Outdated helper functions
 
def data_initial_reshape(data):
  return data.reshape((data.shape[0],data.shape[1],3,64,64)).transpose((0,1,3,4,2))

def data_reshape_to_new(data):
  return data.reshape((data.shape[0],3,64,64)).transpose((0,2,3,1))

def data_flattened_to_3d(data):
  return data.reshape((data.shape[0],64,64,3))

def data_3d_to_1d(data):
  return data.reshape((data.shape[0], 12288))

"""
