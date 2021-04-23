import numpy as np

def load_images():
  images = np.load("/content/data/images/images.npy")
  image_labels = np.load("/content/data/images/image_labels.npy")
  return images, image_labels
  
def flatten_images(data):
  return data.reshape((data.shape[0], 12288))

def reshape_images(data):
  return data.reshape((data.shape[0],64,64,3))
