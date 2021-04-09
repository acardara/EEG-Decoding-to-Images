import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import torch
import collections


def label2class_mapper():
    torch_file = torch.load("eeg_14_70.pth")
    torch_labels = torch_file["labels"]

    mapping = {}
    for index, label in enumerate(torch_labels):
        mapping[index] = label

    return mapping


def label2class64_mapper(label2class):
    mapping = {}
    with open('Imagenet64_map.txt') as f:
        datafile = f.readlines()
        for key in label2class:
            value = label2class[key]

            for line in datafile:
                if line.startswith(value):
                    class64 = line.split(" ")[1]
                    mapping[class64] = key
    return mapping


def label2image64_mapper(label2image64, image_path):
    mapping = {}
    with open(image_path, 'rb') as fo:
        dict = pickle.load(fo)

        images = dict['data']
        labels = dict['labels']

        for index, label in enumerate(labels):
            if str(label) in label2image64:
                if label2image64[str(label)] not in mapping:
                    mapping[label2image64[str(label)]] = [images[index]]
                else:
                    mapping[label2image64[str(label)]].append(images[index])

        return mapping


def merge_dictionaries(dict1, dict2):
    print(len(dict1[0]))

    for label in dict2:
        images = dict2[label]
        for image in images:
            dict1[label].append(image)

    print(len(dict1[0]))


def map_images(dir):

    mapping = {}
    # map eeg labels to ImageNet class id {label : class}
    label2class = label2class_mapper()

    # map eeg labels to ImageNet64 class id {class64 : label}
    label2class64 = label2class64_mapper(label2class)

    for file in os.listdir(dir):
        path = os.path.join(dir, file)

        # map eeg labels to ImageNet64 images
        label2image64 = label2image64_mapper(label2class64, path)

        if len(mapping) == 0:
            mapping = label2image64
        else:
            merge_dictionaries(mapping, label2image64)

    return mapping


def array_to_image(img):
    return img.reshape(3, 64, 64).transpose(1, 2, 0) / np.float32(255)


def save_images(image_mapping_array, label_number):
    images = image_mapping_array[label_number]

    # Loop through all images for a specific label and save them
    for index, image in enumerate(images):
        img = array_to_image(image)
        plt.imsave("images/" + str(index) + ".png", img, format="png")


def load_mapping(path, as_dict=False):
    # Load image mapping file
    label2image64 = np.load(path, allow_pickle=True)
    if as_dict:
        return label2image64[()]
    else:
        # Create a numpy ready array of image mappings [[class 0 images], [class 1 images], ... , [class 39 images]]
        return list(collections.OrderedDict(sorted(label2image64[()].items())).values())


# # Create image mapping file
# mapping = map_images("imagenet_images")
#
# # Write list to file
# np.save("image_mapping.npy", mapping)

mapping = load_mapping("image_mapping.npy", as_dict=True)
save_images(mapping, 3)
