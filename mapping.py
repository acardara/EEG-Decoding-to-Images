import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import torch
import collections
import requests
import time
import shutil
from os import listdir
from os.path import isfile, join

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


def find_id_urls(eeg_path, url_path):
    torch_file = torch.load(eeg_path)
    ids = torch_file["images"]
    big_map = {}
    small_map = {}

    with open(url_path, 'rb') as f:
        lines = f.readlines()

        for line in lines:
            try:
                line = line.decode("utf-8")
                line = line.split("\t")
                big_map[line[0]] = line[1]
            except Exception:
                print(line)

    for id in ids:
        small_map[id] = "None"
        if id in big_map:
            small_map[id] = big_map[id]

    print(small_map)
    outF = open("url_map.txt", "w")
    for k in small_map.keys():
        url = small_map[k]

        # write line to output file
        outF.write(k)
        outF.write(' ')
        outF.write(url)
        if url == "None":
            outF.write("\n")

    outF.close()


def find_missing_urls():
    count = 0
    none_count = 0
    with open("url_map.txt", 'rb') as f:
        lines = f.readlines()

        for line in lines:
            count += 1
            line = line.decode("utf-8")
            line = line.split(" ")
            print(line)
            if line[1].startswith("None"):
                none_count += 1
    print(count, none_count)
# test("eeg_5_95_std.pth", "fall11_urls.txt")


def find_missing_image():
    broken_links = []
    count = 0
    with open("missing_images.txt", 'rb') as f:
        lines = f.readlines()

        for line in lines:
            print(count)
            count += 1

            line = line.decode("utf-8")
            line = line.split(" ")
            if len(line) > 1:

                print(line)
                id = line[0]
                url = line[1][:-1]
                if url.startswith("None"):
                    broken_links.append(id)
                else:
                    try:
                        time.sleep(0.3)
                        if requests.get(url, timeout=5).status_code != 200:
                            broken_links.append(id)
                    except Exception:
                        broken_links.append(id)
                        pass
            else:
                broken_links.append(line[0])

    with open("test.txt", "w") as f:
        for b in broken_links:
            f.write(b)


# find_missing_image()

def test():
    bad_urls = []
    with open("test.txt", 'rb') as f:
        lines = f.readlines()
        for line in lines:
            if line == lines[65]:
                line = lines[65]
                line = line.decode("utf-8")
                line = ['n'+e for e in line.split('n') if e]
                for s in line:
                    bad_urls.append(s)

            else:
                line = line.decode("utf-8")[:-1]
                bad_urls.append(line)

    with open("bad_urls.txt", "w") as f:
        for b in bad_urls:
            f.write(b)
            f.write("\n")


def download_images():
    bad_ids= []
    with open("bad_urls.txt", 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode("utf-8")[:-1]
            bad_ids.append(line)

    all_urls = []
    with open("url_map.txt", 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode("utf-8")[:-1]
            all_urls.append(line)

    print(all_urls)
    print(bad_ids)

    for url in all_urls:
        line = url.split(" ")
        id = line[0]
        url = line[1]
        if id not in bad_ids:
            time.sleep(0.2)
            # Open the url image, set stream to True, this will return the stream content.
            r = requests.get(url, stream=True)

            # Check if the image was retrieved successfully
            if r.status_code == 200:
                # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                r.raw.decode_content = True

                # Open a local file with wb ( write binary ) permission.
                with open("images/"+id+".jpg", 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
    #     print(url,id)
    # print(bad_urls)
    # print(all_urls)

def download_images_cont():
    bad_ids= []
    with open("bad_urls.txt", 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode("utf-8")[:-1]
            bad_ids.append(line)

    all_urls = []
    with open("url_map.txt", 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode("utf-8")[:-1]
            all_urls.append(line)

    print(all_urls)
    print(bad_ids)
    print(all_urls[1576])

    for url in all_urls[1878:]:
        line = url.split(" ")
        id = line[0]
        url = line[1]
        if id not in bad_ids:
            time.sleep(0.2)
            # Open the url image, set stream to True, this will return the stream content.
            r = requests.get(url, stream=True)

            # Check if the image was retrieved successfully
            if r.status_code == 200:
                # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                r.raw.decode_content = True

                # Open a local file with wb ( write binary ) permission.
                with open("images/"+id+".jpg", 'wb') as f:
                    shutil.copyfileobj(r.raw, f)


def final_scan_for_missing_images():

    all_urls = []
    with open("url_map.txt", 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode("utf-8")[:-1]
            all_urls.append(line)

    images = [f for f in listdir("images")]
    all_ids = []
    for image in images:
        id = image.split(".")[0]
        all_ids.append(id)

    missing = []
    have = []
    for url in all_urls:
        id = url.split(" ")[0]
        if id not in all_ids:
            missing.append(id)
        else:
            have.append(id)
    print(missing)
    print(have)
    print(len(missing), len(have), (len(missing)+len(have)))
    with open("missing_ids.txt", "w") as f:
        for b in missing:
            f.write(b)
            f.write("\n")
            

final_scan_for_missing_images()


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

# mapping = load_mapping("image_mapping.npy", as_dict=True)
# save_images(mapping, 3)
