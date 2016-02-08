import numpy as np
import pywt as wt
import cv2 as cv
import os
import csv


def get_wavelet_features(input_image):
    feature_array = []
    coefficients = wt.wavedec2(input_image, "haar", level=4)

    for level in reversed(coefficients[1:]):
        for details in level:
            feature_array.append((np.sum(details ** 2)) / np.size(input_image))

    return feature_array


def train(training_dir="./training/"):
    feature_vectors = []
    train_data = {}

    with open(training_dir + "train.txt") as training_file:
        read_file = csv.reader(training_file, delimiter=',')
        for row in read_file:
            if len(row) == 2:
                train_data[row[0]] = int(row[1])
            elif len(row) == 1:
                train_data["Classes"] = row[0]

    files = os.listdir(training_dir)
    for file in files:
        input_image = cv.imread(training_dir + file, cv.IMREAD_GRAYSCALE)

        if file != "train.txt" and file != "output.txt":
            if input_image is None:
                raise ValueError("File \"" + file + "\" in the training directory is not a valid image")
            else:
                feature_vectors.append([train_data.get(file)] + get_wavelet_features(input_image))
    with open(training_dir + "output.txt", 'a') as output:
        output.write(train_data["Classes"] + '\n')
        csv_writer = csv.writer(output, lineterminator='\n')
        csv_writer.writerows(feature_vectors)
