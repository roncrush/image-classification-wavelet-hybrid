import numpy as np
import cv2 as cv
import os
import csv
import train
from collections import Counter


def classify(class_names, classify_dir="./classification/", training_dir="./training/"):
    model_data = []
    classify_images = os.listdir(classify_dir)

    if not isinstance(class_names, dict):
        raise TypeError("class_names must be a list")

    with open(training_dir + "output.txt") as model_file:
        reader = csv.reader(model_file)
        for row in reader:
            if len(row) == 1:
                if not int(row[0]) == len(class_names):
                    if int(row[0]) > len(class_names):
                        raise Exception("There are not enough class names")
                    else:
                        raise Exception("There are more class names than needed")
            else:
                model_data.append(row)

    model_data = np.rot90(model_data, 3)

    files = os.listdir(classify_dir)

    for image in files:
        input_image = cv.imread(classify_dir + image, cv.IMREAD_GRAYSCALE)
        input_wavelets = train.get_wavelet_features(input_image)

        decisions = []

        for index, row in enumerate(model_data[1:]):
            #print(input_wavelets[index])
            #print(row)

            closest = min(row, key=lambda x: abs(float(x) - input_wavelets[index]))

            #print("Closest number is: " + closest)
            decisions.append(int(model_data[0][list(row).index(closest)]))

        common = Counter(decisions)
        text = class_names[common.most_common(1)[0][0]]

        cv.imshow("Input", input_image)
        class_mat = np.zeros((120, 400, 3), np.uint8)
        cv.putText(class_mat, text, (20, 80), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Output", class_mat)
        cv.waitKey(0)
