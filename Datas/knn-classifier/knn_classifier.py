from random import uniform
import numpy as np
import cv2
import os
import time

from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

absPATH = '/home/pi/Datas'
DEMO = 'knn-classifier'


class ColorClassifier():
    upper_hsv = [220, 255, 255]

    def __init__(self, demo_name="", path=absPATH):
        self.knn = KNeighborsClassifier()
        self.demo_name = demo_name
        self.label_names = []
        self.feature_names = ['h', 's', 'v']
        self.label_keys = []
        self.label_num = len(self.label_names)
        self.feature_num = len(self.feature_names)
        self.labels = []
        self.features = []
        self.data_cnt = {}
        self.divided_labels = []
        self.divided_features = []
        self.data_file_name = self.demo_name + "_KNN_data"
        self.current_image = ''
        self.current_hsv_value = ''
        self.current_label = ''
        self.reactions = []
        self.predicts = None
        self.label_cnt = -1
        self.absPATH = path
        self.__create_base_folders()

    def load_model(self, name):
        self.demo_name = name
        self.set_values_from_data()

    def read_datas(self):
        while True:
            self.demo_name = input("Please enter the project name you want to run: ")

            if os.path.isdir(os.path.join(self.absPATH, DEMO, self.demo_name)):
                print("Okay, start ", self.demo_name)
                self.set_values_from_data()
                break
            else:
                print("This project does not exist.")

    def set_values(self):
        if self.demo_name == "":
            self.demo_name = input("What is the name of your project? : ")

        if os.path.isdir(os.path.join(self.absPATH, DEMO, self.demo_name)):
            if "y" == input("This project already exists. Do you want to test this model? (y/n) : "):
                self.set_values_from_data()
                return False
            else:
                self.demo_name = ""
                return self.set_values()
        else:
            self.data_file_name = self.demo_name + "_KNN_data"
            while True:
                try:
                    self.label_num = int(input("What is the total number of labels? : "))
                except ValueError:
                    print("Oops! Input was not an integer.")
                else:
                    break

            for i in range(self.label_num):
                self.label_names.append(input("Label name (" + str(i + 1) + "/" + str(self.label_num) + ") : "))
                key = 'n'
                while key == 'n':
                    key = input(
                        "Keyboard command for label (" + str(i + 1) + "/" + str(self.label_num) + ") : ")
                self.label_keys.append(key)

            return True

    def set_values_from_data(self):
        self.data_file_name = self.demo_name + "_KNN_data"
        file_path = os.path.join(self.absPATH, DEMO, self.demo_name, str(self.data_file_name + ".txt"))

        current_label = ""
        with open(file_path) as f:
            for line in f.readlines():
                line = line.split(' ')
                if line[0] == "Labels":
                    for cursor in line[2:-1]:
                        self.label_names.append(cursor)
                    self.label_num = len(self.label_names)

                elif line[0] == "Features":
                    for cursor in line[2:-1]:
                        self.feature_names.append(cursor)
                    self.feature_num = len(self.feature_names)

                elif line[0] in self.label_names:
                    current_label = line[0]

                elif line[0] == ">":
                    feature = []
                    for cursor in line[1:-1]:
                        feature.append(int(cursor))
                    self.labels.append(current_label)
                    self.features.append(feature)

    def check_enough_datas(self):
        print('\nYou took .......')
        for label in self.data_cnt.keys():
            print(label + ':' + str(self.data_cnt[label]))
        print('datas ..........\n')

        if len(self.data_cnt) < self.label_num:
            print('There is not enough data. Try again')
            return False

        for num in self.data_cnt.values():
            if num < 10:
                print('There is not enough data. Try again')
                return False
        return True

    def add_data(self, label, feature):
        self.current_image = []

        if not self.is_in_labels(label):
            return
        label = self.label_names[self.label_keys.index(label)]
        self.current_label = label
        self.label_cnt += 1

        if not isinstance(feature, list):
            feature = self.get_hsv_data(feature)

            if feature[0] < 40:
                temp = [180 + feature[0], feature[1], feature[2]]
                self.features.append(temp)
            elif feature[0] > 180:
                temp = [0 + feature[0], feature[1], feature[2]]
                self.features.append(temp)

        self.labels.append(label)
        self.features.append(feature)
        if label in self.data_cnt.keys():
            self.data_cnt[label] += 1
        else:
            self.data_cnt[label] = 1

    def add_datas(self, camera, save_image =False):
        while True:
            label = input("Please enter the keyboard command for the label")

            if label == "q":
                break
            if self.is_in_labels(label):
                if label in self.label_names:
                    label = self.label_keys[self.label_names.index(label)]

                for i in range(10):
                    image = camera.capture()
                    self.add_data(label, image)
                    if save_image:
                        self.save_image(label, image)
                    time.sleep(0.1)

        if not self.check_enough_datas():
            return self.add_datas(camera)

    def save_image(self):
        if len(self.current_image)!=0:
            file_name = os.path.join(self.absPATH, DEMO, self.demo_name, "images", str(self.current_label + "_" + str(
                self.label_cnt) + "" + '_'.join(map(str, self.current_hsv_value)) + ".jpg"))
            cv2.imwrite(file_name, self.current_image)

    def save_data_set(self):
        self.__create_folders()
        self.remove_outlier()

        f = open(os.path.join(self.absPATH, DEMO, self.demo_name, str(self.data_file_name + ".txt")), 'w')
        f.write('[' + self.data_file_name + ']\n')
        f.write("Labels : " + " ".join(self.label_names) + " \n")
        f.write(" \n")
        f.write("Features : " + " ".join(self.feature_names) + " \n")

        current_label_name = ""
        for i in range(len(self.labels)):
            if current_label_name != self.labels[i]:
                f.write(self.labels[i] + " ---------- \n")
                current_label_name = self.labels[i]
            f.write("> " + ' '.join(map(str, self.features[i])) + ' \n')

        f.close()

    def get_hsv_data(self, image):
        image = cv2.flip(image, -1)
        height, width, channel = image.shape

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, s, v = cv2.split(hsv)

        top = int(height / 2 - 2 * height / 10)
        bottom = int(height / 2)
        left = int(width / 2 - width / 10)
        right = int(width / 2 + width / 10)

        h = h[top:bottom, left:right]
        s = s[top:bottom, left:right]
        v = v[top:bottom, left:right]

        cv2.rectangle(rgb, (left, top), (right, bottom), (255, 0, 0), 2)

        mean_h = int(np.mean(h)) + 20
        mean_s = int(np.mean(s))
        mean_v = int(np.mean(v))

        print(mean_h, mean_s, mean_v)
        self.current_hsv_value = [mean_h, mean_s, mean_v]
        self.current_image = rgb

        return self.current_hsv_value

    def get_accuracy(self):
        x, y = shuffle(self.features, self.labels, random_state=0)

        cut = int(len(x) / 10)
        if cut == 0:
            print("There is not enough data. Please add more.")
            return

        x_train = x[:-cut]
        y_train = y[:-cut]
        x_test = x[-cut:]
        y_test = y[-cut:]

        self.knn.fit(x_train, y_train)

        print(self.knn.predict(x_test))
        print(y_test)

        n = 0
        for i in range(0, len(self.knn.predict(x_test))):
            if self.knn.predict(x_test)[i] == y_test[i]:
                n += 1
        accuracy = (n / len(y_test)) * 100

        print("Accuracy : " + str(accuracy))

        return accuracy

    def predict(self, features):
        if not isinstance(features, list):
            features = self.get_hsv_data(features)

        pred_list = []
        pred_list.append(features)

        self.knn.fit(self.features, self.labels)
        self.predicts = self.knn.predict(pred_list)

        return self.predicts

    def divide_data(self):
        self.labels, self.features = zip(*sorted(zip(self.labels, self.features)))

        divided_labels = [None] * self.label_num
        divided_features = [None] * self.label_num

        current_label_name = self.labels[0]
        prev_idx = 0

        for i in range(1, len(self.labels)):
            if current_label_name != self.labels[i] or i == len(self.labels) - 1:
                idx = self.label_names.index(self.labels[i - 1])
                if i == len(self.labels) - 1:
                    i = len(self.labels)
                else:
                    current_label_name = self.labels[i]
                divided_labels[idx] = self.labels[prev_idx:i]
                divided_features[idx] = self.features[prev_idx: i]
                prev_idx = i

        self.divided_labels = divided_labels
        self.divided_features = divided_features

    def remove_outlier(self):
        self.divide_data()

        divided_features = []
        divided_labels = []
        for i in range(self.label_num):
            features = list(self.divided_features[i])
            labels = list(self.divided_labels[i])
            length = len(labels)
            features = features[int(length/5):int(length-length/5)]
            means = np.array(np.mean(features, axis=0))
            stds = np.array(np.std(features, axis=0))

            for j in range(length - len(features)):
                random_val = uniform(-2, 2)
                stds = np.array(random_val * stds)
                feature = []
                for k in range(self.feature_num):
                    temp = int(means[k] + stds[k])
                    if temp > ColorClassifier.upper_hsv[k]:
                        temp = ColorClassifier.upper_hsv[k]
                    elif temp < 0:
                        temp = 0
                    feature.append(temp)

                features.append(feature)
            divided_features.append(features)
            divided_labels.append(labels)

        self.divided_features = divided_features
        self.divided_labels = divided_labels

        sorted_labels = []
        sorted_features = []

        for i in range(self.label_num):
            sorted_labels += self.divided_labels[i]
            sorted_features += self.divided_features[i]

        self.labels = sorted_labels
        self.features = sorted_features

    def is_in_labels(self, label):
        if label in self.label_names or label in self.label_keys:
            return True
        print("There's no " + label + " label.")
        return False

    def __create_folders(self):
        try:
            if not (os.path.isdir(os.path.join(self.absPATH, DEMO, self.demo_name))):
                os.makedirs(os.path.join(self.absPATH, DEMO, self.demo_name))

            if not (os.path.isdir(os.path.join(self.absPATH, DEMO, self.demo_name, 'images'))):
                os.makedirs(os.path.join(self.absPATH, DEMO, self.demo_name, 'images'))

        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create path.")
                raise

    def __create_base_folders(self):
        try:
            if not (os.path.isdir(self.absPATH)):
                os.makedirs(self.absPATH)

            if not (os.path.isdir(os.path.join(self.absPATH, DEMO))):
                os.makedirs(os.path.join(self.absPATH, DEMO))

        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create path.")
                raise
