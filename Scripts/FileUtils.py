import numpy as np
import cv2
import xml.etree.ElementTree as et
from Settings import Settings


class FileUtils:
    def __init__(self):
        print("Start to load data")
        self.image_names = self.load_image_names()
        self.images = self.obtain_images(self.image_names)
        self.annotations = self.gain_annotations(self.image_names)
        print("End")

    @staticmethod
    def load_image_names(dataset=Settings.dataset, path=Settings.path_dataset):
        file_path = path + '/ImageSets/Main/' + dataset + '.txt'
        with open(file_path) as f:
            lines = f.readlines()
            image_names = [line.split(None, 1)[0].strip() for line in lines]
        return image_names

    @staticmethod
    def obtain_images(names, path=Settings.path_dataset):
        images = list()
        print("Start to load images: ", end="")
        for name in names:
            image_path = path + '/JPEGImages/' + name + '.jpg'
            p = cv2.imread(image_path)
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)  # generate RGB image
            images.append(p)
            if len(images) % 1000 == 0:
                print(len(images), end="=>")
        print(len(images), end="!")
        return images

    @staticmethod
    def obtain_ground_truth(image_name, path=Settings.path_dataset):
        file = path + '/Annotations/' + image_name + '.xml'
        with open(file, 'r') as f:
            xml_contains = f.read()
            root = et.XML(xml_contains)
            objects = list()
            for child in root:
                if child.tag == 'object':
                    obj = dict()
                    for child2 in child:
                        if child2.tag == 'name':
                            obj[child2.tag] = child2.text
                        elif child2.tag == 'bndbox':
                            for child3 in child2:
                                obj[child3.tag] = int(child3.text)
                    objects.append(obj)
            return objects

    def gain_annotations(self, names, path=Settings.path_dataset):
        annotations = list()
        print("Start to load annotations: ", end="")
        for name in names:
            annotations.append(self.obtain_ground_truth(name, path))
            if len(annotations) % 1000 == 0:
                print(len(annotations), end="=>")
        print(len(annotations), end="!")
        return annotations


