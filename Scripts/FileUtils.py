import cv2
import xml.etree.ElementTree as et
from Settings import Settings


class FileUtils:
    def __init__(self, limit=None):
        print("Start to load data")
        self.image_names = self.load_image_names()
        if limit:
            self.image_names = self.image_names[:limit]
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
        print(len(images), end="!\n")
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
        print(len(annotations), end="!\n")
        return annotations

    def filter_by_class(self, class_name):
        """call this method to filter by class name"""
        indice = list()
        for i, annotation in enumerate(self.annotations):
            for obj in annotation:
                if obj["name"] == class_name:
                    indice.append(i)
                    break

        def filter_by_index(indice, sequence):
            new_list = list()
            for i, item in enumerate(sequence):
                if i in indice:
                    new_list.append(item)
            return new_list

        self.image_names = filter_by_index(indice, self.image_names)
        self.images = filter_by_index(indice, self.images)
        self.annotations = filter_by_index(indice, self.annotations)

        for i, annotation in enumerate(self.annotations):
            li = list()
            for obj in annotation:
                if obj["name"] == class_name:
                    li.append(obj)
            self.annotations[i] = li
            del annotation












