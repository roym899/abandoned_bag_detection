from collections import OrderedDict

import numpy as np
from scipy.spatial.distance import cdist

BAG_LABEL = 28
PERSON_LABEL = 0


def split_bag_persons(centers, labels):
    bag_bounding_centers = centers[labels == BAG_LABEL]
    persons_bounding_centers = centers[labels == PERSON_LABEL]

    return bag_bounding_centers, persons_bounding_centers


def compute_center(bounding_boxes):
    x_dist = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    y_dist = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    centers = bounding_boxes[:, 0:2] + 0.5 * np.stack((x_dist, y_dist), axis=1)
    centers_3d = add_z_coordinate(centers)
    return centers_3d


def add_z_coordinate(centers):
    return np.concatenate((centers, np.zeros(shape=(centers.shape[0], 1))), axis=1)


def extract_bag_to_ppl_vectors(boudning_boxes, labels):
    centers = compute_center(boudning_boxes)
    bag_centers, persons_centers = split_bag_persons(centers, labels)
    distances = cdist(bag_centers, persons_centers)
    ind = distances.argrmin(axis=1)
    dist = distances[np.arange(len(ind)), ind]
    vectors = persons_centers[ind, :] - bag_centers

    return dist, ind


def array_ind_to_key(ordered_dict, ind):
    return ordered_dict.keys()[ind]


class SimpleDetector:

    def __init__(self, im_height, im_width, self_association_thres, bag_person_thres):
        self.bag_centers = OrderedDict()
        self.persons_centers = OrderedDict()
        self.bag_person_association = dict()
        self.im_height = im_height
        self.im_width = im_width
        self.bag_count = 0
        self.persons_count = 0
        self.bag_person_thres = bag_person_thres
        self.self_association_thres = self_association_thres

    def bag_dict_to_array(self):
        return np.array([self.bag_centers[bag_id] for bag_id in self.bag_centers.keys()])

    def person_dict_to_array(self):
        return np.array([self.bag_centers[bag_id] for bag_id in self.bag_centers.keys()])

    def update(self, boxes, labels):
        centers = compute_center(boxes)
        bag_bounding_centers, persons_bounding_centers = split_bag_persons(centers, labels)

        persons_array = self.person_dict_to_array()

        persons_distances = cdist(persons_bounding_centers, persons_array)
        persons_association_ind = persons_distances.argmin(axis=1)
        persons_min_dist = persons_distances[np.arange(len(persons_association_ind)), persons_association_ind]
        for persons_dist, persons_ind, input_persons_center in zip(persons_min_dist, persons_association_ind,
                                                                   persons_bounding_centers):
            if persons_dist > self.self_association_thres:
                self.persons_centers[self.persons_count] = input_persons_center
                self.persons_count += 1
            else:
                self.persons_centers[list(self.persons_centers.keys())[persons_ind]] = input_persons_center

        persons_array = self.person_dict_to_array()
        bag_array = self.bag_dict_to_array()
        bag_distances = cdist(bag_bounding_centers, bag_array)
        bag_association_ind = bag_distances.argmin(axis=1)
        bag_min_dist = bag_distances[np.arange(len(bag_association_ind)), bag_association_ind]
        for bag_dist, bag_ind, input_bag_center in zip(bag_min_dist, bag_association_ind, bag_bounding_centers):
            if bag_dist > self.self_association_thres:
                self.bag_centers[self.bag_count] = input_bag_center
                distances = cdist(input_bag_center, persons_array)
                closest_person_ind = distances.argmin(axis=1)
                self.bag_person_association[self.bag_count] = list(self.persons_centers.keys())[closest_person_ind]
                self.bag_count += 1
            else:
                self.bag_centers[list(self.bag_centers.keys())[bag_ind]] = input_bag_center

        bag_array = self.bag_dict_to_array()
        person_ind = [ind for bag_id, ind in self.bag_person_association.items()]
        return bag_array, persons_array[person_ind, :], person_ind
