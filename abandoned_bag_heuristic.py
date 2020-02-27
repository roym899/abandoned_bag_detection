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
    return dist, ind


