
import numpy as np


def split_bag_persons(boudning_boxes, labels):

    bag_bounding_boxes = boudning_boxes[labels == 'bag']
    persons_bounding_boxes = boudning_boxes[labels == 'person']

    return bag_bounding_boxes, persons_bounding_boxes


def compute_center(bounding_boxes):
    x_dist = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    y_dist = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    centers = bounding_boxes[:, 0:2] + 0.5 * np.stack(x_dist, y_dist, axis=1)
    return centers

def extract_bag_to_ppl_vectors(bag_bounding_boxes, persons_bounding_boxes):

    centers =