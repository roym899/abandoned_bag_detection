from collections import OrderedDict, Counter
from typing import Tuple, Optional
import numpy as np
from scipy.spatial.distance import cdist

BAG_LABEL = 1
PERSON_LABEL = 0


def compute_center(bounding_boxes):
    # type: (np.ndarray) -> np.ndarray
    """
    Computes the bounding box centers given the bounding boxes.
    Args:
        bounding_boxes: numpy array containing two diagonal corner coordinates of the bounding boxes,
        shape [num_bounding_boxes, 4]
    Returns:
        centers_3d: numpy array of the bounding box centers, shape [num_bounding_boxes, 3]
    """
    x_dist = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    y_dist = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    centers = bounding_boxes[:, 0:2] + 0.5 * np.stack((x_dist, y_dist), axis=1)
    centers_3d = add_z_coordinate(centers)
    return centers_3d


def add_z_coordinate(centers):
    # type: (np.ndarray) -> np.ndarray
    """
    Adds a dummy 0. z-coordinate to the object centers.
    Could be replaced with some depth estimation algorithm in the future.
    Args:
        centers: 2d coordinates for the observed bounding box centers, shape [num_centers, 2]
    :return:
        3d coordinates for the observed bounding box centers, shape [num_centers, 2]
    """
    return np.concatenate((centers, np.zeros(shape=(centers.shape[0], 1))), axis=1)


def split_bag_persons(centers, labels):
    # type: (np.ndarray, np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    """
    Splits the centers array based on the label for each center. Labels considered are BAG_LABEL and PERSON_LABEL
    Args:
        centers: Array with bounding box centers, shape [num_centers, ...]
        labels: Array with bounding box labes, shape [num_centers]
    Returns:
        bag_bounding_centers: centers for the bags
        persons_bounding_centers: centers for the persons
    """
    assert isinstance(labels, np.ndarray)
    assert isinstance(centers, np.ndarray)
    print(labels)
    bag_bounding_centers = centers[labels == BAG_LABEL]
    persons_bounding_centers = centers[labels == PERSON_LABEL]

    return bag_bounding_centers, persons_bounding_centers


# def extract_bag_to_ppl_vectors(boudning_boxes, labels):
#     centers = compute_center(boudning_boxes)
#     bag_centers, persons_centers = split_bag_persons(centers, labels)
#     distances = cdist(bag_centers, persons_centers)
#     ind = distances.argmin(axis=1)
#     dist = distances[np.arange(len(ind)), ind]
#
#     return dist, ind


class SimpleTracker:
    """
    Location based tracker for persons and bags.
    Also does location based bag-person association.
    """

    def __init__(self, self_association_thres, bag_person_thres):
        # type: (float, float) -> None
        """
        Initialize tracker.

        Args:
            self_association_thres: threshold value for person-person and bag-bag association between frames in pixel
                units.
            bag_person_thres: threshold value for initial bag-person association.
        """
        self.all_centers = {'bags': OrderedDict(),
                            'persons': OrderedDict()}  # Stores seen bounding box centers by object id
        self.prev_frame_ids = {'bags': [],
                               'persons': []}  # Stores ids of objects observed in the last frame

        self.bag_person_association = dict()  # Maps bag id to bag owner id
        self.bag_person_dist = dict()  # Maps bag id to distance to the bag owner
        self.instance_count = {'bags': 0,
                               'persons': 0}  # Counts how many bags and persons have been seen
        self.bag_person_thres = bag_person_thres
        self.self_association_thres = self_association_thres
        self.prev_frame_kept = {'bags': False,
                                'persons': False}  # Tracks if last frame's bounding boxes were kept or ignored
        self.keep_frame_counter = {'bags': 0,
                                   'persons': 0}  # Counts how many frames back object centers have been stored

    def update(self, boxes, labels):
        # type: (np.ndarray, np.ndarray) -> None
        """
        Updates the trackers state given bounding box and class detections.
        Args:
            boxes: numpy array containing two diagonal corner coordinates of the bounding boxes,
                shape [num_bounding_boxes, 4]
            labels: Array with bounding box labes, shape [num_centers]
        """
        centers = compute_center(boxes)
        bag_bounding_centers, persons_bounding_centers = split_bag_persons(centers, labels)

        self.frame2frame_association(persons_bounding_centers, 'persons')
        self.frame2frame_association(bag_bounding_centers, 'bags')
        self.update_bag_person_association()

        print(self.prev_frame_ids)

    def is_unattended(self, bag_id):
        # type: (int) -> bool
        """
        Checks if a given bag misses an owner or  has it's owner at a distance larger that the bag_person_thres
            threshold.
        Args:
            bag_id:
        Returns:
            True if the bag does not have an owner or is too far away from its owner.
            False otherwise.
        """

        person_id = self.bag_person_association[bag_id]
        if person_id is None:
            return True
        person_center = self.all_centers['persons'][person_id]
        bag_center = self.all_centers['bags'][bag_id]

        if np.sqrt(((person_center - bag_center) ** 2).sum()) > self.bag_person_thres:
            return True

        return False

    def frame2frame_association(self, new_centers, tag):
        # type: (np.ndarray, str) -> None
        """
        Associates centers of 'persons' and 'bags' observed in the last frame with centers observed in the current
        frame.
        The association is done forward in time, i.e. we find the closest center in the new frame for each center
        observed in the previous frame.
        If two centers in the previous frame map to the same center in the new frame the closest center gets the
        association.
        In case some center in the new frame can't find a match in the old frame it is added as a new object with a
        new id.
        In case there were no observed objects in the previous frame or no observed objects in the current frame, the
        state is not updated.

        Args:
            new_centers: Array of bounding box centers detected in the current frame.
            tag: Either 'persons' or 'bags'.
        """
        frame_ids = []
        frame_centers = []
        new_frame_unused_centers = list(range(new_centers.shape[0]))
        if len(self.prev_frame_ids[tag]) > 0 and len(new_centers) > 0:
            prev_frame_centers = np.stack([self.all_centers[tag][id] for id in self.prev_frame_ids[tag]], axis=0)
            distances = cdist(prev_frame_centers, new_centers)

            cc_in_new_frame_index = distances.argmin(axis=1)
            new_frame_unused_centers = list(set(new_frame_unused_centers) - set(cc_in_new_frame_index.tolist()))

            min_dist = distances[range(len(self.prev_frame_ids[tag])), cc_in_new_frame_index]
            index_counter = Counter(cc_in_new_frame_index)

            for dist, prev_frame_id, new_center, index in zip(min_dist,
                                                              self.prev_frame_ids[tag],
                                                              new_centers[cc_in_new_frame_index],
                                                              cc_in_new_frame_index):

                if dist < self.self_association_thres and index_counter[index] <= 1:
                    # case where there is a unique closest center
                    self.all_centers[tag][prev_frame_id] = new_center
                    frame_ids.append(prev_frame_id)
                    frame_centers.append(new_center)
                elif dist > self.self_association_thres and index_counter[index] <= 1:
                    # case where the closest frame is too far away
                    self.all_centers[tag][self.instance_count[tag]] = new_center
                    frame_ids.append(self.instance_count[tag])
                    frame_centers.append(new_center)
                    self.instance_count[tag] += 1
                else:
                    # case where one new center is closest to several centers
                    other_dists = min_dist[cc_in_new_frame_index == index]
                    if dist <= other_dists.min():
                        self.all_centers[tag][prev_frame_id] = new_center
                        frame_ids.append(prev_frame_id)
                        frame_centers.append(new_center)

        # add the new centers which were not closest to any old center
        for new_center in new_centers[new_frame_unused_centers, :]:
            self.all_centers[tag][self.instance_count[tag]] = new_center
            frame_ids.append(self.instance_count[tag])
            frame_centers.append(new_center)
            self.instance_count[tag] += 1

        if frame_ids:
            self.prev_frame_ids[tag] = frame_ids
            self.prev_frame_kept[tag] = False
            self.keep_frame_counter[tag] = 0
        else:
            self.keep_frame_counter[tag] += 1
            if self.keep_frame_counter[tag] > 8:
                for id in self.prev_frame_ids[tag]:
                    self.all_centers[tag][id] = np.array([np.Inf, np.Inf, np.Inf])
                self.prev_frame_ids[tag] = []
                self.prev_frame_kept[tag] = False
            else:
                self.prev_frame_kept[tag] = True

        # print(frame_ids, self.prev_frame_ids[tag])
        # print(self.all_centers[tag])

    def update_bag_person_association(self):
        # type: () -> None
        """
        Iterates over all detected bags in the last frame (current frame) and updates the bag-person association and
        the bag person distance.
        """

        for bag_id in self.prev_frame_ids['bags']:
            if bag_id not in self.bag_person_association or self.bag_person_association[bag_id] is None:
                # Case were the bag has not previous owner
                person_id, dist = self.find_closest_person_to_bag(bag_id)
                self.bag_person_association[bag_id] = person_id
                self.bag_person_dist[bag_id] = dist
            elif self.bag_person_association[bag_id] not in self.prev_frame_ids['persons']:
                # Case were the bags owner as not observed in the current frame
                self.bag_person_dist[bag_id] = float('inf')
            else:
                # Case were both bag and owner were observed in the current frame
                bag_person_vector = (self.all_centers['persons'][self.bag_person_association[bag_id]] -
                                     self.all_centers['bags'][bag_id])
                self.bag_person_dist[bag_id] = np.sqrt(np.power(bag_person_vector, 2).sum())

    def find_closest_person_to_bag(self, bag_id):
        # type: (int) -> Tuple[Optional[int], float]
        """
        Checks for closest person in the current frame given an id of a detected bag.
        Returns the id of the person and the distance given that a person could be found with a distance below the
        bag_person_thres threshold.

        Args:
            bag_id: Id of a bag observed in the current frame.
        Returns:
            person_id: Id of the closest person or None if no person could be found with a distance smaller than
                bag_person_thres
            distance: Distance in pixels between the person and the bag. Inf if not person could be found.
        """
        bag_center = self.all_centers['bags'][bag_id]
        dists = []
        for person_id in self.prev_frame_ids['persons']:
            person_center = self.all_centers['persons'][person_id]
            dists.append(np.sqrt(np.power(person_center - bag_center, 2).sum()))
        if not self.prev_frame_ids['persons']:
            return None, float('inf')
        closest_person_ind = int(np.array(dists).argmin())
        if dists[closest_person_ind] < self.bag_person_thres:
            return self.prev_frame_ids['persons'][closest_person_ind], dists[closest_person_ind]
        else:
            return None, float('inf')
