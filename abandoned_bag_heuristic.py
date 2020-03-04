from collections import OrderedDict, Counter
import numpy as np
from scipy.spatial.distance import cdist

BAG_LABEL = 1
PERSON_LABEL = 0


def split_bag_persons(centers, labels):
    assert isinstance(labels, np.ndarray)
    assert isinstance(centers, np.ndarray)
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
    ind = distances.argmin(axis=1)
    dist = distances[np.arange(len(ind)), ind]
    # vectors = persons_centers[ind, :] - bag_centers

    return dist, ind


def array_ind_to_key(ordered_dict, ind):
    return ordered_dict.keys()[ind]


class SimpleTracker:

    def __init__(self, self_association_thres, bag_person_thres):
        self.all_centers = {'bags': OrderedDict(),
                            'persons': OrderedDict()}
        self.prev_frame_centers = {'bags': np.full(shape=(1, 3), fill_value=np.inf),
                                   'persons': np.full(shape=(1, 3), fill_value=np.inf)}
        self.prev_frame_ids = {'bags': [],
                               'persons': []}

        self.bag_person_association = dict()
        self.bag_person_dist = dict()
        self.instance_count = {'bags': 0,
                               'persons': 0}
        self.bag_person_thres = bag_person_thres
        self.self_association_thres = self_association_thres

    def frame2frame_association(self, new_centers, tag):
        frame_ids = []
        new_frame_unused_centers = list(range(new_centers.shape[0]))
        if len(self.prev_frame_ids[tag]) > 0 and len(new_centers) > 0:
            distances = cdist(self.prev_frame_centers[tag], new_centers)

            new_frame_closest_centers = distances.argmin(axis=1)
            new_frame_unused_centers = list(set(new_frame_unused_centers) - set(new_frame_closest_centers.tolist()))

            min_dist = distances[range(len(self.prev_frame_ids[tag])), new_frame_closest_centers]

            index_counter = Counter(new_frame_closest_centers)

            for dist, prev_frame_id, new_center, index in zip(min_dist,
                                                              self.prev_frame_ids[tag],
                                                              new_centers[new_frame_closest_centers],
                                                              new_frame_closest_centers):

                if dist < self.self_association_thres and index_counter[index] <= 1:
                    # case where there is a unique closest center
                    self.all_centers[tag][prev_frame_id] = new_center
                    frame_ids.append(prev_frame_id)
                elif dist > self.self_association_thres and index_counter[index] <= 1:
                    # case where the closest frame is too far away
                    self.all_centers[tag][self.instance_count[tag]] = new_center
                    frame_ids.append(self.instance_count[tag])
                    # print('create new nothing close', self.instance_count[tag])
                    self.instance_count[tag] += 1
                else:
                    # case where one new center is closest to several centers
                    other_dists = min_dist[new_frame_closest_centers == index]
                    if dist <= other_dists.min():
                        self.all_centers[tag][prev_frame_id] = new_center
                        frame_ids.append(prev_frame_id)

        # add the new centers which were not closest to any old center
        for new_center in new_centers[new_frame_unused_centers, :]:
            self.all_centers[tag][self.instance_count[tag]] = new_center
            frame_ids.append(self.instance_count[tag])
            # print('create new', self.instance_count[tag])
            self.instance_count[tag] += 1

        if frame_ids:
            self.prev_frame_centers[tag] = new_centers
            self.prev_frame_ids[tag] = np.array(frame_ids)
        else:
            pass
            # self.prev_frame_ids[tag] = np.zeros((0,))

        print(frame_ids, self.prev_frame_ids[tag])
        print(self.all_centers[tag])

    def update_bag_person_association(self):
        for bag_id in self.prev_frame_ids['bags']:
            if bag_id not in self.bag_person_association or self.bag_person_association[bag_id] is None:
                person_id, dist = self.find_bag_owner(bag_id)
                self.bag_person_association[bag_id] = person_id
                self.bag_person_dist[bag_id] = dist
            elif self.bag_person_association[bag_id] not in self.prev_frame_ids['persons']:
                self.bag_person_dist[bag_id] = float('inf')
            else:
                bag_person_vector = (self.all_centers['persons'][self.bag_person_association[bag_id]] -
                                     self.all_centers['bags'][bag_id])
                self.bag_person_dist[bag_id] = np.sqrt(np.power(bag_person_vector, 2).sum())
                print(bag_id, self.bag_person_dist[bag_id], self.bag_person_association[bag_id],
                      self.all_centers['persons'][self.bag_person_association[bag_id]])

    def find_bag_owner(self, bag_id):
        bag_center = self.all_centers['bags'][bag_id]
        dists = []
        for person_id in self.prev_frame_ids['persons']:
            person_center = self.all_centers['persons'][person_id]
            dists.append(np.sqrt(np.power(person_center - bag_center, 2).sum()))
        closest_person_ind = int(np.array(dists).argmin())
        if dists[closest_person_ind] < self.bag_person_thres:
            return self.prev_frame_ids['persons'][closest_person_ind], dists[closest_person_ind]
        else:
            return None, float('inf')

    def update(self, boxes, labels):
        centers = compute_center(boxes)
        bag_bounding_centers, persons_bounding_centers = split_bag_persons(centers, labels)

        self.frame2frame_association(persons_bounding_centers, 'persons')
        self.frame2frame_association(bag_bounding_centers, 'bags')
        self.update_bag_person_association()

    def is_unattended(self):
        distances = np.array([self.bag_person_dist[bag_id] for bag_id in self.prev_frame_ids['bags']])
        return distances > self.bag_person_thres
