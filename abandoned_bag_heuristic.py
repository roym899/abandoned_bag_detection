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

    return dist, ind


def array_ind_to_key(ordered_dict, ind):
    return ordered_dict.keys()[ind]


class SimpleTracker:

    def __init__(self, self_association_thres, bag_person_thres):
        self.all_centers = {'bags': OrderedDict(),
                            'persons': OrderedDict()}
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
                    # print('create new nothing close', self.instance_count[tag])
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

        # print(frame_ids, self.prev_frame_ids[tag])
        # print(self.all_centers[tag])

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
                # print(bag_id, self.bag_person_dist[bag_id], self.bag_person_association[bag_id],
                #       self.all_centers['persons'][self.bag_person_association[bag_id]])

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

    # def update_bag_person_association(self):
    #     for bag_id in self.last_frame['bags']['id']:
    #         if bag_id not in self.bag_person_association or self.bag_person_association[bag_id] is None:
    #             person_id, dist = self.find_bag_owner(bag_id)
    #             self.bag_person_association[bag_id] = person_id
    #             self.bag_person_dist[bag_id] = dist
    #         elif self.bag_person_association[bag_id] not in self.last_frame['persons']['id']:
    #             self.bag_person_dist[bag_id] = float('inf')
    #         else:
    #             bag_person_vector = (self.all_centers['persons'][self.bag_person_association[bag_id]] -
    #                                  self.all_centers['bags'][bag_id])
    #             self.bag_person_dist[bag_id] = np.sqrt(np.power(bag_person_vector, 2).sum())
    #             print(bag_id, self.bag_person_dist[bag_id], self.bag_person_association[bag_id],
    #                   self.all_centers['persons'][self.bag_person_association[bag_id]])
    #
    # def find_bag_owner(self, bag_id):
    #     bag_center = self.all_centers['bags'][bag_id]
    #     dists = []
    #     for person_id in self.last_frame['persons']['id']:
    #         person_center = self.all_centers['persons'][person_id]
    #         dists.append(np.sqrt(np.power(person_center - bag_center, 2).sum()))
    #     closest_person_ind = int(np.array(dists).argmin())
    #     if dists[closest_person_ind] < self.bag_person_thres:
    #         return self.last_frame['persons']['id'][closest_person_ind], dists[closest_person_ind]
    #     else:
    #         return None, float('inf')

    def update(self, boxes, labels):
        centers = compute_center(boxes)
        bag_bounding_centers, persons_bounding_centers = split_bag_persons(centers, labels)

        self.frame2frame_association(persons_bounding_centers, 'persons')
        self.frame2frame_association(bag_bounding_centers, 'bags')
        self.update_bag_person_association()
        
        print(self.prev_frame_ids)
        print(self.is_unattended())

    def is_unattended(self):
        distances = np.array([self.bag_person_dist[bag_id] for bag_id in self.prev_frame_ids['bags']])
        return distances > self.bag_person_thres
