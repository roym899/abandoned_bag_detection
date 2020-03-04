#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import time

import abandoned_bag_heuristic as abh

with open('predictions.pkl', 'rb') as fp:
    predictions = pickle.load(fp)

dect = abh.SimpleTracker(150, 100)
fig, ax = plt.subplots(1)


def set_resolution():
    ax.set(xlim=(0, 640), ylim=(0, 480))
    ax.invert_yaxis()


plt.ion()
plt.show()

colors = {0: 'r', 1: 'b'}

for pred in predictions:
    bboxs = pred.pred_boxes.tensor.numpy()
    labels = pred.pred_classes.numpy()
    dect.update(boxes=bboxs, labels=labels)
    ax.cla()
    set_resolution()
    for bbox, label in zip(bboxs, labels):
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                 edgecolor=colors[label], linewidth=2.5)
        ax.add_patch(rect)
    for tag, ids in dect.prev_frame_ids.items():
        for id in ids:
            center = dect.all_centers[tag][id]
            ax.text(*center[:2], str(id))
    for bag_id, person_id in dect.bag_person_association.items():
        if person_id is not None and dect.bag_person_dist[bag_id] < np.inf:
            bag_center = dect.all_centers['bags'][bag_id][:2]
            person_center = dect.all_centers['persons'][person_id][:2]
            ax.arrow(*bag_center, *(person_center - bag_center))

    plt.draw()
    plt.pause(0.1)
    # input("Press enter to continue... ")

# # In[42]:
#
#
# box_frame_one = np.array([
#     [1., 1., 3., 3.],
#     [2., 2., 4., 4.],
#     [10., 1., 12., 3.]])
#
# labels_one = np.array([0, 28, 0])
#
# box_frame_two = np.array([
#     [2., 1., 4., 3.],
#     [9., 1., 11., 3.],
#     [2., 2., 4., 4.]])
#
# labels_two = np.array([0, 0, 28])
#
# box_frame_three = np.array([
#     [2., 1., 4., 3.],
#     [9., 1., 11., 3.],
#     [10., 2., 12., 4.],
#     [2., 2., 4., 4.]])
#
# labels_three = np.array([0, 0, 28, 28])
#
#
# # In[65]:
#
# dect = abh.SimpleTracker(2, 2)
#
#
# # In[66]:
#
#
# dect.update(box_frame_one, labels_one)
#
#
# # In[67]:
#
#
# print(dect.bag_person_association)
# print(dect.bag_person_dist)
#
#
# # In[68]:
#
# dect.update(box_frame_two, labels_two)
#
#
# # In[69]:
#
#
# print(dect.bag_person_association)
# print(dect.bag_person_dist)
#
#
# # In[69]:
#
#
# dect.update(box_frame_three, labels_three)
#
#
# # In[59]:
#
#
# print(dect.bag_person_association)
# print(dect.bag_person_dist)
# print(dect.prev_frame_ids)
# print(dect.all_centers)
#
