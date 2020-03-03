#!/usr/bin/env python
# coding: utf-8



import numpy as np
import abandoned_bag_heuristic as abh
import pickle

with open('predictions.pkl', 'rb') as fp:
    predictions = pickle.load(fp)

dect = abh.SimpleTracker(50, 100)

for pred in predictions:
    dect.update(boxes=pred.pred_boxes.tensor.numpy(), labels=pred.pred_classes.numpy())

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




