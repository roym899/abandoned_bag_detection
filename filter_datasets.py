import json
import os
from shutil import copy

from tqdm import tqdm

OUTPUT_FOLDER = '/home/semaiov/Documents/cvenv/ws/datasets/person_bag'
DATASETS = ['ADE20K', 'COCO']
SPLITS = ['train', 'val']
ADE20K_FOLDER = '/home/semaiov/Documents/cvenv/ws/datasets/ADE20K_coco'
COCO_FOLDER = '/home/semaiov/Documents/cvenv/ws/datasets/coco'
PERSON_ID = 1
BAG_ID = 2

def get_json_path(dataset, split):
    if dataset == 'ADE20K' and split == 'val':
        return os.path.join(ADE20K_FOLDER, 'annotations', 'val.json')
    if dataset == 'ADE20K' and split == 'train':
        return os.path.join(ADE20K_FOLDER, 'annotations', 'train.json')
    if dataset == 'COCO' and split == 'train':
        return os.path.join(COCO_FOLDER, 'annotations', 'instances_train2017.json')
    if dataset == 'COCO' and split == 'val':
        return os.path.join(COCO_FOLDER, 'annotations', 'instances_val2017.json')
                
def get_person_ids_for_dataset(dataset):
    if dataset == 'COCO':
        return [1]
    elif dataset == 'ADE20K':
        return [24]
        
def get_bag_ids_for_dataset(dataset):
    if dataset == 'COCO':
        return [27, 31, 33]
    elif dataset == 'ADE20K':
        return [29, 124, 1011, 1515, 1513, 1537, 2217, 2347, 2604, 178, 833]
    
def get_image_folder(dataset, split):
    if dataset == 'ADE20K' and split == 'val':
        return os.path.join(ADE20K_FOLDER, 'val')
    if dataset == 'ADE20K' and split == 'train':
        return os.path.join(ADE20K_FOLDER, 'train')
    if dataset == 'COCO' and split == 'train':
        return os.path.join(COCO_FOLDER, 'images', 'train2017')
    if dataset == 'COCO' and split == 'val':
        return os.path.join(COCO_FOLDER, 'images', 'val2017')
    

def main():
    for split in SPLITS:
        # create coco format json
        output_data = {'info': {},
                    'licenes': {},
                    'images': [],
                    'annotations': [],
                    'categories': []}
        
        # TODO: licenses structs
        
        # add info dictionary
        output_data['info'] = {
            'contributor': 'WASP AS1',
            'date_created': '2020/02/27',
            'description': 'ADE20K + COCO persons and bags only',
            'url': '',
            'version': '0.1',
            'year': 2020
        }
        
        # add the categories that we want
        output_data['categories'] = [
            {
                'supercategory': 'person',
                'id': PERSON_ID,
                'name': 'person'
            },
            {
                'supercategory': 'bag',
                'id': BAG_ID,
                'name': 'bag'
            }
        ]
        for dataset in DATASETS:
            with open(get_json_path(dataset, split)) as f:
                data = json.load(f)
                
                id_to_index = {}
                for index, image in enumerate(data['images']):
                    id_to_index[image['id']] = index
                
                
                # go through coco annotations
                filtered_image_ids = set()
                filtered_dataset_images = []

                for annotation in data['annotations']:
                    if annotation['category_id'] in get_person_ids_for_dataset(dataset):
                        output_data['annotations'].append(annotation)
                        output_data['annotations'][-1]['category_id'] = PERSON_ID
                        filtered_image_ids.add(annotation['image_id'])
                    elif annotation['category_id'] in get_bag_ids_for_dataset(dataset):
                        output_data['annotations'].append(annotation)
                        output_data['annotations'][-1]['category_id'] = BAG_ID
                        filtered_image_ids.add(annotation['image_id'])
                
                # go through the filtered images
                for filtered_image_id in filtered_image_ids:
                    output_data['images'].append(data['images'][id_to_index[filtered_image_id]])
                    filtered_dataset_images.append(data['images'][id_to_index[filtered_image_id]])
            

                # move the images
                output_folder = os.path.join(OUTPUT_FOLDER, 'images', split)
                os.makedirs(output_folder, exist_ok=True)
                for image in tqdm(filtered_dataset_images):
                    image_path = os.path.join(get_image_folder(dataset, split), image['file_name'])
                    output_path = os.path.join(output_folder, image['file_name'])
                    copy(image_path, output_path)
                    
        json_output_folder = os.path.join(OUTPUT_FOLDER, 'annotations')
        os.makedirs(json_output_folder, exist_ok=True)
        json_output_path = os.path.join(json_output_folder, f'{split}.json')
        with open(json_output_path, 'w') as f:
            json.dump(output_data, f)

if __name__ == '__main__':
    main()
