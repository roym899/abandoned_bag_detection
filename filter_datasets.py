import json
import os
from shutil import copy

from tqdm import tqdm

OUTPUT_FOLDER = '/home/leo/datasets/person_bag/'
COCO_FOLDER = '/home/leo/datasets/coco/'
COCO_TRAIN_JSON = '/home/leo/datasets/coco/annotations/instances_train2017.json'
COCO_PERSON_IDS = [1]
COCO_BAG_IDS = [27, 31, 33]
PERSON_ID = 1
BAG_ID = 2

def main():
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
    
    with open(COCO_TRAIN_JSON) as f:
        data = json.load(f)
        
        id_to_index = {}
        for index, image in enumerate(data['images']):
            id_to_index[image['id']] = index
        
        
        # go through coco annotations
        filtered_image_ids = set()

        for annotation in data['annotations']:
            if annotation['category_id'] in COCO_PERSON_IDS:
                output_data['annotations'].append(annotation)
                output_data['annotations'][-1]['category_id'] = PERSON_ID
                filtered_image_ids.add(annotation['image_id'])
            elif annotation['category_id'] in COCO_BAG_IDS:
                output_data['annotations'].append(annotation)
                output_data['annotations'][-1]['category_id'] = BAG_ID
                filtered_image_ids.add(annotation['image_id'])
        
        # go through the filtered images
        for filtered_image_id in filtered_image_ids:
            output_data['images'].append(data['images'][id_to_index[filtered_image_id]])
    

        # move the images
        output_folder = os.path.join(OUTPUT_FOLDER, 'images', 'train')
        os.makedirs(output_folder, exist_ok=True)
        for image in tqdm(output_data['images']):
            image_path = os.path.join(COCO_FOLDER, 'images', 'train2017', image['file_name'])
            output_path = os.path.join(output_folder, image['file_name'])
            copy(image_path, output_path)
            
    json_output_folder = os.path.join(OUTPUT_FOLDER, 'annotations')
    os.makedirs(json_output_folder, exist_ok=True)
    json_output_path = os.path.join(json_output_folder, 'train.json')
    with open(json_output_path, 'w') as f:
        json.dump(output_data, f)

if __name__ == '__main__':
    main()