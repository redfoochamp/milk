import numpy as np
import pandas as pd
import os
from PIL import Image
from collections import OrderedDict

# df_annotations = pd.read_csv('your_file')


def to_coco(df_annotations, images_dir):
    json_contents = OrderedDict()
    json_contents['images'] = []
    json_contents['categories'] = [{'id': 0, 'name': 'product'}]
    json_contents['annotations'] = []

    image_id = 0
    annotation_id = 0

    for file_name in df_annotations['file_name'].unique():
        if file_name not in os.listdir(images_dir):
            print(f'{file_name} not FOUND!')
            continue

        file_path = f'{images_dir}/{file_name}'
        width, height = Image.open(file_path).size

        json_contents['images'].append({'width': width,
                                        'height': height,
                                        'id': image_id,
                                        'file_name': file_path})

        for _, row in df_annotations[df_annotations['file_name'] == file_name].iterrows():
            annotation = OrderedDict()
            annotation['id'] = annotation_id
            annotation['image_id'] = image_id
            annotation['category_id'] = 0
            annotation['segmentation'] = []

            bbox_width = int((row.x_max-row.x_min) * width)
            bbox_height = int((row.y_max-row.y_min) * height)
            annotation['bbox'] = [max(int(row.x_min * width), 0),
                                  max(int(row.y_min * height), 0),
                                  bbox_width,
                                  bbox_height]
            annotation['ignore'] = 0
            annotation['iscrowd'] = 0
            annotation['area'] = bbox_width * bbox_height

            json_contents['annotations'].append(annotation)

            annotation_id += 1

        image_id += 1