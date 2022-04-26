import cv2
import json
import numpy as np
import PIL
import os, glob
import shutil

from sklearn.model_selection import train_test_split


# Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


# COCO JSON to YOLOv5 PyTorch format (txt labels)
def coco_to_yolo(data_json, dir_to_write):
    for i in range(len(data_json['images'])):
        image_id = data_json['images'][i]['id']
        move_file = data_json['images'][i]['file_name']
        file_name = data_json['images'][i]['file_name'].replace('original_images/', '').replace('.jpg', '')
        width, height = data_json['images'][i]['width'], data_json['images'][i]['height']

        scale_x = 1 / width
        scale_y = 1 / height

        dir_to_file = f'{dir_to_write}/{file_name}.txt'

        for annotation in data_json['annotations']:
            if annotation['image_id'] == image_id:
                bbox_yolo = []
                category = annotation['category_id']

                xc = (annotation['bbox'][0] + (annotation['bbox'][2] / 2)) * scale_x
                yc = (annotation['bbox'][1] + (annotation['bbox'][3] / 2)) * scale_y
                xmax = annotation['bbox'][2] * scale_x
                ymax = annotation['bbox'][3] * scale_y
                bbox_yolo.append([category, xc, yc, xmax, ymax])

                yolo_format = ' '.join(str(x) for x in bbox_yolo[0])

                with open(dir_to_file, 'a') as f:
                    f.write(yolo_format)
                    f.write('\n')

    images = [os.path.join('original_images', x) for x in os.listdir('/content/original_images')]
    annotations = [os.path.join('original_labels', x) for x in os.listdir('/content/original_labels')]

    images = [x for x in images if x.replace('original_images/', '').replace('.jpg', '') in
              list(map(lambda x: x.replace('original_labels/', '').replace('.txt', ''), annotations))]

    images.sort()
    annotations.sort()

    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.1,
                                                                                    random_state=42)

    move_files_to_folder(train_images, 'train/images/')
    move_files_to_folder(val_images, 'val/images/')
    move_files_to_folder(train_annotations, 'train/labels/')
    move_files_to_folder(val_annotations, 'val/labels/')