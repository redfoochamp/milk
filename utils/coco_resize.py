import cv2
import json
import os
import pathlib
import shutil

''' Code for Resizing COCO datasets '''


def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def remove_class(json_contents, class_name='Плохой bbox'):
    for categorie in json_contents['categories']:
        if categorie['name'] == class_name:
            class_remove_id = categorie['id']
            class_remove_name = categorie['name']
            break

    json_contents['categories'] = [obj for obj in json_contents['categories'] \
                                   if obj['name'] != class_remove_name]
    json_contents['annotations'] = [obj for obj in json_contents['annotations'] \
                                    if obj['category_id'] != class_remove_id]

    return json_contents


def resize_dataset(dataset_dir, output_dataset_dir, target_height):
    copy_and_overwrite(dataset_dir, output_dataset_dir)

    with open(os.path.join(output_dataset_dir, 'result.json')) as json_fp:
        json_contents = json.load(json_fp)

    json_contents = remove_class(json_contents)

    for index, image_contents in enumerate(tqdm(json_contents['images'])):
        w, h = image_contents['width'], image_contents['height']

        image_path = str(os.path.join(output_dataset_dir, image_contents['file_name']))
        if not os.path.exists(image_path):  # if label-studio added some nonsence
            file_name = image_contents['file_name'].split(os.path.sep)
            image_contents['file_name'] = os.path.sep.join(file_name[:-2] + file_name[-1:])
            image_path = str(os.path.join(output_dataset_dir, image_contents['file_name']))

        image = cv2.imread(image_path)

        assert (h, w) == image.shape[:2], f'image shape is not same as the notation says, {image.shape[:2]}'

        target_width = int(w / (h / target_height))
        image_resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(image_path, image_resized)
        image_contents['width'] = target_width
        image_contents['height'] = target_height
        json_contents['images'][index] = image_contents
        for annotation_index, annotation in enumerate(json_contents['annotations']):
            if annotation['image_id'] == image_contents['id']:
                annotation['bbox'][0] = int(round(annotation['bbox'][0] / w * target_width))
                annotation['bbox'][1] = int(round(annotation['bbox'][1] / h * target_height))
                annotation['bbox'][2] = int(round(annotation['bbox'][2] / w * target_width))
                annotation['bbox'][3] = int(round(annotation['bbox'][3] / h * target_height))
                annotation['area'] = annotation['bbox'][1] * annotation['bbox'][3]

    with open(os.path.join(output_dataset_dir, 'result.json'), 'w') as json_fp:
        json.dump(json_contents, json_fp, indent=4)


def remove_multiclass(annotation_json):
    with open(annotation_json) as json_fp:
        json_contents = json.load(json_fp)

    for annotation in json_contents['annotations']:
        annotation['category_id'] = 0

    json_contents['categories'] = [{'id': 0, 'name': 'box'}]

    with open(annotation_json, 'w') as json_fp:
        json.dump(json_contents, json_fp, indent=4)