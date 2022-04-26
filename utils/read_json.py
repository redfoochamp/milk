import json

'''                   COCO-JSON
      READS COCO JSONS WITH FOLLOWING INFO IN THEM

- info: width, height, id and filename of image
- categories: classes of images
- annotations: id of annot, image id, category id, bbox and area '''


def read_jsons(train_json, val_json):
    with open(val_json) as val_json_fp, open(train_json) as train_json_fp:
        val_json_contents = json.load(val_json_fp)
        train_json_contents = json.load(train_json_fp)

        assert all(item in train_json_contents['categories']
                   for item in val_json_contents['categories'])

        val_json_contents['categories'] = train_json_contents['categories']

    with open(val_json, 'w') as val_json_fp:
        json.dump(val_json_contents, val_json_fp, indent=4)

# EXAMPLE
# train_json = '/content/train/result.json'
# val_json = '/content/val/result.json'
#
# read_jsons(train_json, val_json)