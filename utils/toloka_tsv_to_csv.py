import pandas as pd
import json


def read_pool(pool_filename):
    df = pd.read_csv(f'{pool_filename}', sep='\t')[['INPUT:image', 'OUTPUT:result']]
    df = df.dropna()
    df['file_name'] = df['INPUT:image'].str.replace('/milk-disk/', '')
    return df

def parse_annotations(df):
    annotation_data = []

    for _, row in df.iterrows():
        json_contents = json.loads(row['OUTPUT:result'])
        for annotation in json_contents:
            x1 = annotation['data']['p1']['x']
            y1 = annotation['data']['p1']['y']
            x2 = annotation['data']['p2']['x']
            y2 = annotation['data']['p2']['y']
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            filename = row['file_name']
            annotation_data.append([filename, x_min, x_max, y_min, y_max])

    return pd.DataFrame(annotation_data, columns=['file_name', 'x_min', 'x_max', 'y_min', 'y_max'])

# EXAMPLE
# df1 = parse_annotations(read_pool('assignments_from_pool_29245653__23-11-2021.tsv'))
# df2 = parse_annotations(read_pool('assignments_from_pool_29537023__23-11-2021.tsv'))
#
# df_annotations = pd.concat([df1, df2]).reset_index()
# df_annotations.to_csv('toloka_detections.csv', index=None)