import os


def transform_box(bbox_xmax, bbox_xmin, bbox_ymax, bbox_ymin, **_):
    center_x = (bbox_xmax + bbox_xmin) / 2
    center_y = (bbox_ymax + bbox_ymin) / 2
    width = (bbox_xmax - bbox_xmin)
    height = (bbox_ymax - bbox_ymin)
    return center_x, center_y, width, height


def parse(*, label_map, appearances, local_path, **_):
    folder, basename = os.path.split(local_path)
    txt_basename = os.path.splitext(basename)[0] + '.txt'
    txt_fullpath = os.path.join(folder, txt_basename)
    with open(txt_fullpath, 'w') as f:
        for appearance in appearances:
            for appearance_labels in appearance['appearance_labels']:
                f.write('{} {} {} {} {}\n'.format(
                    label_map[appearance_labels['label']['id']],
                    *transform_box(**appearance)
                ))


def create_data(data, meta_path):
    data_path = os.path.join(meta_path, 'obj.data')
    with open(data_path, 'w') as f:
        for k, v in data.items():
            f.write('{} = {}\n'.format(k, v))
    return data_path


def create_names(labels, meta_path):
    names_path = os.path.join(meta_path, 'obj.names')
    with open(names_path, 'w') as f:
        for label in labels:
            f.write('{}\n'.format(label['scientificName']))
    return names_path


def create_file_list(paths, path):
    with open(path, 'w') as f:
        for p in paths:
            f.write('{}\n'.format(p))
    return path


def create_meta(labels, train_paths, test_paths, meta_path, temp_path):
    data = {
        'classes': len(labels),
        'names': create_names(labels, meta_path),
        'valid': create_file_list(test_paths, os.path.join(meta_path, 'test.txt')),
        'train': create_file_list(train_paths, os.path.join(meta_path, 'train.txt')),
        'backup': temp_path
    }
    return create_data(data, meta_path)
