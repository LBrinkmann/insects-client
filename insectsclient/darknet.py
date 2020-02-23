import os


def transform_box(bbox_xmax, bbox_xmin, bbox_ymax, bbox_ymin, **_):
    center_x = (bbox_xmax + bbox_xmin) / 2
    center_y = (bbox_ymax + bbox_ymin) / 2
    width = (bbox_xmax - bbox_xmin)
    height = (bbox_ymax - bbox_ymin)
    return center_x, center_y, width, height


def back_transform_box(x, y, w, h):
    bbox_xmin = x - w / 2
    bbox_xmax = x + w / 2
    bbox_ymin = y - h / 2
    bbox_ymax = y + h / 2
    return bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax



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



def parse_txt(*, label_map, appearances, local_path, **_):
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


def get_appearances(image_path):
    folder, basename = os.path.split(image_path)
    txt_basename = os.path.splitext(basename)[0] + '.txt'
    txt_path = os.path.join(folder, txt_basename)
    apps = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            label_id, x, y, w, h = line.split(' ')
            bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax = back_transform_box(
                float(x), float(y), float(w), float(h))
            apps.append({
                'label_id': label_id,
                'bbox_xmin': bbox_xmin,
                'bbox_xmax': bbox_xmax,
                'bbox_ymin': bbox_ymin,
                'bbox_ymax': bbox_ymax
            })
    return apps
