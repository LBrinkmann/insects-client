import os
import progressbar
import requests
import random
from . import darknet
from . import load_image


# from PIL import Image

PLATFORM_URL = os.environ.get('INSECTS_PLATFORM_URL', 'http://0.0.0.0:5000/')
# DATA_DIR = os.environ.get('INSECTS_DATA_DIR', 'data')



def get_collection(collection_id):
    path = os.path.join(PLATFORM_URL, 'dataset', str(collection_id))
    r = requests.get(path)
    return r.json()['collection']


def get_all_labels(frames):
    labels = {}
    for frame in frames:
        for appearance in frame['appearances']:
            for appearance_labels in appearance['appearance_labels']:
                labels[appearance_labels['label']['id']] = appearance_labels['label']
    return list(labels.values())


def parse_frames(frames, label_map, parser, data_dir='data/images'):
    paths = []
    for frame in progressbar.progressbar(frames):
        local_path = load_image.download(frame['url'], data_dir)
        # im = Image.open(local_path)
        # width, height = im.size
        parser(**frame, label_map=label_map, local_path=local_path)
        paths.append(local_path)
    return paths


def import_collection(collection_id, export_format='darknet'):
    frames = get_collection(collection_id)
    labels = get_all_labels(frames)
    label_map = {lid['id']: i for i, lid in enumerate(labels)}
    if export_format == 'darknet':
        frame_paths = parse_frames(frames, label_map, darknet.parse)
    else:
        raise NotImplementedError('currently only darknet is supported')
    return labels, frame_paths


def create_file_list(file_list, path):
    path = darknet.create_file_list(file_list, path)
    return path


def create_train_obj(labels, frame_paths, train_fraction=0.8, data_dir='data/meta', temp_dir='data/temp'):
    random.shuffle(frame_paths)
    n_train = int(len(frame_paths)*train_fraction)
    train_paths = frame_paths[:n_train]
    test_paths = frame_paths[n_train:]

    return darknet.create_meta(
        labels, train_paths, test_paths, data_dir, temp_dir)










