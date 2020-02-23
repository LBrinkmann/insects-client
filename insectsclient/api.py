import os
import progressbar
import requests
import random
import json
from . import darknet
from . import load_image


def get_collection(collection_id, with_appearances_only=False):
    with_appearances_only_str = 'true' if with_appearances_only else 'false'
    PLATFORM_URL = os.environ.get('INSECTS_PLATFORM_URL', 'http://0.0.0.0:5000/')
    path = os.path.join(PLATFORM_URL, 'dataset', str(collection_id))
    r = requests.get(path, params={'with_appearances_only': with_appearances_only_str})
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
        paths.append({'path': local_path, 'id': frame['id']})
    return paths


def import_collection(collection_id, data_dir, with_appearances_only=False, export_format='darknet'):
    frames = get_collection(collection_id, with_appearances_only)
    labels = get_all_labels(frames)
    label_map = {lid['id']: i for i, lid in enumerate(labels)}
    if export_format == 'darknet':
        frame_paths = parse_frames(frames, label_map, darknet.parse, data_dir)
    else:
        raise NotImplementedError('currently only darknet is supported')
    return labels, frame_paths


def upload_appearances(frame_paths, labels, creator_id):
    PLATFORM_URL = os.environ.get('INSECTS_PLATFORM_URL', 'http://0.0.0.0:5000/')
    all_apps = []
    for frame in frame_paths:
        try:
            apps = darknet.get_appearances(frame['path'])
            apps = [{**a, 'label_id': labels[int(a['label_id'])]['id'], 'frame_id': frame['id']} for a in apps]
            all_apps.extend(apps)
        except:
            pass
    data = {
        'creator_id': creator_id,
        'label_appearances': all_apps
    }
    headers = {'content-type': 'application/json'}
    r = requests.post(url=os.path.join(PLATFORM_URL, 'label_appearances/'), data=json.dumps(data), headers=headers)
    if r.json()['success']:
        print('Successfully uploaded {} appearances'.format(len(all_apps)))


def create_file_list(frame_paths, path):
    _frame_paths = [fp['path'] for fp in frame_paths]
    path = darknet.create_file_list(_frame_paths, path)
    return path


def create_train_obj(labels, frame_paths, train_fraction=0.8, data_dir='data/meta', temp_dir='data/temp'):
    _frame_paths = [fp['path'] for fp in frame_paths]
    random.shuffle(_frame_paths)
    n_train = int(len(_frame_paths)*train_fraction)
    train_paths = _frame_paths[:n_train]
    test_paths = _frame_paths[n_train:]

    return darknet.create_meta(
        labels, train_paths, test_paths, data_dir, temp_dir)
