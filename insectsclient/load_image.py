import wget
import os


def _download(_from, _to):
    if not os.path.exists(_to):
        wget.download(_from, _to, bar=None)


def download(url, out_dir):
    basename = os.path.basename(url)
    _to = os.path.join(out_dir, basename)
    _download(url, _to)
    return _to
