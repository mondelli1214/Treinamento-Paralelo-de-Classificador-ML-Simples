
import gzip
import os
import urllib.request
import numpy as np

MNIST_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

def _download_if_needed(path, url):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Baixando {url} -> {path}")
        urllib.request.urlretrieve(url, path)

def _read_idx_images(path_gz):
    with gzip.open(path_gz, 'rb') as f:
        data = f.read()
    # parse IDX
    magic = int.from_bytes(data[0:4], 'big')
    assert magic == 2051, f"magic errado para imagens: {magic}"
    n_images = int.from_bytes(data[4:8], 'big')
    n_rows = int.from_bytes(data[8:12], 'big')
    n_cols = int.from_bytes(data[12:16], 'big')
    imgs = np.frombuffer(data, dtype=np.uint8, offset=16)
    imgs = imgs.reshape(n_images, n_rows*n_cols).astype(np.float32) / 255.0
    return imgs

def _read_idx_labels(path_gz):
    with gzip.open(path_gz, 'rb') as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], 'big')
    assert magic == 2049, f"magic errado para labels: {magic}"
    n_items = int.from_bytes(data[4:8], 'big')
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels

def load_mnist(data_dir="data", auto_download=True):
    paths = {
        "train_images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
        "test_images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        "test_labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    }
    if auto_download:
        for k, url in MNIST_URLS.items():
            _download_if_needed(paths[k], url)
    X_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    X_test  = _read_idx_images(paths["test_images"])
    y_test  = _read_idx_labels(paths["test_labels"])
    return (X_train, y_train), (X_test, y_test)
