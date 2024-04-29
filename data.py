from typing import Literal
import numpy as np

DATASET_INFO = {
    "digit": {
        "images": {
            "train": "digitdata/trainingimages",
            "validation": "digitdata/validationimages",
            "test": "digitdata/testimages",
        },
        "labels": {
            "train": "digitdata/traininglabels",
            "validation": "digitdata/validationlabels",
            "test": "digitdata/testlabels",
        },
        "width": 28,
        "height": 28,
    },
    "face": {
        "images": {
            "train": "facedata/facedatatrain",
            "validation": "facedata/facedatavalidation",
            "test": "facedata/facedatatest",
        },
        "labels": {
            "train": "facedata/facedatatrainlabels",
            "validation": "facedata/facedatavalidationlabels",
            "test": "facedata/facedatatestlabels",
        },
        "width": 60,
        "height": 70,
    },
}


def load_data(
    dataset: Literal["digit", "face"],
    split: Literal["train", "validation", "test"],
):
    info = DATASET_INFO[dataset]

    # imgs
    ipath = info["images"][split]
    with open(ipath) as f:
        ilines = f.readlines()

    ibuf = np.zeros(shape=(len(ilines), info["width"]), dtype=np.uint8)

    for row, line in enumerate(ilines):
        for col, ch in enumerate(line):
            if ch == " ":
                pixel = 0
            elif ch == "+":
                pixel = 1
            elif ch == "#":
                pixel = 2
            else:
                continue
            ibuf[row, col] = pixel

    ibuf = ibuf.reshape(len(ilines) // info["height"], info["height"], info["width"])

    # labels
    lpath = info["labels"][split]
    with open(lpath) as f:
        llines = f.readlines()

    lbuf = np.zeros(shape=(len(llines)), dtype=np.uint8)
    for i, label in enumerate(llines):
        lbuf[i] = int(label)

    return ibuf, lbuf


if __name__ == "__main__":
    imgs, labels = load_data("digit", "train")
    for img, label in zip(imgs, labels):
        print(img, label)

    # get first 2 10%s example
    # imgs[: (imgs.shape[0] // 10) * 2]
