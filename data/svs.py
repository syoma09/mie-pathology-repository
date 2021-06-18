#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from openslide import OpenSlide


class SVS(object):
    zoom = 1.0

    def __init__(self, path: Path, annotation=None):
        if not path.suffix == ".svs":
            raise IOError("*.svs file is required by SVS()")

        print("SVS() load image from     :", path)
        self.image = SVSImage(path)

        if annotation is None:
            # print(path.parent / (path.stem + ".xml"))
            annotation = path.parent / (path.stem + ".xml")
        print("SVS() load annotation from:", annotation)
        self.annotation = SVSAnnotation(annotation)

        self.mask = self.__create_mask(
            self.image.slide.dimensions,
            [(x, y) for x, y in self.annotation.Vertices]
        )

    @staticmethod
    def __create_mask(shape, polygon):
        """
        Mask iamge requires about 20GiB RAM.

        :param shape:
        :param polygon:
        :return:
        """
        mask = Image.new("1", shape, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon, fill=1)

        # WARNING: This conversion requires 8 times larger RAM
        # mask = np.array(mask, dtype=bool)

        return mask

    def get_thumbnail(self, shape):
        dims = self.image.slide.dimensions
        zoom = min(shape[0] / dims[0], shape[1] / dims[1])

        # Convert PIL image to Numpy array, swap X and Y axes
        image = self.image.slide.get_thumbnail(shape)
        # Multiply image zoom ratio to fit the annotation vertices to resized image
        annot = (self.annotation.Vertices * zoom).astype(np.int)

        return image, annot

    def patches(self, size=(256, 256), stride=(64, 64)):
        shape = self.image.slide.dimensions

        for j in range(0, shape[1], stride[1]):
            for i in range(0, shape[0], stride[0]):
                yield (i, j), (i + size[0], j + size[1])

    def extract_img_mask(self, location, size):
        """
        Return image and mask at given location + size
        :param location:
        :param size:
        :return:
        """
        img = self.image.slide.read_region(location, 0, size)

        mask = self.mask.crop(box=(
            location[0],
            location[1],
            location[0] + size[0],
            location[1] + size[1]
        ))
        mask = np.array(mask, dtype='uint8') * 255

        return img, mask


class SVSImage(object):
    def __init__(self, path):
        self.slide = OpenSlide(str(path))

        # print("level_count: ", self.slide.level_count)
        # print("dimensions : ", self.slide.dimensions)

        # img = self.slide.get_thumbnail(self.slide.dimensions)
        # print(img)
        # img.save("./test.jpg")


class SVSAnnotation(object):
    def __init__(self, path):
        et = ET.parse(path)
        root = et.getroot()

        self._microns_per_pixel = root.attrib["MicronsPerPixel"]
        self._vertices = np.array([
            [int(v.attrib["X"]), int(v.attrib["Y"])]    #, int(v.attrib["Z"])] # Ignore Z
            for v in root.find("Annotation").find("Regions").find("Region").find("Vertices").findall("Vertex")
        ])

    @property
    def MicronsPerPixel(self):
        return self._microns_per_pixel

    @property
    def Vertices(self):
        """
        :return:    vertices in pixel
        """
        return self._vertices


def save_patches(path_svs: Path, path_xml: Path, base, size, stride, resize=None):
    """

    :param path_svs:    Path to image svs
    :param path_xml:    Path to contour xml
    :param base:        Base string of output file name
    :param size:        Patch size
    :param stride:      Patch stride
    :param resize:      Resize extracted patch
    :return:        None
    """

    svs = SVS(
        path_svs,
        annotation=path_xml
    )

    for i, (p0, p1) in enumerate(svs.patches(size=size, stride=stride)):
        patch_path = str(base) + f"{i:08}img.png"
        if Path(patch_path).exists():
            continue

        # print(p0, p1)
        img, mask = svs.extract_img_mask(p0, size)

        if np.sum(mask) == size[0] * size[1] * 255:
            if resize is not None:
                img = img.resize(resize)

            print(patch_path)
            img.save(patch_path)
            # Image.fromarray(mask).save(str(base) + f"{i:08}mask.png")

    del svs
