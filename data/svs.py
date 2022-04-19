#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from openslide import OpenSlide


class SVS(object):
    # zoom = 1.0

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

        # self.mask = self.__create_mask(
        #     self.image.slide.dimensions,
        #     [(x, y) for x, y in self.annotation.Vertices]
        # )

    # @staticmethod
    # def __create_mask(shape, polygon):
    #     """
    #     Mask iamge requires about 20GiB RAM.
    #
    #     :param shape:
    #     :param polygon:
    #     :return:
    #     """
    #     mask = Image.new("1", shape, 0)
    #     draw = ImageDraw.Draw(mask)
    #     draw.polygon(polygon, fill=1)
    #
    #     # WARNING: This conversion requires 8 times larger RAM
    #     # mask = np.array(mask, dtype=bool)
    #
    #     return mask

    def thumbnail(
            self,
            shape           # Max-shape of target image
    ) -> (Image, float):    # Thumbnail image, and zoom ratio
        dims = self.image.slide.dimensions
        zoom = min(shape[0] / dims[0], shape[1] / dims[1])

        # Convert PIL image to Numpy array, swap X and Y axes
        image = self.image.slide.get_thumbnail(shape)

        return image, zoom

    def mask(self, vertices: list, zoom: float = 1.0) -> Image:
        """
        Mask image requires about 20GiB RAM.

        :param vertices:    Result of SVSAnnotation.vertices()
        :param zoom:        Zoom ratio of vertices
        :return:
        """

        # Init
        mask = Image.new("1", self.image.slide.dimensions, 0)
        draw = ImageDraw.Draw(mask)

        for _, annot in vertices:
            for _, region in annot:
                draw.polygon(
                    [(x * zoom, y * zoom) for x, y in region],
                    fill=1
                )

        # WARNING: This conversion requires 8 times larger RAM
        # mask = np.array(mask, dtype=bool)

        return mask

    def patches(self, size=(256, 256), stride=(64, 64)):
        shape = self.image.slide.dimensions

        for j in range(0, shape[1], stride[1]):
            for i in range(0, shape[0], stride[0]):
                yield (i, j), (i + size[0], j + size[1])

    def crop_img(self, location, size):
        """
        :param location:    Left-upper position
        :param size:        (width, height)
        :return:            image given location + size
        """
        return self.image.slide.read_region(location, 0, size)

    def crop_mask(self, location, size):
        """
        :param location:    Left-upper position
        :param size:        (width, height)
        :return:            Mask image of given location + size
        """
        mask = self.mask().crop(box=(
            location[0],
            location[1],
            location[0] + size[0],
            location[1] + size[1]
        ))

        return np.array(mask, dtype='uint8') * 255

    def extract_img_mask(self, location, size):
        return self.crop_img(location, size), self.crop_mask(location, size)


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
        self.__root = et.getroot()

        self._microns_per_pixel = self.__root.attrib["MicronsPerPixel"]
        # self._vertices = np.array([
        #     [int(v.attrib["X"]), int(v.attrib["Y"])]    #, int(v.attrib["Z"])] # Ignore Z
        #     for v in root.find("Annotation").find("Regions").find("Region").find("Vertices").findall("Vertex")
        # ])

    @property
    def MicronsPerPixel(self):
        return self._microns_per_pixel

    # @property
    # def Vertices(self):
    #     return self.vertices(1, 1)

    def vertices(self, index: int = None, region: int = None, zoom: float = 1.0) -> list:
        """
        :return:    vertices in pixel
        """

        index = '' if index is None else f'[@Id="{index}"]'
        region = '' if region is None else f'[@Id="{region}"]'

        return [
            (
                annot.get('Id'),
                [
                    (
                        region.get('Id'),
                        np.array([
                            [int(v.attrib["X"]), int(v.attrib["Y"])]
                            for v in region.findall('Vertices/Vertex')
                        ])
                    )
                    for region in annot.findall(f'Regions/Region{region}')
                ]
            )
            for annot in self.__root.findall(f'Annotation{index}')
        ]


def save_patches(
        path_svs: Path, path_xml: Path,
        base: Path,
        size: (int, int), stride: (int, int), resize: (int, int) = None,
        index: int = None, region: int = None
):
    """

    :param path_svs:    Path to image svs
    :param path_xml:    Path to contour xml
    :param base:        Base string of output file name
    :param size:        Patch size
    :param stride:      Patch stride
    :param resize:      Resize extracted patch
    :param index:       Numerical index of annotation
    :param region:      Numerical index of region in the annotation
    :return:        None
    """

    svs = SVS(
        path_svs,
        annotation=path_xml
    )
    mask = svs.mask(
        vertices=svs.annotation.vertices(index, region, zoom=1.0),
        zoom=1.0
    )

    for i, (p0, p1) in enumerate(svs.patches(size=size, stride=stride)):
        patch_path = str(base) + f"{i:08}img.png"
        if Path(patch_path).exists():
            continue

        # mask = svs.crop_mask(p0, size)
        cropped_mask = mask.crop(box=(
            p0[0], p0[1],
            p0[0] + size[0], p0[1] + size[1]
        ))
        if np.sum(np.array(cropped_mask)) < size[0] * size[1]:
            # Ignore if the mask does not full-cover the patch region
            continue

        img = svs.crop_img(p0, size)
        if resize is not None:
            img = img.resize(resize)

        print(patch_path)
        img.save(patch_path)

    del svs, mask
