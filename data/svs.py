#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from openslide import OpenSlide


class SVS(object):
    # zoom = 1.0

    def __init__(
            self,
            path: Path,
            annotation=None,
            # cache: Path = Path("~/data/_cache/mie-pathology/").expanduser()
    ):
        if not path.suffix == ".svs":
            raise IOError("*.svs file is required by SVS()")

        print("SVS() load image from     :", path)
        self.image = SVSImage(path)

        if annotation is None:
            # print(path.parent / (path.stem + ".xml"))
            annotation = path.parent / (path.stem + ".xml")
        print("SVS() load annotation from:", annotation)
        self.annotation = SVSAnnotation(annotation)

        # self.mask_path = cache / "mask" / (path.stem + ".npy")
        # if not self.mask_path.exists():
        #     # Create directory
        #     if not self.mask_path.parent.exists():
        #         self.mask_path.parent.mkdir(parents=True, exist_ok=True)
        #     # Create mask
        #     print("Start __init_mask() <<<<<", path)
        #     self.__init_mask()
        #     print("     >>>>> Finish __init_mask()")

        # self.mask = np.memmap(
        #     str(self.mask_path), mode="r",
        #     shape=self.image.slide.dimensions, dtype=bool
        # )

        # self.mask = self.__create_mask(
        #     self.image.slide.dimensions,
        #     [(x, y) for x, y in self.annotation.Vertices]
        # )
        self.mask = self.__init_mask(zoom=1.0)

    def __init_mask(self, zoom: float = 1.0) -> Image:
        """
        Mask image requires about 20GiB RAM.

        :return:
        """
        mask = Image.new("1", self.image.slide.dimensions, 0)
        draw = ImageDraw.Draw(mask)

        print("-- Pillow")
        # Use last annotation (Layer2) only
        _, annot = self.annotation.vertices()[-1]
        # Draw region polygons
        for _, region in annot:
            draw.polygon(
                [(x * zoom, y * zoom) for x, y in region],
                fill=1
            )

        return mask

        # print("-- Allocating memmap")
        # # WARNING: This conversion requires 8 times larger RAM
        # # mask = np.array(mask, dtype=bool)
        # mmap = np.memmap(
        #     str(self.mask_path), mode='w+',
        #     shape=(mask.width, mask.height), dtype=bool
        # )
        #
        # print("-- Copy pixels")
        # # TODO: Bad impl
        # for j in range(mask.height):
        #     for i in range(mask.width):
        #         mmap[i, j] = mask.getpixel((i, j)) > 0
        #
        # print("-- Writing")
        # mmap.flush()
        # print("-- Finish")

    def thumbnail(
            self,
            shape           # Max-shape of target image
    ) -> (Image, float):    # Thumbnail image, and zoom ratio
        dims = self.image.slide.dimensions
        zoom = min(shape[0] / dims[0], shape[1] / dims[1])

        # Convert PIL image to Numpy array, swap X and Y axes
        image = self.image.slide.get_thumbnail(shape)

        return image, zoom

    # def mask(self, vertices: list, zoom: float = 1.0) -> Image:
    #     """
    #     Mask image requires about 20GiB RAM.
    #
    #     :param vertices:    Result of SVSAnnotation.vertices()
    #     :param zoom:        Zoom ratio of vertices
    #     :return:
    #     """
    #
    #     # Init
    #     mask = Image.new("1", self.image.slide.dimensions, 0)
    #     draw = ImageDraw.Draw(mask)
    #
    #     for _, annot in vertices:
    #         for _, region in annot:
    #             draw.polygon(
    #                 [(x * zoom, y * zoom) for x, y in region],
    #                 fill=1
    #             )
    #
    #     # WARNING: This conversion requires 8 times larger RAM
    #     # mask = np.array(mask, dtype=bool)
    #
    #     return mask

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
        # mask = self.mask(
        #     self.annotation.vertices()[-1:]  # TODO: Layer2 only
        # ).crop(box=(
        #     location[0],
        #     location[1],
        #     location[0] + size[0],
        #     location[1] + size[1]
        # ))
        # mask = self.mask[
        #     location[0]:location[0] + size[0],
        #     location[1]:location[1] + size[1],
        # ]
        mask = self.mask.crop(box=(
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
    patch_list = pd.DataFrame(
        columns=("x", "y", "width", "height", "path")
    )

    for i, (p0, p1) in enumerate(svs.patches(size=size, stride=stride)):
        patch_path = str(base) + f"{i:08}img.png"
        print(patch_path)
        # if Path(patch_path).exists():
        #     continue

        mask = svs.crop_mask(p0, size)
        if np.sum(np.array(mask)) < size[0] * size[1]:
            # Ignore if the mask does not full-cover the patch region
            continue

        img = svs.crop_img(p0, size)
        if resize is not None:
            img = img.resize(resize)

        img.save(patch_path)
        patch_list = pd.concat([
            patch_list,
            pd.DataFrame({"x": [p0[0]], "y": [p0[1]], "width": [size[0]], "height": [size[1]], "path": [patch_path]})
        ], ignore_index=True)

    del svs

    # Save patch list
    patch_list.to_csv(str(base) + "list.csv", index=False)
