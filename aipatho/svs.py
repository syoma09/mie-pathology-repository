#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from enum import IntEnum
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from openslide import OpenSlide


class SVS(object):
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
        #print(annotation)
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
        # self.mask = self.__init_mask(zoom=1.0)

    # def __init_mask(self, zoom: float = 1.0) -> Image:
    #     """
    #     Mask image requires about 20GiB RAM.
    #
    #     :return:
    #     """
    #     mask = Image.new("1", self.image.slide.dimensions, 0)
    #     draw = ImageDraw.Draw(mask)
    #
    #     # print("-- Pillow")
    #     # Use last annotation (Layer2) only
    #     _, annot = self.annotation.vertices()[-1]
    #     # Draw region polygons
    #     for _, region in annot:
    #         draw.polygon(
    #             [(x * zoom, y * zoom) for x, y in region],
    #             fill=1
    #         )
    #
    #     return mask
    #
    #     # WARNING: This conversion requires 8 times larger RAM
    #     # mask = np.array(mask, dtype=bool)

    def mask(self, vertices: list = None, zoom: float = 1.0) -> Image:
        """
        Mask image requires about 20GiB RAM.

        :param vertices:    Result of SVSAnnotation.vertices()
        :param zoom:        Zoom ratio of vertices
        :return:
        """

        if vertices is None:
            vertices = self.annotation.vertices()[-1:]

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
        mask = self.mask(
            self.annotation.vertices()[-1:]  # TODO: Layer2 only
        ).crop(box=(
            location[0],
            location[1],
            location[0] + size[0],
            location[1] + size[1]
        ))
        # mask = self.mask[
        #     location[0]:location[0] + size[0],
        #     location[1]:location[1] + size[1],
        # ]
        # mask = self.mask.crop(box=(
        #     location[0],
        #     location[1],
        #     location[0] + size[0],
        #     location[1] + size[1]
        # ))

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
                            [int(float(v.attrib["X"]) * zoom), int(float(v.attrib["Y"]) * zoom)]
                            for v in region.findall('Vertices/Vertex')
                        ])
                    )
                    for region in annot.findall(f'Regions/Region{region}')
                ]
            )
            for annot in self.__root.findall(f'Annotation{index}')
        ]


# def patch_divide(self, img, patch_path):
#     numpy_img = np.array(img)
#     #初期化
#     l=0
#     b_ave=0; g_ave=0; r_ave=0
#
#     for i in range(256):
#         for j in range(256):
#             #画素値[0,0,0]（Black）を除外してピクセルの和とbgrの画素値の合計を計算する
#             if(numpy_img[i,j,0] != 0 or numpy_img[i,j,1] != 0 or numpy_img[i,j,2] != 0 ):
#                 l+=1    #対象となるピクセル数を計算する
#                 #対象となるピクセルの画素値の和を計算する
#                 b_ave=b_ave+numpy_img[i,j,0]
#                 g_ave=g_ave+numpy_img[i,j,1]
#                 r_ave=r_ave+numpy_img[i,j,2]
#
#     #画素値合計をピクセル数で除することでRGBの画素値の平均値を求める
#     l = 3*l
#     bgr = (b_ave+g_ave+r_ave)/l
#     #print(bgr)
#     if bgr < 206:
#         print(patch_path)
#         img.save(patch_path)
#     else:
#         #print(bgr)
#         print('not maiking patch')
def patch_divide(self, img, patch_path):
    numpy_img = np.array(img)
    #初期化
    l=0
    b_ave=0; g_ave=0; r_ave=0

    for i in range(256):
        for j in range(256):
            #画素値[0,0,0]（Black）を除外してピクセルの和とbgrの画素値の合計を計算する
            if(numpy_img[i,j,0] != 0 or numpy_img[i,j,1] != 0 or numpy_img[i,j,2] != 0 ):
                l+=1    #対象となるピクセル数を計算する
                #対象となるピクセルの画素値の和を計算する
                b_ave=b_ave+numpy_img[i,j,0]
                g_ave=g_ave+numpy_img[i,j,1]
                r_ave=r_ave+numpy_img[i,j,2]

    #画素値合計をピクセル数で除することでRGBの画素値の平均値を求める
    l = 3*l
    bgr = (b_ave+g_ave+r_ave)/l
    #print(bgr)
    if bgr < 206:
        print(patch_path)
        img.save(patch_path)
    else:
        #print(bgr)
        print('not maiking patch')


class PatchSaverBase(ABC):
    def __init__(
            self,
            path_svs: Path, path_xml: Path,
            index: int = 1, region: int = None
    ):
        self.path_svs = path_svs
        self.path_xml = path_xml
        # self.index = index
        self.region = region

        # Property place holder, initialized at 1st call of is_foreground()
        self.__svs = None       # SVS() object
        self.__mask_t = None    # Tumor region mask
        self.__mask_s = None    # Sever region mask

        # ToDo: clean from here
        subject = path_svs.stem
        # All tumor mask
        if index == 1:
            indices = {
                '57-10': -103,  '57-11': 229,   '58-7': 230,    '58-8': 231,    '59-19': 232,
                '59-20': 233,   '59-21': 234,   '59-22': 235,   '61-5': 236,    '61-6': 237,
                '61-9': 238,    '62-2': 239,    '62-3': 240
            }
            if subject in indices:
                self.index_t = indices[subject]
            else:
                self.index_t = 1
        elif index == 2:
            indices = {
                '57-10': 1,     '57-11': 230,   '58-7': 231,    '58-8': 232,    '59-19': 233,
                '59-20': 234,   '59-21': 235,   '59-22': 236,   '61-5': 237,    '61-6': 238,
                '61-9': 239,    '62-2': 240,    '62-3': 241
            }
            if subject in indices:
                self.index_t = indices[subject]
            else:
                self.index_t = 2
        # Severe region mask
        if subject == "57-10":
            self.index_s = -103
        else:
            self.index_s = self.index_t - 1
        # ToDo: Clean to here

    @property
    def svs(self) -> SVS:
        if self.__svs is None:
            self.__svs = SVS(self.path_svs, annotation=self.path_xml)

        return self.__svs

    @property
    def mask_t(self) -> Image:
        if self.__mask_t is None:
            self.__mask_t = self.svs.mask(
                vertices=self.svs.annotation.vertices(self.index_t, self.region, zoom=1.0),
                zoom=1.0
            )
        return self.__mask_t

    @property
    def mask_s(self) -> Image:
        if self.__mask_s is None:
            self.__mask_s = self.svs.mask(
                vertices=self.svs.annotation.vertices(self.index_s, self.region, zoom=1.0),
                zoom=1.0
            )
        return self.__mask_s

    def save(
            self, base: Path,
            size: (int, int), stride: (int, int), resize: (int, int) = None
    ):
        """

        :param base:        Base string of output file name
        :param size:        Patch size
        :param stride:      Patch stride
        :param resize:      Resize extracted patch
        :return:            None
        """

        patch_list = pd.DataFrame(
            columns=("x", "y", "width", "height", "path")
        )

        for i, (p0, p1) in enumerate(self.svs.patches(size=size, stride=stride)):
            patch_path = str(base) + f"{i:08}img.png"
            if Path(patch_path).exists():
                continue

            if not self.is_foreground(p0[0], p0[1], size[0], size[1]):
                # Ignore background
                continue

            img = self.svs.crop_img(p0, size)
            if resize is not None:
                img = img.resize(resize)
            # svs.patch_divide(img=img, patch_path=patch_path)
            print(patch_path)
            img.save(patch_path)
            patch_list = pd.concat([
                patch_list,
                pd.DataFrame(
                    {"x": [p0[0]], "y": [p0[1]], "width": [size[0]], "height": [size[1]], "path": [patch_path]})
            ], ignore_index=True)

        # if ("not" in str(dst)):
        #     del svs, mask, mask_low
        # else:
        #     del svs, mask

        # Save patch list
        patch_list.to_csv(str(base) + "list.csv", index=False)

    @abstractmethod
    def is_foreground(self, x: int, y: int, w: int, h: int) -> bool:
        raise NotImplementedError

class PatchSaveFull(PatchSaverBase):
    def __init__(self, path_svs: Path, path_xml: Path, index: int = 1, region: int = None):
        super().__init__(path_svs, path_xml, index, region)

    def is_foreground(self, x: int, y: int, w: int, h: int) -> bool:
        cropped_mask = self.mask_t.crop(box=(
            x, y, x + w, y + h
        ))
        return np.sum(np.array(cropped_mask)) >= w * h


class PatchSaveSevere(PatchSaverBase):
    def __init__(self, path_svs: Path, path_xml: Path, index: int = 1, region: int = None):
        super().__init__(path_svs, path_xml, index, region)

    def is_foreground(self, x: int, y: int, w: int, h: int) -> bool:
        # mask1 = self.mask_t.crop(box=(
        #     x, y, x + w, y + h
        # ))
        mask2 = self.mask_s.crop(box=(
            x, y, x + w, y + h
        ))

        # if np.sum(np.array(low_mask)) < w * h:
        #     # Ignore if the mask covers the patch region
        #     continue
        # elif np.sum(np.array(cropped_mask)) > 0:
        #     # Ignore if the mask covers the patch region
        #     continue
        return np.sum(np.array(mask2)) >= w * h


class TumorMasking(IntEnum):
    FULL = 0    # All tumor region (Layer-1)
    SEVERE = 1  # Only sever region in tumor (Layer-2)


def save_patches(
        path_svs: Path, path_xml: Path,
        base: Path,
        size: (int, int), stride: (int, int), resize: (int, int) = None,
        index: int = None, region: int = None,
        target: TumorMasking = TumorMasking.FULL
):
    if target == TumorMasking.FULL:
        PatchSaveFull(
            path_svs=path_svs, path_xml=path_xml, index=index, region=region
        ).save(
            base=base, size=size, stride=stride, resize=resize
        )
    elif target == TumorMasking.SEVERE:
        PatchSaveSevere(
            path_svs=path_svs, path_xml=path_xml, index=index, region=region
        ).save(
            base=base, size=size, stride=stride, resize=resize
        )

