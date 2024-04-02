#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os
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

    #@staticmethod
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

    # ToDo: No diff.
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

    # ToDo: No diff.
    def patches(self, size=(256, 256), stride=(64, 64)):
        shape = self.image.slide.dimensions

        for j in range(0, shape[1], stride[1]):
            for i in range(0, shape[0], stride[0]):
                yield (i, j), (i + size[0], j + size[1])

    # ToDo: No diff.
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
            location[0]+size[0],
            location[1]+size[1],
        ))

        return np.array(mask, dtype='uint8') * 255

    # ToDo: No diff.
    def extract_img_mask(self, location, size):
        return self.crop_img(location, size), self.crop_mask(location, size)

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
        base: Path, dst: Path,   
        size: (int, int), stride: (int, int), resize: (int, int) = None,
        index: int = 1, region: int = None
):
        """

        :param path_svs:    Path to image svs
        :param path_xml:    Path to contour xml
        :param base:        Base string of output file name
        :param dst          Path to dataset
        :param size:        Patch size
        :param stride:      Patch stride
        :param resize:      Resize extracted patch
        :param index:       Numerical index of annotation
        :param region:      Numerical index of region in the annotation
        :return:        None
        """
        print(index)

        svs = SVS(
            path_svs,
            annotation=path_xml
        )
        basename_without_ext = os.path.splitext(os.path.basename(path_svs))[0]
        
        print(basename_without_ext)
        
        #print(index)
        
        if (index == 1):
            if (basename_without_ext == '57-10'):
                index = -103
            elif(basename_without_ext == '57-11'):
                index = 229
            elif(basename_without_ext == '58-7'):
                index = 230
            elif(basename_without_ext == '58-8'):
                index = 231
            elif(basename_without_ext == '59-19'):
                index = 232
            elif(basename_without_ext == '59-20'):
                index = 233
            elif(basename_without_ext == '59-21'):
                index = 234
            elif(basename_without_ext == '59-22'):
                index = 235
            elif(basename_without_ext == '61-5'):
                index = 236
            elif(basename_without_ext == '61-6'):
                index = 237
            elif(basename_without_ext== '61-9'):
                index = 238
            elif(basename_without_ext == '62-2'):
                index = 239
            elif(basename_without_ext == '62-3'):
                index = 240
            else:
                index = 1
        elif (index == 2):
            if (basename_without_ext == '57-10'):
                index = 1
            elif(basename_without_ext == '57-11'):
                index = 230
            elif(basename_without_ext == '58-7'):
                index = 231
            elif(basename_without_ext == '58-8'):
                index = 232
            elif(basename_without_ext == '59-19'):
                index = 233
            elif(basename_without_ext == '59-20'):
                index = 234
            elif(basename_without_ext == '59-21'):
                index = 235
            elif(basename_without_ext == '59-22'):
                index = 236
            elif(basename_without_ext == '61-5'):
                index = 237
            elif(basename_without_ext == '61-6'):
                index = 238
            elif(basename_without_ext== '61-9'):
                index = 239
            elif(basename_without_ext == '62-2'):
                index = 240
            elif(basename_without_ext == '62-3'):
                index = 241
            else:
                index = 2
        print(index)
        mask = svs.mask(
            vertices=svs.annotation.vertices(index, region, zoom=1.0),
            zoom=1.0
        )
        if(basename_without_ext == '57-10'):
            index = -103
        else:
            index = index -1
        mask_low = svs.mask(
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
            
            if("not" in str(dst)):    
                low_mask = mask_low.crop(box=(
                    p0[0], p0[1],
                    p0[0] + size[0], p0[1] + size[1]
                ))
                
                if np.sum(np.array(low_mask)) < size[0] * size[1]:
                    # Ignore if the mask covers the patch region
                    continue
                elif np.sum(np.array(cropped_mask)) > 0:
                    # Ignore if the mask covers the patch region
                    continue
            else:
                if np.sum(np.array(cropped_mask)) < size[0] * size[1]:
                    # Ignore if the mask does not full-cover the patch region
                    continue
                
            img = svs.crop_img(p0, size)
            if resize is not None:
                img = img.resize(resize)
            #svs.patch_divide(img=img, patch_path=patch_path)
            print(patch_path)
            img.save(patch_path)
        if("not" in str(dst)):
            del svs, mask, mask_low
        else:
            del svs, mask
