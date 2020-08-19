#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from openslide import OpenSlide


# class SVSRegion(object):
#     def __init__(self):


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

    def get_thumbnail(self, shape):
        dims = self.image.slide.dimensions
        zoom = min(shape[0] / dims[0], shape[1] / dims[1])

        # Convert PIL image to Numpy array, swap X and Y axes
        image = np.array(self.image.slide.get_thumbnail(shape)) #.transpose(1, 0, 2)
        # Multiply image zoom ratio to fit the annotation vertices to resized image
        annot = (self.annotation.Vertices * zoom).astype(np.int)

        return image, annot


class SVSImage(object):
    def __init__(self, path):
        self.slide = OpenSlide(str(path))

        # print("level_count: ", self.slide.level_count)
        print("dimensions : ", self.slide.dimensions)
        # img = self.slide.get_thumbnail((512, 512))
        # img = slide.get_thumbnail(slide.dimensions)
        # print(img)
        # img.save("./test.jpg")


class SVSAnnotation(object):
    def __init__(self, path):
        et = ET.parse(path)
        root = et.getroot()

        self._microns_per_pixel = root.attrib["MicronsPerPixel"]
        self._vertices = np.array([
            [int(v.attrib["Y"]), int(v.attrib["X"])]    #, int(v.attrib["Z"])] # Ignore Z
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
