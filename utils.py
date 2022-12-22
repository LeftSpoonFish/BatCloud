# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:25:12 2022

@author: BRL
"""

import numpy as np
from PIL import Image

def crop_face(img, box, margin=1):
    """
    Crops facial images based on bounding box. A margin greater than one increases
    the size of the bounding box; less than one decreas the bounding box; and
    equal to one would be the bounding box size.
    """
    x1, y1, x2, y2 = box
    size = int(max(x2-x1, y2-y1) * margin)
    center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
    x1, x2 = center_x-size//2, center_x+size//2
    y1, y2 = center_y-size//2, center_y+size//2
    face = Image.fromarray(img).crop([x1, y1, x2, y2])
    return np.asarray(face)

def cosine_similarity(x, y):
    """
    Measures the similarity betwen two embeddings extracted from facial images.
    If cosine distance is near 0, then vectors have similar orientation and
    close to each other; if cosine distance is near 1, then the vectors differ
    (i.e. orthogonal to each other).
    
    Reference: https://en.wikipedia.org/wiki/Cosine_similarity
    """
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(y, y))

class Point:
    """
    Class to define a point (x, y) in 2-D coordinate system.
    """
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return 'Point(x={},y={})'.format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class Line:
    """
    Class to define a line segment given two points. The distance between
    two points can be calculated using length(). A line can also be
    lengthened or shorted using a scaling factor with scale(n), where n is
    a floating point value.

        ex:
            from line import Line
            from point import Point
            p1 = Point(1, 3)
            p2 = Point(2, 4)
            myline = Line(p1, p2)
            print(p1)
            print(p2)
            print(myline)
            print(myline.length)

    """
    def __init__(self, point_one, point_two):
        self.point_one = point_one
        self.point_two = point_two

    def __str__(self):
        return 'Line(p1:{},p2:{})'.format(self.point_one, self.point_two)

    @property
    def points(self):
        return self.point_one, self.point_two

    @property
    def length(self):
        return ((self.point_one.x - self.point_two.x)**2 + (self.point_one.y - self.point_two.y)**2)**0.5

    def scale(self, factor):
        self.point_one.x, self.point_two.x = Line.scale_dimension(self.point_one.x, self.point_two.x, factor)
        self.point_one.y, self.point_two.y = Line.scale_dimension(self.point_one.y, self.point_two.y, factor)

    @staticmethod
    def scale_dimension(dim1, dim2, factor):
        base_length = dim2 - dim1
        ret1 = dim1 - (base_length * (factor-1) / 2)
        ret2 = dim2 + (base_length * (factor-1) / 2)
        return ret1, ret2
