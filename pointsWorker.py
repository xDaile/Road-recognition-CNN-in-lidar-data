#!/usr/bin/env python3
import parameters

def getXPointCoordinate(point):
    return point[0]
def getYPointCoordinate(point):
    return point[1]
def getZPointCoordinate(point):
    return point[2]
# coordinate I means reflectivity
def getIPointCoordinate(point):
    return point[3]
#Coordinate C means class from this point
def getCPointCoordinate(point):
    return point[4]
#Flag coordinate shows if the point is added or original
def getFLAGPointCoordinate(point):
    return point[5]

def checkPointsBoundary(Point):
    if (
        float(Point[0]) > parameters.xDownBoundary
        and float(Point[0]) < parameters.xUpBoundary
        and float(Point[1]) > parameters.yDownBoundary
        and float(Point[1]) < parameters.yUpBoundary
    ):
        return True
    return False
