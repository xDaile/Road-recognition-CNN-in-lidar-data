#!/usr/bin/env python3

import sys
import re
import math
import pandas as pd


def saveToCsv(variable, name):
    print(name)
    Csv = pd.DataFrame(variable)
    Csv.to_csv(name, index=None, header=False)

    return


# booundaries(area) where points will be counted
xDownBoundary = 6
xUpBoundary = 46
yDownBoundary = -10
yUpBoundary = 10

# first argument is file with points
if len(sys.argv) != 3:
    print("Error, use gridMaker.py fileWithPCL folderToSave")
    exit()

# opening and reading the file
m = open(sys.argv[1], "r").readlines()

# first eleven lines are metainfo
m = m[11:]

# index variable
i = 0

# here we will store the points
new_m = []

# selecting the points which we will count stats from
while i < len(m):

    # split items by every whitespace
    s = re.split(r"\s+", m[i])

    # last item in list is only whitespace for newline
    s = s[:4]

    # check if the point is in the area 400x200 metres that we will consider for counting stats
    if (
        float(s[0]) > xDownBoundary
        and float(s[0]) < xUpBoundary
        and float(s[1]) > yDownBoundary
        and float(s[1]) < yUpBoundary
    ):
        new_m.append(s)
    i = i + 1

# creating 400x200 empty array
x = [[[] for i in range(200)] for j in range(400)]

i = 0

# transforming the list of points into grid structure(each cell is 0.1*0.1 cm) -> array 400x200
# 6 to 46 -> 40metres, -10 to 10 ->20 metres, grid is 0.1*0.1 metres, so it will be 400*200 array
#
#    ->>> |--|  <<<- to field like this points are sorted

#                       400metres                            0.1m
# --|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|0.1m
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|   200 metres
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|.........................................|--|--|--|
# --|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|

# creating the grid(2d array) by the x,y coordinates
while i < len(new_m):
    xCoord =int(399- (int((float(new_m[i][0])) * 10-60)))
    yCoord =int(199- (int((float(new_m[i][1])) * 10+100)))
    x[xCoord][yCoord].append(new_m[i])
    i = i + 1
# creating empty 2d 400x200 arrays 6times, for each statistic one array
density = [[0 for i in range(200)] for j in range(400)]
minEL = [[0 for i in range(200)] for j in range(400)]
maxEL = [[0 for i in range(200)] for j in range(400)]
avgELList = [[0 for i in range(200)] for j in range(400)]
avgRefList = [[0 for i in range(200)] for j in range(400)]
stdEL = [[0 for i in range(200)] for j in range(400)]

i = 0
j = 0

# going throught each cell, and counting the stats in each cell
while i < 400:
    while j < 200:
        item = x[i][j]

        # counting the cell density
        density[i][j] += float(len(item))

        # if the cell is empty
        if len(item) == 0.0:
            minEL[i][j] = 0.0
            maxEL[i][j] = 0.0
            avgELList[i][j] = 0.0
            avgRefList[i][j] = 0.0
            density[i][j] = 0.0
            stdEL[i][j] = 0.0

        # points are [x,y,z,r], where x is x coordinate, y is y coordinate, z is elevation and r is reflectivity
        else:
            # setting the minimum and maximum with the first point in the cell(because we want to have something to compare to)
            minEL[i][j] = float(item[0][2])
            maxEL[i][j] = float(item[0][2])

            numOfPointsInCell = len(item)

            # average elevation
            avgELSum = 0.0

            # average reflectivity
            avgRefSum = 0.0

            # standard deviation of elevation(variable for computing)
            std = 0.0

            # going throught the list of points in the cell
            for t in item:
                # actual elevation
                elevation = float(t[2])

                # summary of elevations(z coordinate)
                avgELSum = avgELSum + float(t[2])

                # summary of reflectivity(r coordinate)
                avgRefSum = avgRefSum + float(t[3])

                # finding lowest point
                if elevation < minEL[i][j]:
                    minEL[i][j] = elevation

                # finding highest point
                if elevation > maxEL[i][j]:
                    maxEL[i][j] = elevation

            # counting the mean of elevation and reflectivity
            avgEL = avgELSum / numOfPointsInCell
            avgRef = avgRefSum / numOfPointsInCell

            # againg going throught the points in cell
            # for counting the standard deviation(in previous step we did not have mean - we have to count mean first and then standard deviation)
            for t in item:
                std = std + ((float(t[2]) - avgELSum) * (float(t[2]) - avgELSum))
            std = std / numOfPointsInCell
            std = math.sqrt(std)

            # saving the data
            stdEL[i][j] = std
            avgELList[i][j] = avgEL
            avgRefList[i][j] = avgRef

        j = j + 1
    j = 0
    i = i + 1


# print(density[399])
# print(maxEL[399])
# print(minEL[399])
# print(avgRefList[399])
# print(avgELList[399])
# print(stdEL[399])
# print("x", x[399])

# nameOfFileWithoutEnd = re.split("\.", sys.argv[1])
nameOfFile = re.split(r'\/', sys.argv[1])
var = len(nameOfFile)
nameOfFile = nameOfFile[var - 1]

nameOfFile=re.split(r'\.',nameOfFile)
nameOfFile=nameOfFile[0]

nameOfOutput = re.split(nameOfFile, sys.argv[1])
nameOfOutput = sys.argv[2]


nameOfOutput = nameOfOutput + nameOfFile
# print(nameOfOutput)
# print(type(maxEL[0][0]), maxEL[0][0])
# print(type(minEL[0][0]), minEL[0][0])
# print(type(density[0][0]), density[0][0])
# print(type(avgRefList[0][0]), avgRefList[0][0])
# print(type(avgELList[0][0]), avgELList[0][0])
# print(type(stdEL[0][0]), stdEL[0][0])


saveToCsv(density, nameOfOutput + "_density.csv")
saveToCsv(maxEL, nameOfOutput + "_maxEl.csv")
saveToCsv(minEL, nameOfOutput + "_minEL.csv")
saveToCsv(avgRefList, nameOfOutput + "_meanRef.csv")
saveToCsv(avgELList, nameOfOutput + "_meanEL.csv")
saveToCsv(stdEL, nameOfOutput + "_stdEL.csv")
