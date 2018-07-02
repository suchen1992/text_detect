# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cv2
import numpy as np
import imageProcessing as ip
from scipy.stats import norm, mode
import math
import matplotlib.pyplot as plt

# Parameters
AREA_LIM = 0.5
ASPECT_RATIO_LIM = 3.0
OCCUPATION_LIM = (0.23, 0.90)
COMPACTNESS_LIM = (3e-3, 1e-1)
SWT_TOTAL_COUNT = 10
SWT_STD_LIM = 20.0
STROKE_WIDTH_SIZE_RATIO_LIM = 0.02			## Min value
STROKE_WIDTH_VARIANCE_RATIO_LIM = 0.15		## Min value
STEP_LIMIT = 10

def getAspectRatio(width, height):
    return (1.0 * max(width, height)) / (min(width, height) + 1e-4)

def calculateDistance(point1, point2):
    return math.sqrt((point1.getX() - point2.getX())**2+(point1.getY() - point2.getY())**2)

def minBoxesDistance((x1, y1, w1, h1), (x2, y2, w2, h2)):
    distances = []
    box1 = box(x1, y1, w1, h1)
    box2 = box(x2, y2, w2, h2)
    points1 = box1.getPoints()
    points2 = box2.getPoints()
    for point1 in points1:
        for point2 in points2:
            distances.append(calculateDistance(point1, point2))
    return min(distances)

class box(object):
    def __init__(self, x, y, w, h):
        self.points = [point(x, y), point(x, y+h), point(x+w, y), point(x+w, y+h)]

    def getPoints(self):
        return self.points

class point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

class StrokeWidthObject(object):

    def __init__(self, strokeWidth, coordinateMap):
        self.widthList = []
        self.areaList = []
        self.addWidth(strokeWidth)
        self.addArea(coordinateMap)

    def addWidth(self, strokeWidth):
        self.widthList.append(strokeWidth)

    def addArea(self, coordinateMap):
        self.areaList.append(coordinateMap)

    def getWidthList(self):
        return self.widthList

    def getAreaList(self):
        return self.areaList

    def setAreaList(self, areaList):
        self.widthList = areaList

class TextDetection(object):

    def __init__(self, image_path):
        self.imagaPath = image_path
        img = cv2.imread(image_path)
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = rgbImg
        self.final = rgbImg.copy()
        self.height, self.width = self.img.shape[:2]
        self.grayImg = cv2.cvtColor(self.img.copy(), cv2.COLOR_RGB2GRAY)
        self.processedImg = ip.originalBin(self.grayImg) # preprocessed image
        self.cannyImg = self.applyCanny(self.processedImg)
        self.sobelX = cv2.Sobel(self.processedImg, cv2.CV_64F, 1, 0, ksize=-1)
        self.sobelY = cv2.Sobel(self.processedImg, cv2.CV_64F, 0, 1, ksize=-1)
        self.stepsX = self.sobelY.astype(int)  ## Steps are inversed!! (x-step -> sobelY)
        self.stepsY = self.sobelX.astype(int)
        self.magnitudes = np.sqrt(self.stepsX * self.stepsX + self.stepsY * self.stepsY)
        self.gradsX = self.stepsX / (self.magnitudes + 1e-10)
        self.gradsY = self.stepsY / (self.magnitudes + 1e-10)

    def applyCanny(self, img, sigma=0.33):
        v = np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(img, lower, upper)

    def getStrokes(self, (x, y, w, h)):
        strokeWidths = np.array([[np.Infinity, np.Infinity]])
        for i in xrange(y, y + h):
            for j in xrange(x, x + w):
                if self.cannyImg[i, j] != 0:
                    gradX = self.gradsX[i, j]
                    gradY = self.gradsY[i, j]

                    prevX, prevY, prevX_opp, prevY_opp, stepSize = i, j, i, j, 0

                    """
                    if DIRECTION == "light":
                        go, go_opp = True, False
                    elif DIRECTION == "dark":
                        go, go_opp = False, True
                    else:
                        go, go_opp = True, True
                    """
                    go, go_opp = True, True

                    strokeWidth = np.Infinity
                    strokeWidth_opp = np.Infinity
                    while (go or go_opp) and (stepSize < STEP_LIMIT):
                        stepSize += 1

                        if go:
                            curX = np.int(np.floor(i + gradX * stepSize))
                            curY = np.int(np.floor(j + gradY * stepSize))
                            if curX <= y or curY <= x or curX >= y + h or curY >= x + w:
                                go = False
                            if go and ((curX != prevX) or (curY != prevY)):
                                try:
                                    if self.cannyImg[curX, curY] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX, curY] + gradY * -self.gradsY[curX, curY]) < np.pi / 2.0:
                                            strokeWidth = int(np.sqrt((curX - i) ** 2 + (curY - j) ** 2))

                                            go = False
                                except IndexError:
                                    go = False

                                prevX = curX
                                prevY = curY

                        if go_opp:
                            curX_opp = np.int(np.floor(i - gradX * stepSize))
                            curY_opp = np.int(np.floor(j - gradY * stepSize))
                            if curX_opp <= y or curY_opp <= x or curX_opp >= y + h or curY_opp >= x + w:
                                go_opp = False
                            if go_opp and ((curX_opp != prevX_opp) or (curY_opp != prevY_opp)):
                                try:
                                    if self.cannyImg[curX_opp, curY_opp] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX_opp, curY_opp] + gradY * -self.gradsY[curX_opp, curY_opp]) < np.pi / 2.0:
                                            strokeWidth_opp = int(np.sqrt((curX_opp - i) ** 2 + (curY_opp - j) ** 2))

                                            go_opp = False

                                except IndexError:
                                    go_opp = False

                                prevX_opp = curX_opp
                                prevY_opp = curY_opp

                    strokeWidths = np.append(strokeWidths, [(strokeWidth, strokeWidth_opp)], axis=0)

        strokeWidths_opp = np.delete(strokeWidths[:, 1], np.where(strokeWidths[:, 1] == np.Infinity))
        strokeWidths = np.delete(strokeWidths[:, 0], np.where(strokeWidths[:, 0] == np.Infinity))
        return strokeWidths, strokeWidths_opp

    def getStrokeProperties(self, strokeWidths):
        if len(strokeWidths) == 0:
            return (0, 0, 0, 0, 0, 0)
        try:
            mostStrokeWidth = mode(strokeWidths, axis=None)[0][0]  ## Most probable stroke width is the most one
            mostStrokeWidthCount = mode(strokeWidths, axis=None)[1][0]  ## Most probable stroke width is the most one
        except IndexError:
            mostStrokeWidth = 0
            mostStrokeWidthCount = 0
        try:
            mean, std = norm.fit(strokeWidths)
            xMin, xMax = int(min(strokeWidths)), int(max(strokeWidths))
        except ValueError:
            mean, std, xMin, xMax = 0, 0, 0, 0
        return (mostStrokeWidth, mostStrokeWidthCount, mean, std, xMin, xMax)

    def detect(self):
        kernel = np.ones((6, 6), np.uint8)
        closed = cv2.erode(self.processedImg, kernel)
        _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 检测文字轮廓
        strokeObjectList = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 30 or h < 30:
                continue
            if w * h > self.grayImg.shape[0] * self.grayImg.shape[1] * AREA_LIM:
                continue
            if getAspectRatio(w, h) > ASPECT_RATIO_LIM:
                continue

            strokeWidths, strokeWidths_opp = self.getStrokes((x, y, w, h))

            strokeWidth, strokeWidthCount, mean, std, xMin, xMax = self.getStrokeProperties(strokeWidths)
            strokeWidth_opp, strokeWidthCount_opp, mean_opp, std_opp, xMin_opp, xMax_opp = self.getStrokeProperties(strokeWidths_opp)
            if strokeWidthCount_opp > strokeWidthCount:  ## Take the strokeWidths with max of counts strokeWidth (most probable one)
                strokeWidths = strokeWidths_opp
                strokeWidth = strokeWidth_opp
                strokeWidthCount = strokeWidthCount_opp
                mean = mean_opp
                std = std_opp
                xMin = xMin_opp
                xMax = xMax_opp

            #cv2.imwrite("/home/suchen/桌面/temp1/"+str(i)+".jpg", self.img[y: y+h, x: x+w])
            if len(strokeWidths) < SWT_TOTAL_COUNT:
                continue

            if std > SWT_STD_LIM:
                continue

            strokeWidthSizeRatio = strokeWidth / (1.0 * max(w, h))
            if strokeWidthSizeRatio < STROKE_WIDTH_SIZE_RATIO_LIM:
                continue

            strokeWidthVarianceRatio = (1.0 * strokeWidth) / (std ** std)
            if strokeWidthVarianceRatio > STROKE_WIDTH_VARIANCE_RATIO_LIM:
                if len(strokeObjectList) == 0:
                    tempStrokeObject = StrokeWidthObject(strokeWidth, {"x":x , "y":y , "w":w , "h":h})
                    strokeObjectList.append(tempStrokeObject)
                else:
                    addNewFlag = True
                    for strokeObject in strokeObjectList:
                        meanStrokeWidth = mode(strokeObject.getWidthList(), axis=None)[0][0]
                        if max(meanStrokeWidth, strokeWidth) / min(meanStrokeWidth, strokeWidth) < 2:
                            strokeObject.addWidth(strokeWidth)
                            strokeObject.addArea({"x":x , "y":y , "w":w , "h":h})
                            addNewFlag = False
                    if addNewFlag:
                        tempStrokeObject = StrokeWidthObject(strokeWidth, {"x": x, "y": y, "w": w, "h": h})
                        strokeObjectList.append(tempStrokeObject)

            """
                tempImg = self.img[y: y+h, x: x+w]
                #cv2.imwrite("/home/suchen/桌面/temp2/"+str(i)+".jpg", tempImg)
                #print "{}'s std = {}".format(i, strokeWidthVarianceRatio)
                cv2.imshow("temp", tempImg)
                print "{}'s strike = {}".format(i, strokeWidth)
                cv2.waitKey(0)
            i += 1
            """
        i = 0
        for strokeObject in strokeObjectList:
            strokeObject.setAreaList(ip.boxListSort(strokeObject.getAreaList()))
            areaList = strokeObject.getAreaList()
            mergeCoordinate = {}
            t = 0
            for areaCoordinate in areaList:
                """# 检测轮廓图数量是否完整
                x, y, w, h = areaCoordinate["x"], areaCoordinate["y"], areaCoordinate["w"], areaCoordinate["h"]
                tempImg = self.img[y: y + h, x: x + w]
                cv2.imwrite("/home/suchen/桌面/temp2/"+str(t)+".jpg", tempImg)
                t += 1
                continue"""
                if len(mergeCoordinate) == 0:
                    mergeCoordinate = areaCoordinate
                    continue
                else:
                    x, y, w, h = areaCoordinate["x"], areaCoordinate["y"], areaCoordinate["w"], areaCoordinate["h"]
                    mergeX, mergeY, mergeW, mergeH = mergeCoordinate["x"], mergeCoordinate["y"], mergeCoordinate["w"], mergeCoordinate["h"]
                    """
                    # 检测面积扩大问题
                    if len(areaList) > 6 and areaCoordinate == areaList[10]:
                        tempImg = self.img[y: y + h, x: x + w]
                        cv2.imshow("temp", tempImg)
                        cv2.waitKey(0)
                        temp2 = self.img[mergeY: mergeY+mergeH, mergeX:mergeX+mergeW]
                        cv2.imshow("t2", temp2)
                        cv2.waitKey(0)
                        print "merge coordinate is x:{}, y:{}, w:{}, h:{}".format(mergeCoordinate["x"],mergeCoordinate["y"],mergeCoordinate["w"],mergeCoordinate["h"])
                        print "tempImg coordinate is x:{}, y:{}, w:{}, h:{}".format(x,y,w,h)
                        boxesDistance = minBoxesDistance((mergeX, mergeY, mergeW, mergeH), (x, y, w, h))
                        print(boxesDistance)
                    """

                    if minBoxesDistance((mergeX, mergeY, mergeW, mergeH), (x, y, w, h)) <= 2 * min(w, h):
                        mergeCoordinate["x"] = min(mergeCoordinate["x"], x)
                        mergeCoordinate["y"] = min(mergeCoordinate["y"], y)
                        mergeCoordinate["w"] = max(mergeCoordinate["x"] + mergeCoordinate["w"], x + w) - mergeCoordinate["x"]
                        mergeCoordinate["h"] = max(mergeCoordinate["y"] + mergeCoordinate["h"], y + h) - mergeCoordinate["y"]
            """# roi区域"""
            x, y, w, h = mergeCoordinate["x"], mergeCoordinate["y"], mergeCoordinate["w"], mergeCoordinate["h"]
            tempImg = self.img[y: y+h, x: x+w]
            cv2.imshow("temp", tempImg)
            cv2.waitKey(0)



if __name__=="__main__":
    #18 6
    td = TextDetection("/home/suchen/桌面/originTop100/6.jpg")
    td.detect()
















