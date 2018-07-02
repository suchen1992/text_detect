# -*- coding:utf8 -*-
import cv2
import numpy as np
import math

# 八方向联通域判断
angleList = [np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
             np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
             np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]]),
             np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]),
             np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]),
             np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
             np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]]),
             np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]]),
             np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]]),
             np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]]),
             np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]]),
             np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]]),
             np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1]]),
             np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0]]),
             np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]]),
             np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0]]),
             np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0]]),
             np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])]

# 根据方向角计算方向值移动倍率
def calculateXYRate(angle):
    # 方向角为0 或 π的时候,垂直方向不移动
    if angle == 0 or angle == math.pi:
        rateX = 1 if angle == 0 else -1
        rateY = 0
    # 方向角为π/2 或 -π/2时,水平方向不移动
    elif angle == math.pi / 2 or angle == -math.pi / 2:
        rateX = 0
        rateY = 1 if angle < 0 else -1
    # 方向角为π/2+π/4 或 -π/4
    elif angle == 3 * math.pi / 4 or angle == -math.pi / 4 \
            or angle == 5 * math.pi / 4 or angle == math.pi / 4:
        rateX = 1 if angle == math.pi / 4 or angle == -math.pi / 4 else -1
        rateY = 1 if angle == -math.pi / 4 or angle == 5 * math.pi / 4 else -1
    # 方向角为
    else:
        rateX = 1 if angle > 0 and angle < math.pi/2 else -1
        rateY = math.tan(angle)
    return rateY, rateX

# 方向角判断
def judgeAngle(image, x, y):
    arr = np.array([[image[x-1, y-1], image[x-1, y], image[x-1, y+1]],
                    [image[x, y-1], image[x, y], image[x, y+1]],
                    [image[x+1, y-1], image[x+1, y], image[x+1, y+1]]])
    angle1, angle2 = 0,0
    for i in range(len(angleList)):
        resultMatrix = np.multiply(arr, angleList[i]).reshape((1,9))
        if np.sum(resultMatrix)/255 != 3:
            continue
        index = i + 1
        if index == 1 or index == 9 or index == 10:
            angle1, angle2 = 0, math.pi
        elif index == 2 or index == 7 or index == 8:
            angle1, angle2 = math.pi/2, -math.pi/2
        elif index == 3 or index == 6:
            angle1, angle2 = 3*math.pi/4, -math.pi/4
        elif index == 4 or index == 5:
            angle1, angle2 = math.pi/4, 5*math.pi/4
        elif index == 11 or index == 14:
            angle1, angle2 = -3*math.pi/8, 5*math.pi/8
        elif index == 12 or index == 13:
            angle1, angle2 = 3*math.pi/8, 11*math.pi/8
        elif index == 15 or index == 18:
            angle1, angle2 = math.pi/8, 9*math.pi/8
        else:
            angle1, angle2 = 7*math.pi/8, -math.pi/8
    return angle1, angle2

# 查找对应点
def findSymmetryPoint(image, x, y, angle1, angle2, height, width):
    threshold = max(height, width)/5
    symmetryPointX = 0
    symmetryPointY = 0
    rateX1, rateY1 = calculateXYRate(angle1)
    rateX2, rateY2 = calculateXYRate(angle2)
    distance = 0
    flag = False
    for t in range(threshold):
        # 判断坐标是否越界 判断该点是否有值
        x1Interval = int(round(rateX1 * (t+1)))
        y1Interval = int(round(rateY1 * (t+1)))
        x2Interval = int(round(rateX2 * (t+1)))
        y2Interval = int(round(rateY2 * (t+1)))
        if (x + x1Interval > 0 and x + x1Interval < height-1) and (y + y1Interval > 0 and y + y1Interval < width-1) and image[x + x1Interval, y + y1Interval] != 0:
            symmetryPointX = x + x1Interval
            symmetryPointY = y + y1Interval
            symAngle1, symAngle2 = judgeAngle(image, symmetryPointX, symmetryPointY)
            if (abs(angle1) >=  abs(symAngle1) - math.pi/6 and abs(angle1) <= abs(symAngle1) + math.pi/6) \
                    or (abs(angle1) >=  abs(symAngle2) - math.pi/6 and abs(angle1) <= abs(symAngle2) + math.pi/6):
                flag = True
            break
        if (x + x2Interval > 0 and x + x2Interval < height-1) and (y + y2Interval > 0 and y + y2Interval < width-1) and image[x + x2Interval, y + y2Interval] != 0:
            symmetryPointX = x + x2Interval
            symmetryPointY = y + y2Interval
            symAngle1, symAngle2 = judgeAngle(image, symmetryPointX, symmetryPointY)
            if (abs(angle2) >=  abs(symAngle1) - math.pi/6 and abs(angle2) <= abs(symAngle1) + math.pi/6) \
                    or (abs(angle2) >=  abs(symAngle2) - math.pi/6 and abs(angle2) <= abs(symAngle2) + math.pi/6):
                flag = True
            break
    if flag:
        distance = np.sqrt(np.square(x - symmetryPointX) + np.square(y - symmetryPointY))
    return flag, distance

# 主方法 计算所给图中字符笔画宽度
def calculateFontSize(img):
    if img is None:
        return 0.01
    cpy = cv2.GaussianBlur(img, (3, 3), 0)
    dst = cv2.Canny(cpy, 10, 30)
    h, w = dst.shape[:2]
    distanceSum = 0.0
    legalPointNumber = 0.0001 # 避免除数为0
    for j in range(h):
        for i in range(w):
            # 图片第一圈像素默认跳过
            if (i == 0 or i == w-1) or (j == 0 or j == h-1):
                continue
            # 考察点无像素则跳过
            if dst[j, i] == 0:
                continue
            angle1, angle2 = judgeAngle(dst, j, i)
            flag, distance = findSymmetryPoint(dst, j, i, angle1, angle2, h, w)
            if flag:
                distanceSum += distance
                legalPointNumber += 1
    return distanceSum/legalPointNumber