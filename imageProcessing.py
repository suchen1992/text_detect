# -*- coding:utf8 -*-

import cv2


#预处理
def originalBin(img):
    median = cv2.medianBlur(img, 3)
    _, bin1 = cv2.threshold(median, 130, 255, 2)
    _, bin2 = cv2.threshold(bin1, 20, 255, 0)
    return bin2

#黑底白字像素翻转
def colorTransfer(img):
    # Otsu阈值法
    otsuThreshHold, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = img.shape[:2]
    pixSmaller = 0
    pixGreater = 0
    for i in range(height):
        for j in range(width):
            if img[i, j] < otsuThreshHold:
                pixSmaller = pixSmaller + 1
            elif img[i, j] > otsuThreshHold:
                pixGreater = pixGreater + 1
    if float(pixSmaller) / float(pixGreater) > 0.5:
        img = ~img
    return img

# 判断两个框体是否重叠
def isOverlap(box1, box2):
    if box1["x"] + box1["width"] > box2["x"] and \
       box2["x"] + box2["width"] > box1["x"] and \
       box1["y"] + box1["height"] > box2["y"] and \
       box2["y"] + box2["height"] > box1["y"]:
        return True
    else:
        return False

# 将一个box添加进boxList,如果与boxList中元素有重叠,则融合;若无重叠,则添加
def mixBoxes(box, boxList):
    if len(boxList) == 0:
            boxList.append(box)
    else:
        overLapFlag = False
        for boxTemp in boxList:
            if isOverlap(boxTemp, box):
                overLapFlag = True
                x1_temp = min(boxTemp["x"], box["x"])
                y1_temp = min(boxTemp["y"], box["y"])
                x2_temp = max(boxTemp["x"]+boxTemp["width"], box["x"]+box["width"])
                y2_temp = max(boxTemp["y"]+boxTemp["height"], box["y"]+box["height"])
                boxTemp = {"x":x1_temp, "y":y1_temp,
                           "width":x2_temp - x1_temp, "height":y2_temp - y1_temp}
        if not overLapFlag:
            boxList.append(box)
    return boxList

# 对输入的边缘框进行重叠部分的融合
def mixContours(contourList):
    outputContourList = []
    for i in contourList:
        overLapFlag = False
        if len(outputContourList) == 0:
            outputContourList.append(i)
            continue
        for j in outputContourList:
            boxTemp1 = {"x":i["x1"], "y":i["y1"], "width":i["x2"]-i["x1"], "height":i["y2"]-i["y1"]}
            boxTemp2 = {"x":j["x1"], "y":j["y1"], "width":j["x2"]-j["x1"], "height":j["y2"]-j["y1"]}
            if isOverlap(boxTemp1, boxTemp2):
                overLapFlag = True
                j["x1"] = min(i["x1"], j["x1"])
                j["x2"] = max(i["x2"], j["x2"])
                j["y1"] = min(i["y1"], j["y1"])
                j["y2"] = max(i["y2"], j["y2"])
        if not overLapFlag:
            outputContourList.append(i)
    return outputContourList

# 两个box的二维坐标排序的规则
# 默认x坐标排序优先级高于y坐标, 横坐标相差小于10则视为他们在水平方向处在同一位置
def boxSortRule(box1, box2):
    if abs(box1["x"] - box2["x"]) <= 10:
        return box1["y"] > box2["y"]
    elif box1["x"] > box2["x"]:
        return True
    else:
        return False

# 将boxList进行排序
def boxListSort(boxList):
    length = len(boxList)
    for i in range(0, length - 1):
        for j in range(0, length - 1 - i):
            if boxSortRule(boxList[j], boxList[j+1]):
                tempBox = boxList[j]
                boxList[j] = boxList[j+1]
                boxList[j + 1] = tempBox
    return boxList