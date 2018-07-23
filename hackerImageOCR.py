# -*- coding:utf8 -*-

import cv2
import numpy as np
import imageProcessing
import SWTCode

# 这个路径是保存预处理之后的图像的路径，因为ocr识别软件需要自己从路径中读图，所以事先设定这个地址
roiTempPath = "/home/suchen/桌面/out1/temp.jpg"

# imagePath 需要处理的图片路径
# screenShotFlag 传来的图片是否是截屏大图 True 为大图 False 为小图
def ocrApi(imagePath, screenShotFlag=True):
    image = cv2.imread(imagePath, 1)    # 三通道原始彩色图像
    text = ""   # 保存识别后的字符串
    grayOriginImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 原始灰度图
    grayImg = imageProcessing.originalBin(grayOriginImg)   # 黑客图预处理流程
    if screenShotFlag:
        kernel = np.ones((6, 6), np.uint8)
        closed = cv2.erode(grayImg, kernel)
        _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 检测文字轮廓
        h, w = grayImg.shape[:2]
        contourList = []    # 保存融合后轮廓图的平均笔画宽度、坐标与尺寸
        boxList = []    # box指代画的框，boxList用于存放得到的框的集合
        # 遍历轮廓图
        i = 0
        for contour in contours:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            # 尺寸过小的图片人为抛弃
            if w1 < 30 or h1 < 30:
                continue
            # 尺寸与整张图相同的轮廓抛弃
            if w1 * h1 == h * w:
                continue
            cv2.imwrite("/home/suchen/桌面/temp1/" + str(i) + ".jpg", grayImg[y1: y1 + h1, x1: x1 + w1])
            #i += 1
            #continue
            boxTemp = {"x":x1, "y":y1, "width":w1, "height": h1}
            # 将当前的轮廓图通过判断是否重叠融合进box数组中，
            imageProcessing.mixBoxes(boxTemp, boxList)

        # 二维坐标排序
        imageProcessing.boxListSort(boxList)

        # 遍历得到的框，调用笔画宽度算法，融合轮廓，得到最终的感兴趣区域
        for box in boxList:
            x1, y1, w1, h1 = box["x"], box["y"], box["width"], box["height"] # 根据box获取坐标与尺寸
            boundingImg = grayImg[y1:y1 + h1, x1:x1 + w1]   # 根据坐标与尺寸获取文字框图
            fontSize = SWTCode.calculateFontSize(boundingImg)   # 计算当前box内的字符笔画宽度
            if len(contourList) == 0:   # 若轮廓数组为空则直接添加
                dict = {"fontSize": fontSize, "x1": x1, "y1": y1, "x2": x1 + w1, "y2": y1 + h1}
                contourList.append(dict)
            else:
                for t in range(len(contourList)):
                    contourSize = contourList[t]["fontSize"]
                    if abs(fontSize - contourSize) <= 1.5:  # 若当前box内字符笔画宽度与遍历轮廓中的平均笔画宽度相差小于1.5表示是同一字体
                        # 若横向或纵向相差1.2倍的字符宽度，则默认属于同一个roi区域，进行轮廓融合
                        if abs(contourList[t]["x1"] - (x1 + w1)) <= 1.2 * w1 or abs(x1 - contourList[t]["x2"]) <= 1.2 * w1 \
                                or abs(contourList[t]["y1"] - (y1 + h1)) <= 1.2 * h1 or abs(y1 - contourList[t]["y2"]) <= 1.2 * h1:
                            contourList[t]["x1"] = min(contourList[t]["x1"], x1)
                            contourList[t]["x2"] = max(contourList[t]["x2"], x1 + w1)
                            contourList[t]["y1"] = min(contourList[t]["y1"], y1)
                            contourList[t]["y2"] = max(contourList[t]["y2"], y1 + h1)
                            break
                    else:
                        # 默认最多保存5个roi轮廓
                        if len(contourList) >= 5:
                            continue
                        # 如果没有笔画宽度相近的轮廓，则单另属于一个roi区域
                        dict = {"fontSize": fontSize, "x1": x1, "y1": y1, "x2": x1 + w1, "y2": y1 + h1}
                        contourList.append(dict)
        # 将得到的roi区域进行融合，判断重叠的部分则融合为一个roi
        contourList = imageProcessing.mixContours(contourList)
        for t in range(len(contourList)):
            x1 = contourList[t]["x1"]
            y1 = contourList[t]["y1"]
            x2 = contourList[t]["x2"]
            y2 = contourList[t]["y2"]
            imgTemp = grayImg[y1:y2, x1:x2]
            cv2.rectangle(image, (x1, y1), (x2, y2),
                          (0, 255, 0), 3)
            cv2.imshow("window", image)
            cv2.waitKey(0)
            # 将得到的roi图保存再进行识别
            #cv2.imwrite(roiTempPath, imgTemp)
            #text += tesseractApi.tesseractOCR(roiTempPath)
    else:
        # 按照黑客图预处理的识别结果
        cv2.imwrite(roiTempPath, grayImg)
        # 颜色翻转再预处理结果
        transferImg = imageProcessing.colorTransfer(grayOriginImg)
        # 若进行了翻转，则将翻转后的识别结果也进行保存，防止漏检
        if (transferImg[0][0] != grayOriginImg[0][0]):
            print(False)
            grayImg = imageProcessing.originalBin(transferImg)
            cv2.imwrite(roiTempPath, grayImg)
        # 普通的网站插图，使用大津法进行预处理
        _, grayImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(roiTempPath, grayImg)

    print(text) # 打印识别出的字符串
    return text

if __name__=="__main__":
    # 调用例子
    #ocrApi("/home/suchen/桌面/ocr_files/dataset/textROI/009-1.jpg", False)
    ocrApi("/home/suchen/桌面/ocr_files/dataset/ocrDataset/14.jpg", True)