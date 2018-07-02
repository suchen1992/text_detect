# -*- coding:utf8 -*-

import cv2
import os
import imageProcessing as ip

imgPath = "/home/suchen/桌面/ocr_files/dataset/textROI/"
output = "/home/suchen/桌面/temp3/"
list = os.listdir(imgPath)
for imgName in list:
    img = cv2.imread(imgPath+imgName,0)
    cv2.imwrite(output+imgName, ip.colorTransfer(img))