import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
import os

# capture = cv2.VideoCapture('dance-short.mp4')
capture = cv2.VideoCapture(0)
# 尚未镜像，看到的是右手就是右手
poseDetector = PoseDetector()
handDetector = HandDetector(maxHands=2, detectionCon=0.8)
posList = []


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# get_lm_num(lmList, n, inverse)获取目标lm的坐标并返回一个Point类
def get_lm_num(lmlist, n, inverse):
    '''
    :param lmlist: 当前帧所有lm列表
    :param n: 目标lm的序号（0-32）
    :param inverse: img.shape[0]
    :return: 目标lm(x,y,z)
    '''
    # 由于unity和python的坐标系不同需要倒置y轴数值
    for lm in lmlist:
        if lm[0] == n:
            x = lm[1]
            y = inverse-lm[2]
            z = lm[3]
            roi = Point(x, y, z)
            return roi


# cross_mult(A,B,C,D)计算AB,CD两个向量的叉乘值并返回对应值
def cross_mult(A, B, C, D):
    '''
    :param A: Point类
    :param B: Point类
    :param C: Point类
    :param D: Point类
    :return: AB与CD叉乘结果
    '''
    ABx = B.x - A.x
    ABy = B.y - A.y
    CDx = D.x - C.x
    CDy = D.y - C.y
    return ABx * CDy - CDx * ABy


# quick_judge(A,B,C,D)快速排斥,T-无法判断,F-一定不香蕉
def quick_judge(A, B, C, D):
    if (max(A.x,B.x) < min(C.x, D.x) or
            max(C.x, D.x) < min(A.x, B.x) or
            max(A.y, B.y) < min(C.y, D.y) or
            max(C.y, D.y) < min(A.y, B.y)):
        return False
    else:
        return True


# banana(A,B,C,D)判断线段AB,CD是否香蕉并返回一个布尔值
def banana(A, B, C, D):
    '''
    :param A: Point类
    :param B: Point类
    :param C: Point类
    :param D: Point类
    :return: Bool型表示AB与CD是否香蕉
    '''
    if not quick_judge(A, B, C, D):
        return False
    CAxCD = cross_mult(C, A, C, D)
    CBxCD = cross_mult(C, B, C, D)
    BCxBA = cross_mult(B, C, B, A)
    BDxBA = cross_mult(B, D, B, A)
    if (CAxCD * CBxCD < 0) and (BCxBA * BDxBA < 0):
        return True
    else:
        return False


# 逐帧处理
while True:
    success, img = capture.read()
    # img = cv2.flip(img, flipCode=1)
    # 镜像语句
    lmList, bboxInfo = poseDetector.findPosition(img)
    # lmHandList, bboxHandInfo = handDetector.findHands(img)
    if bboxInfo:

        # 选了左上臂，触碰到会打印touch
        p11 = get_lm_num(lmList, 11, img.shape[0])
        p13 = get_lm_num(lmList, 13, img.shape[0])
        p16 = get_lm_num(lmList, 16, img.shape[0])

        p18 = get_lm_num(lmList, 18, img.shape[0])
        p20 = get_lm_num(lmList, 20, img.shape[0])
        p22 = get_lm_num(lmList, 22, img.shape[0])

        banana1 = banana(p11,p13,p16,p18)
        banana2 = banana(p11,p13,p16,p20)
        banana3 = banana(p11,p13,p16,p22)

        if banana1 or banana2 or banana3:
            print('touch!')
    cv2.imshow("image", img)
    cv2.waitKey(1)

