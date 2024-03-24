import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

capture = cv2.VideoCapture(0)
poseDetector = PoseDetector()

class Point:
    def __init__(self, x, y, z=0):  # Assuming z is not always needed
        self.x = x
        self.y = y
        self.z = z

def get_point(lmlist, n, inverse):
    if n < len(lmlist):
        lm = lmlist[n]
        return Point(lm[1], inverse - lm[2])
    return None

def cross_mult(A, B, C, D):
    ABx = B.x - A.x
    ABy = B.y - A.y
    CDx = D.x - C.x
    CDy = D.y - C.y
    return ABx * CDy - CDx * ABy

def quick_judge(A, B, C, D):
    return not (max(A.x, B.x) < min(C.x, D.x) or
                max(C.x, D.x) < min(A.x, B.x) or
                max(A.y, B.y) < min(C.y, D.y) or
                max(C.y, D.y) < min(A.y, B.y))

def is_intersecting(A, B, C, D):
    if not quick_judge(A, B, C, D):
        return False
    return cross_mult(C, A, C, D) * cross_mult(C, B, C, D) < 0 and \
           cross_mult(B, C, B, A) * cross_mult(B, D, B, A) < 0

while True:
    success, img = capture.read()
    if success:
        img = poseDetector.findPose(img)
        lmList, bboxInfo = poseDetector.findPosition(img, draw=False)
        if bboxInfo:
            p11 = get_point(lmList, 11, img.shape[0])
            p13 = get_point(lmList, 13, img.shape[0])
            p16 = get_point(lmList, 16, img.shape[0])
            p18 = get_point(lmList, 18, img.shape[0])
            p20 = get_point(lmList, 20, img.shape[0])
            p22 = get_point(lmList, 22, img.shape[0])

            if p11 and p13 and p16 and p18 and p20 and p22:
                intersect1 = is_intersecting(p11, p13, p16, p18)
                intersect2 = is_intersecting(p11, p13, p16, p20)
                intersect3 = is_intersecting(p11, p13, p16, p22)

                if intersect1 or intersect2 or intersect3:
                    print('touch!')

        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
