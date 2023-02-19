import cv2
import numpy as np


path = '1917.bmp'
read = cv2.imread(path)

img = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY) 
img = cv2.resize(img, (1024, 1024))
img2 = img.copy()


def Gradient (img):
    gray = img
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradient_X = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_Y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = (gradient_X ** 2 - gradient_Y ** 2)
    cv2.imshow('grad', grad)


def Gauss (x):
    global gaker, img, img2
    pos = cv2.getTrackbarPos('gauss', 'butt')
    print (pos)
    if pos > 0:
        for i in range(0, pos):
            print (i)
            if i == 0:
                Gauss = cv2.GaussianBlur(img2, (gaker, gaker), 0)
                img = cv2.addWeighted(img, 0, Gauss, 1, 0)
            elif i > 0:
                Gauss = cv2.GaussianBlur(img, (gaker, gaker), 0)
                img = cv2.addWeighted(img, 0, Gauss, 1, 0)

    elif pos == 0:
        img = img2

gaker = 0
def Gauss_kern (x):
    global gaker
    pos = cv2.getTrackbarPos('gauss_ker', 'butt')
    gaker = pos


def stand_blur (val):
    global img, stdker
    pos = cv2.getTrackbarPos('stand_blr', 'butt')
    if pos > 0:
        for i in range(0, pos):
            if i == 0:
                Stand = cv2.blur(img2, (stdker, stdker))
                img = cv2.addWeighted(img, 0, Stand, 1, 0)
            elif i > 0:
                Stand = cv2.blur(img, (stdker, stdker))
                img = cv2.addWeighted(img, 0, Stand, 1, 0)
    elif pos == 0:
        img = img2

stdker = 0
def stand_kern (val):
    global stdker
    pos = cv2.getTrackbarPos('stand_kern', 'butt')
    stdker = pos


def median_blur (val):
    global img, medpos
    pos = cv2.getTrackbarPos('median', 'butt')
    if pos > 0:
        for i in range(0, pos):
            if i == 0:
                Median = cv2.medianBlur(img2, medpos)
                img = cv2.addWeighted(img, 0, Median, 1, 0)
            elif i > 0:
                Median = cv2.medianBlur(img, medpos)
                img = cv2.addWeighted(img, 0, Median, 1, 0)
    elif pos == 0:
        img = img2



medpos = 0
def median_kern (x):
    global medpos
    pos = cv2.getTrackbarPos('median_kern', 'butt')
    medpos = pos


def erode (x):
    global img, img2, pos
    poss = cv2.getTrackbarPos('erode', 'butt')
    kernel = np.ones((pos, pos), 'uint8')
    img = cv2.erode(img2, kernel, iterations=poss)


pos = 0
def erode_kern (x):
    global pos
    poss = cv2.getTrackbarPos('erode_kern', 'butt')
    pos = poss


def sharp (x):
    global img, img2
    pos = cv2.getTrackbarPos('sharp', 'butt')
    kernel = np.array([[-1,-1,-1],
                       [-1,9,-1],
                       [-1,-1,-1]])


    if pos > 0:
        for i in range(0, pos):
            if i == 0:
                Sharp = cv2.filter2D(img2, -1, kernel)
                img = Sharp
            elif i > 0:
                Sharp = cv2.filter2D(img, -1, kernel)
                img = Sharp
    elif pos == 0:
        img = img2


tres1 = 0
tres2 = 0
def canny (x):
    global img, img2, tres1, tres2
    pos = cv2.getTrackbarPos('canny', 'butt')
    if pos >0 :
        for i in range(0, pos):
            if i == 0:
                Canny = cv2.Canny(img2, tres1, tres2)
                img = Canny
            elif i > 0:
                Canny = cv2.Canny(img, tres1, tres2)
                img = Canny
    if pos == 0:
        img = img2


def up_tr1 (x):
    global tres1
    pos = cv2.getTrackbarPos('tr1', 'butt')
    tres1 = pos


def up_tr2 (x):
    global tres2
    pos = cv2.getTrackbarPos('tr2', 'butt')
    tres2 = pos

tress1 = 0
tress2 = 0

def create (x):
    global tress1,tress2
    tress1 = cv2.getTrackbarPos('tress1', 'butt')
    tress2 = cv2.getTrackbarPos('tress2', 'butt')

def treshold (x):
    global img, img2, tress1, tress2
    pos = cv2.getTrackbarPos('treshold', 'butt')
    if pos >0 :
        for i in range(0, pos):
            if i == 0:
                _, tresh = cv2.threshold(img2, tress1, tress2, cv2.THRESH_TOZERO)
                img = tresh
            elif i > 0:
                _, tresh = cv2.threshold(img, tress1, tress2, cv2.THRESH_TOZERO)
                img = tresh
    if pos == 0:
        img = img2

def clahee (x):
    global img, img2
    pos = cv2.getTrackbarPos('clahe', 'butt')
    if pos >0 :
        for i in range(0, pos):
            if i == 0:
                clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
                dst = clahe.apply(img)
                img = dst
            elif i > 0:
                clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
                dst = clahe.apply(img)
                img = dst
    if pos == 0:
        img = img2

cv2.namedWindow('butt')
cv2.resizeWindow('butt', 400, 1000)

cv2.createTrackbar('clahe', 'butt', 0, 1, clahee)


cv2.createTrackbar('treshold', 'butt', 0, 20, treshold)
cv2.createTrackbar('tress1', 'butt', 0, 255, create)
cv2.createTrackbar('tress2', 'butt', 0, 255, create)


cv2.createTrackbar('canny', 'butt', 0 , 20, canny)
cv2.createTrackbar('tr1', 'butt', 0, 100, up_tr1)
cv2.createTrackbar('tr2', 'butt', 0, 200, up_tr2)


cv2.createTrackbar('sharp', 'butt', 0, 20, sharp)
cv2.createTrackbar('stand_blr', 'butt', 0, 20, stand_blur)
cv2.createTrackbar('stand_kern', 'butt', 0, 20, stand_kern)


cv2.createTrackbar('gauss', 'butt', 0, 100, Gauss)
cv2.createTrackbar('gauss_ker', 'butt', 0, 20, Gauss_kern)


cv2.createTrackbar('erode', 'butt', 0, 20, erode)
cv2.createTrackbar('erode_kern', 'butt', 0, 20, erode_kern)


cv2.createTrackbar('median', 'butt', 0, 20, median_blur)
cv2.createTrackbar('median_kern', 'butt', 0, 20, median_kern)

# if imgc[x, down] < 90 and downsum != 1:
#     cv2.circle(img3, (x,down), 1, (255, 0, 0))
#     downsum +=1
# if imgc[left, y] < 90 and leftsum != 1:
#     cv2.circle(img3, (left,y), 1, (255, 0, 0))
#     leftsum += 1
# if imgc[right, y] < 90 and rightsum != 1:
#     cv2.circle(img3, (right, y), 1, (255, 0, 0))
#     rightsum += 1
# def get_shape (dot, img):
#     global xnew
#     imgc = img.copy()
#     cv2.imshow('filtedbyvaas', imgc)
# 
#     # высота вверх - 1 вниз +1
#     # длина влево - 1 вправо +1
#     x = dot[0]
#     y = dot[1]
#     upsum = 0
#     downsum = 0
#     leftsum = 0
#     rightsum = 0
#     for i in range(0, 100):
#         up = y - 1
#         down = y + 1
#         left = x - 1
#         right = x + 1

while (True):

    cv2.namedWindow('image')
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('g'):
        Gradient(img)
    if k == ord('s'):
        print ('save')
        img2 = img
    if k == ord('w'):
        img = cv2.imread(path)
        img = cv2.resize(img, (1024,1024))
        img2 = img.copy()
        cv2.setTrackbarPos('canny', 'butt', 0)
        cv2.setTrackbarPos('tr1', 'butt', 0)
        cv2.setTrackbarPos('tr2', 'butt', 0)
        cv2.setTrackbarPos('sharp', 'butt', 0)
        cv2.setTrackbarPos('stand_blr', 'butt', 0)
        cv2.setTrackbarPos('stand_kern', 'butt', 0)
        cv2.setTrackbarPos('gauss', 'butt', 0)
        cv2.setTrackbarPos('gauss_ker', 'butt', 0)
        cv2.setTrackbarPos('erode', 'butt', 0)
        cv2.setTrackbarPos('erode_kern', 'butt', 0)
        cv2.setTrackbarPos('median', 'butt', 0)
        cv2.setTrackbarPos('median_kern', 'butt', 0)

    if k == ord('a'):
        cv2.imwrite('savecanny.png', img2)
        img = cv2.imread('savecanny.png')
    if k == ord('t'):
        img = cv2.imread('savecanny.png')
        img2 = img.copy()
    if k == ord('m'):
        img = 255-img

cv2.destroyAllWindows()

