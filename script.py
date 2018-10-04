import cv2
import numpy as np
from matplotlib import pyplot as plt
#import pyautogui,sys
#fellow contributor: Sudutt Harne
#fellow contributor: Jai Shukla

def nothing(x):
    pass
legend = 0
write = 0
point_left = 4

cap = cv2.VideoCapture(0)
# mouse callback function

kernel = np.ones((6,6),np.float32)
kernel = kernel/kernel.sum()
print kernel
cv2.namedWindow('image')
cv2.namedWindow('frame')
cv2.createTrackbar('Hl','image',0,255,nothing)
cv2.createTrackbar('Sl','image',0,255,nothing)
cv2.createTrackbar('Vl','image',225,255,nothing)

cv2.createTrackbar('Hu','image',0,255,nothing)
cv2.createTrackbar('Su','image',0,255,nothing)
cv2.createTrackbar('Vu','image',255,255,nothing)

cv2.createTrackbar('Bl','image',0,255,nothing)
cv2.createTrackbar('Bu','image',0,255,nothing)

#constants for image detection
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
meth = methods[0]
method = eval(meth)
template = cv2.imread('point.png',0)
w, h = template.shape[::-1]

#Coordinates for processing C
coords = np.zeros((4,2),np.uint16)
m_coords_raw = np.zeros((1,2),np.uint16)

#paint make
def draw_circle(event,x,y,flags,param):
    global point_left
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(255,0,0),-1)

        if point_left > 0 :
            coords[4 - point_left] = (x,y)
            point_left = point_left - 1
        print 'DotMade',(x,y)

        if point_left == 0 :
            print coords

cv2.setMouseCallback('frame',draw_circle)

while(1):

    # Take each frame
    _, frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);

    #selecting filter
    hl = cv2.getTrackbarPos('Hl','image')
    sl = cv2.getTrackbarPos('Sl','image')
    vl = cv2.getTrackbarPos('Vl','image')

    hu = cv2.getTrackbarPos('Hu','image')
    su = cv2.getTrackbarPos('Su','image')
    vu = cv2.getTrackbarPos('Vu','image')

    bl = cv2.getTrackbarPos('Bl','image')
    bu = cv2.getTrackbarPos('Bu','image')

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    #hsv(0, 100%, 100%) hsv(10, 28%, 59%)
    lower_blue = np.array([hl,sl,vl])
    upper_blue = np.array([hu,su,vu])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #blackTrade = cv2.inRange(grey, bl, bu)

    res = cv2.bitwise_and(frame,frame, mask= mask)
    f = cv2.filter2D(res,-1,kernel)

    '''# Bitwise-AND mask and original image
    #blur = cv2.GaussianBlur(res,(5,5),0)
#    _,binary = cv2.threshold(f,bl,bu,cv2.THRESH_BINARY)
#    bblur = cv2.filter2D(binary,-1,kernel)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        write = 1
        mask2 = mask.copy()
        cv2.imshow('newMask',mask2)
        mask_diff = cv2.bitwise_xor(mask,mask2)
        cv2.imshow('XOR',mask_diff)
        print "2nd Mask ready"

    '''
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
#    cv2.imshow('greyMask',blackTrade)

    '''
    if write:
        res2 = cv2.bitwise_and(frame,frame, mask= mask_diff)
        #cv2.imshow('mask_diff',mask_diff)
        cv2.imshow('res2',res2)
    '''
    cv2.imshow('res',res)
    cv2.imshow('filter',f)

    #cv2.imshow('bblur',bblur)
    #cv2.imshow('binary',binary)

    img = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY).copy()
    img2 = img.copy()

    resl = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resl)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    m_coords_raw = ((top_left[0] + bottom_right[0])/2,(top_left[0] + bottom_right[0])/2)

    #code for handling mouse
    '''if point_left ==0:
        mnew=m_coords_raw - coords[0]
        b1=coords[3][0] - coords[2][0]
        b2=coords[1][0] - coords[0][0]
        h=(coords[3]+coords[2])/2 - (coords[1]+coords[0])/2
        H = h*b1/(b2-b1)
        sh=H/(H-h)
        final_x=mnew[0]*sh
        final_y=mnew[1]

    #pyautogui.moveTo(final_x,final_y)
    '''

    print m_coords_raw
    cv2.imshow("Processing",resl);
    cv2.imshow("detection",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('Dot.png',res)
        cv2.imwrite('filtered.png',f)
        break
    #Differential Filter

cv2.destroyAllWindows()
