import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def dect_image(path):
    cascade_limestone = cv2.CascadeClassifier('cascade-5/cascade.xml')
    frame = cv2.imread(path)


        # get V-channel from hsv model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = gray[:,:,2]


        # filter blob using thresholding
    v1, t_f = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t_f = cv2.bitwise_and(gray, gray, mask=t_f)
    t_f = cv2.dilate(t_f, None, 15)
    # cv2.imshow('gray image', gray)
    # cv2.imshow('otsu thresholding', t_f)

    vehicle = cascade_limestone.detectMultiScale(t_f)
        #draw candidate
    sorted_vehicle = sorted(vehicle, key=lambda veh: veh[0])
    print(sorted_vehicle)
    sorted_vehicle = list(sorted_vehicle)
    i = 0
    counter = len(sorted_vehicle)
    while(i < counter - 1):
        is_shadow = False
        check_x = sorted_vehicle[i][2] * 0.3
        if (abs(sorted_vehicle[i][0] - sorted_vehicle[i+1][0]) < check_x
            and abs(sorted_vehicle[i][0] + sorted_vehicle[i][2] - sorted_vehicle[i+1][0] - sorted_vehicle[i+1][2]) < check_x):
            is_shadow = True
        if (is_shadow):
            if sorted_vehicle[i][1] > sorted_vehicle[i+1][1]:
                sorted_vehicle.pop(i)
            else:
                sorted_vehicle.pop(i+1)
                i = i + 1
        else:
            i = i + 1
        counter = len(sorted_vehicle)

    # print('after')
    # print(sorted_vehicle)
    for (x, y, w, h) in sorted_vehicle:
        idensity = np.mean(gray[y:y+h, x:x+w])
        # print((x,y,w,h))
        # print(idensity)
            #filter by idensity
        if idensity < 192 and idensity >90:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h), (255, 0, 0), 2)
                
    return frame
    # cv2.imshow('dectect video', frame)
    #
    # key = cv2.waitKey(0)
    # cv2.destroyAllWindows()

def dect_video(path):
    cap = cv2.VideoCapture(path)

    if (not cap.isOpened()):
        print('error when open video')

    cascade_limestone = cv2.CascadeClassifier('cascade/cascade.xml')
    while(cap.isOpened()):
        ret, frame = cap.read()
        startT = time.time()

        # get V-channel from hsv model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = gray[:,:,2]
        _, t_f = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        t_f = cv2.bitwise_and(gray, gray, mask=t_f)


        vehicle = cascade_limestone.detectMultiScale(t_f, scaleFactor=1.1, minNeighbors=5)


        #draw candidate
        sorted_vehicle = sorted(vehicle, key=lambda veh: veh[0])
        # print(sorted_vehicle)
        sorted_vehicle = list(sorted_vehicle)
        i = 0
        counter = len(sorted_vehicle)
        while(i < counter - 1):
            is_shadow = False
            check_x = sorted_vehicle[i][2] * 0.3
            if (abs(sorted_vehicle[i][0] - sorted_vehicle[i+1][0]) < check_x
                and abs(sorted_vehicle[i][0] + sorted_vehicle[i][2] - sorted_vehicle[i+1][0] - sorted_vehicle[i+1][2]) < check_x):
                is_shadow = True
            if (is_shadow):
                if sorted_vehicle[i][1] > sorted_vehicle[i+1][1]:
                    sorted_vehicle.pop(i)
                else:
                    sorted_vehicle.pop(i+1)
                    i = i + 1
            else:
                i = i + 1
            counter = len(sorted_vehicle)

        # print('after')
        # print(sorted_vehicle)
            

        for (x, y, w, h) in sorted_vehicle:
            idensity = np.mean(gray[y:y+h, x:x+w])
            #filter by idensity
            if idensity < 192 and idensity > 70:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h), (255, 0, 0), 2)
                
        endT = time.time()
        print('time to process', (endT-startT)*10**3)
        cv2.imshow('dectect video', frame)

        # waits 20 ms every loop to process key presses
        key = cv2.waitKey(20)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()


# dect_video('rawdata/video/37.mp4')
# dect_video('rawdata/video/Cars_driving_at_night.mp4')
# dect_image('005430.png')
# dect_image('019282.png')
# dect_image('019296.png')
# dect_image('036654.png')
# image = dect_image('road_shadow.png')



def test_pos():
    file = open('pos.txt', 'r')
    while(True):
        info = file.readline()
        if not info:
            break;
        info = info.strip().split(' ')
        for i,v in enumerate(info):
            if not i:
                continue
            info[i] = int(v)
        candinate_sample = []
        for i in range(info[1]):
            candinate_sample.append([info[4*i+2], info[4*i+3], info[4*i+4], info[4*i+5]])

        # print(info[0])
        image = dect_image(info[0])
        for [x,y,w,h] in candinate_sample:
            # print([x,y,w,h])
            image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 3)
        cv2.imshow('image', image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_neg():
    file = open('neg.txt', 'r')
    while(True):
        info = file.readline()
        if not info:
            break;
        info = info.strip().split(' ')
        image = dect_image(info[0])
        cv2.imshow('image', image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# test_neg()
# test_pos()
# image = dect_image('rawdata/suong_mu.png')
# start = time.time()
# image = dect_image('rawdata/thu_hiem_mo_hinh.png')
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# end = time.time()
# print("time: ", (end-start)*10**3,"ms")

dect_video('rawdata/video/38.mp4')

