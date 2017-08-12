import time

import cv2
import mss
import numpy


def grabscreen(region=None):
    with mss.mss() as sct:
        box = {'top': 40, 'left': 0, 'width': 950, 'height': 700}
        t = time.time()
        img = numpy.array(sct.grab(box))
        # Display the picture
        # cv2.imshow('test', img)
        img = cv2.resize(img, (180, 133))
        # Display the picture in grayscale
        #cv2.imshow('test', cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

def testscreen():
    with mss.mss() as sct:
        while 'Screen capturing':
            box = {'top': 40, 'left': 0, 'width': 950, 'height': 700}
            t = time.time()
            img = numpy.array(sct.grab(box))
            img = cv2.resize(img, (180, 133))
            # Display the picture
            cv2.imshow('test', img)

            # Display the picture in grayscale
            # cv2.imshow('test', cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

            print('fps: {0}'.format(1 / (time.time() - t)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break