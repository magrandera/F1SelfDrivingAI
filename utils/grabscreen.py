import time

import cv2
import mss
import numpy

with mss.mss() as sct:
    def grabscreen(region=None):
            box = {'top': 40, 'left': 0, 'width': 780, 'height': 540}
            img = numpy.array(sct.grab(box))
            img = cv2.resize(img, (250, 173))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

def testscreen():
    with mss.mss() as sct:
        while 'Screen capturing':
            box = {'top': 40, 'left': 0, 'width': 780, 'height': 540}
            t = time.time()
            img = numpy.array(sct.grab(box))
            img = cv2.resize(img, (250, 173))
            # Display the picture
            cv2.imshow('test', img)

            # Display the picture in grayscale
            # cv2.imshow('test', cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

            print('fps: {0}'.format(1 / (time.time() - t)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
  testscreen()
