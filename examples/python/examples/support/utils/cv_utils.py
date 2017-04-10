import sys
import select
import cv2
import time


def press_key_stop(message = None):
    if message is not None: print(message)
    while True:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            c = sys.stdin.read(1)
            break
        else:
            cv2.waitKey(10)






