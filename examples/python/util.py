import sys
import select
import cv2


def press_key_stop():
    while True:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            c = sys.stdin.read(1)
            break
        else:
            cv2.waitKey(10)