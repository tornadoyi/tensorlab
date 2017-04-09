import sys
import select
import cv2
import time


def press_key_stop():
    while True:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            c = sys.stdin.read(1)
            break
        else:
            cv2.waitKey(10)



last_time = 0
def time_tag():
    global last_time
    subtime = time.time() - last_time
    last_time = time.time()
    return subtime


