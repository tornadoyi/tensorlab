import time


last_time = 0
def TAG_TIME():
    global last_time
    subtime = time.time() - last_time
    last_time = time.time()
    return subtime



def PRINT_TAG_TIME(message = None, unit=None):
    message = "{0}" if message is None else message
    subtime = TAG_TIME()
    if unit is None or unit == "s":
        subtime = subtime
    elif unit == "ms":
        subtime *= 1000

    msg = message.format(subtime)
    print(msg)


def PRINT_TAG_SECOND(message = None): return PRINT_TAG_TIME(message, "s")

def PRINT_TAG_MS(message = None): return PRINT_TAG_TIME(message, "ms")