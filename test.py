import time

count = 0
running_time_dic = {}

def running_time(func):
    def wrapper(*args, **kwargs):
        global running_time_dic
        start = time.time()
        r = func(*args, **kwargs)
        try:
            running_time_dic[func.__name__] += time.time() - start
        except:
            running_time_dic[func.__name__] = 0
            running_time_dic[func.__name__] += time.time() - start
        return r
    return wrapper
