import numpy as np
import time


class Timer:
    """
    Timer class to measure elapsed time
    """
    def __init__(self):
        self.times = np.array([])
        self.runtimes = np.array([])


    def reset(self):
        self.times = np.array([])


    def mark(self):
        self.times = np.append(self.times, time.perf_counter())


    def update(self):
        self.runtimes = np.diff(self.times)