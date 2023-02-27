from read_dat import *
from time import time
import numpy as np
import multiprocessing as mp


file = read_dat('/home/mkidson/gitRepos/dDAQ_dev/STNG.dat')


def save_ev_to_array(read_dat_file, arr, i):
    ev = read_dat_file.read_event()

    arr[i] = ev[0].get_event_id()

events = []

if __name__ == "__main__":
    with mp.Manager() as manager:
        events = manager.list(range(500))
        processes = []
        for c in range(500):
            p =  mp.Process(target=save_ev_to_array, args=(file, events, c))
            p.start()
            processes.append(p)
        for pr in processes:
            pr.join()

        events = list(events)

print(events[:20])
