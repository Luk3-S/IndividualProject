import sys
import multiprocessing
from multiprocessing import Pool
#sys.path.append("C:\\Users\\UKGC-PC\\Documents\\Level 4 Project")
sys.path.append("C:\\Users\\Luke\\Documents\\metal-mario")
from a.main import run_a
from b.main import run_b
from up.main import run_up
from down.main import run_down
from left.main import run_left
from right.main import run_right
from run_agent import run_test
# run all train scripts then execute
from pathlib import Path


buttons_to_train = [run_a(),
run_b(),
run_down(),
run_left(),
run_up(),
run_right(),
]

if __name__ ==  '__main__':
    for function in buttons_to_train:
        Pool(2).starmap(function,[() for _ in range(5)])
