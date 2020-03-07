import sys
import multiprocessing
from multiprocessing import Pool
sys.path.append("C:\\Users\\UKGC-PC\\Documents\\Level 4 Project")
#sys.path.append("C:\\Users\\Luke\\Documents\\metal-mario")
from a.main import a_main
from b.main import b_main
from up.main import up_main
from down.main import down_main
from left.main import left_main
from right.main import right_main
from right_a_b.main import right_a_b_main
from right_a.main import right_a_main
from right.main import run_right


from run_agent import run_test
# run all train scripts then execute
from pathlib import Path


train_functions = [right_main(),
a_main(),
left_main(),
down_main(),
up_main(),
b_main(),
right_a_main(),
right_a_b_main(),
]

## errors - doesn't work
if __name__ ==  '__main__':
    # for function in train_functions:
    #     Pool(2).starmap(function,[() for _ in range(5)])
    run_right
    a_main()
    left_main()
    down_main()
    up_main()
    b_main()
    right_a_main()
    right_a_b_main()
