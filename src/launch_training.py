import sys
import multiprocessing
from multiprocessing import Pool
sys.path.append("C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject")
#sys.path.append("C:\\Users\\Luke\\Documents\\metal-mario")
# from right_a_b.main import run_right_a_b
# from right_a.main import run_right_a
# from right.main import run_right
from controller.main import run_lstm
from src.all_main import setup_training

from src.all_train import train
# run all train scripts then execute
from pathlib import Path

right_only = [['right'],['A'],['right','A'],['right','A','B']]


if __name__ ==  '__main__':
    #setup_training(['right'],-1,1200)
    #setup_training(['A'],0,500)
    #setup_training(['right','A'],0,1000)
    #setup_training(['right','A','B'],1,1000)
    run_lstm(2)