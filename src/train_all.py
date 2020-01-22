import sys
sys.path.append("C:\\Users\\UKGC-PC\\Documents\\Level 4 Project")
from a.main import run_a
from b.main import run_b
from up.main import run_up
from down.main import run_down
from left.main import run_left
from right.main import run_right
from run_agent import run_test
# run all train scripts then execute

run_a()
run_b()
run_down()
run_left()
run_up()
run_right()