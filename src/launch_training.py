import sys
sys.path.append("C:\\Users\\UKGC-PC\\Documents\\metal-mario-master\\IndividualProject")
from controller.main import run_controller_train
from src.all_main import setup_training
from src.all_train import train



if __name__ ==  '__main__':
    
    
    test0= "exp0"
    experiment_num = 0
    setup_training(['right'],-1,500,test0,experiment_num)
    setup_training(['right','A'],0,500,test0,experiment_num)
    setup_training(['right','A','B'],1,500,test0,experiment_num)
    run_controller_train(2,test0,experiment_num)

    test1= "exp1"
    experiment_num = 1
    setup_training(['right'],-1,500,test1,experiment_num)
    setup_training(['right','A'],0,500,test1,experiment_num)
    setup_training(['right','A','B'],1,500,test1,experiment_num)
    run_controller_train(2,test1,experiment_num)

    test2= "exp2"
    experiment_num = 2
    setup_training(['right'],-1,500,test2,experiment_num)
    setup_training(['right','A'],0,500,test2,experiment_num)
    setup_training(['right','A','B'],1,500,test2,experiment_num)
    run_controller_train(2,test2,experiment_num)

    test3= "exp3"
    experiment_num = 3
    setup_training(['right'],-1,500,test3,experiment_num)
    setup_training(['right','A'],0,500,test3,experiment_num)
    setup_training(['right','A','B'],1,500,test3,experiment_num)
    run_controller_train(2,test3,experiment_num)

    test4= "exp4"
    experiment_num = 4
    setup_training(['right'],-1,500,test4,experiment_num)
    setup_training(['right','A'],0,500,test4,experiment_num)
    setup_training(['right','A','B'],1,500,test4,experiment_num)
    run_controller_train(2,test4,experiment_num)

    test5= "exp5"
    experiment_num = 5
    setup_training(['right'],-1,500,test5,experiment_num)
    setup_training(['right','A'],0,500,test5,experiment_num)
    setup_training(['right','A','B'],1,500,test5,experiment_num)
    run_controller_train(2,test5,experiment_num)

    test6= "exp6"
    experiment_num = 6
    setup_training(['right'],-1,500,test6,experiment_num)
    setup_training(['right','A'],0,500,test6,experiment_num)
    setup_training(['right','A','B'],1,500,test6,experiment_num)
    run_controller_train(2,test6,experiment_num)

    test7= "exp7"
    experiment_num = 7
    setup_training(['right'],-1,500,test7,experiment_num)
    setup_training(['right','A'],0,500,test7,experiment_num)
    setup_training(['right','A','B'],1,500,test7,experiment_num)
    run_controller_train(2,test7,experiment_num)