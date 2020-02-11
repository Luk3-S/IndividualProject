# Timelog

* Metal Mario
* Luke Stevenson
* 2242306
* Gerardo Aragon-Camarasa, Richard McCreadie

## Guidance

* This file contains the time log for your project. It will be submitted along with your final dissertation.
* **YOU MUST KEEP THIS UP TO DATE AND UNDER VERSION CONTROL.**
* This timelog should be filled out honestly, regularly (daily) and accurately. It is for *your* benefit.
* Follow the structure provided, grouping time by weeks.  Quantise time to the half hour.

## Week 1
### 23 Sep 2019
* *0.5 hour* Meeting with supervisor to report on work done over summer
* *3 hour* reading background material
*  *2 hour* pytorch tutorials
### 24 Sep 2019
* *3 hour* gained access to gerardo's github, delved into the codebases
* *1.5 hour* read paper written by gerardo
* *3 hour* reading project guidance notes
### 25 Sep 2019
* *0.5 hour* set up github repo for project
* *3 hour* reading hall of fame dissertation projects
* *1 hour* installing tas and super mario bros rom, getting game to run on tas
* *1 hour* reading tas tutorials
### 26 Sep 2019
* *3 hour* Research into A2C vs A3C
* *1 hour* Started drafting minutes for previous meeting, and questions for next meeting
* *2 hour* Research into external technologies mentioned in papers read
### 27 Sep 2019
* *3 hour* Research into ICRA



## Week 2
### 30 Sep 2019
* *2 hour* watching reinforcement learning youtube videos
### 1 Oct 2019
* *3 hour* Reading master thesis
### 2 Oct 2019
* *2.5 hour* Looking through master code base, running the program
### 3 Oct 2019
* *0.5 hour* Meeting with supervisor - questions about designs answered - enough research completed at this point
### 4 Oct 2019
* *1 hour* Looking at structure of ltsm project (simple nn github) and replicating it for this project
### 5 Oct 2019
* *2.5 hour* Writing code for a/actor critic file



## Week 3
### 7 Oct 2019
* *1 hour* writing a/main
* *2 hour* writing a/train
### 8 Oct 2019
* *2.5 hour* attempting to debug a/train method - issues with size mismatch when instantiating actor object
### 9 Oct 2019
* *2 hour* fixed error with a/train
### 10 Oct 2019
* *0.5 hour* meeting with advisor - questions about how to structure the project's files answered; following the same structure as simple nn project.
* *2 hour* writing environment.py
### 11 Oct 2019
* *3 hour* replicating actor and main files for b/
### 12 Oct 2019
* *1 hour* refactoring a/train for b/train



## Week 4
### 17 Oct 2019
* *0.5 hour* meeting with advisor - asked about how the CAE interacts with the project as a whole
### 19 Oct 2019
* *1.5 hour* time spent researching CAE's and how to go about implementing one
### 20 Oct 2019
* *3 hour* refactoring environment.py to reflect design changes I had made regarding actions available to the agent
* *1 hour* refactoring code base where possible


## Week 5: 21 - 27
### 21 - 27 Oct
* *0 hours* Little project progress made due to extenuating circumstances.

## Week 6: 28 - 3
### 28 - 29 Oct 2019
* *3 hour* refactoring actor critic / training /main files for the remaining button processes: down,up,left & right
### 31 Oct 2019
* *3 hour* started writing the CAE file
### 2 Nov 2019
* *2 hour* further process on the CAE - began writing the train function



## Week 7: 4 - 10
### 4 Nov 2019
* *2 hour* Looking through simple nn codebase, investigating SharedAdam
* *1.5 hour* researching online about the adam optimiser and how to use it in own code
* *1 hour* I have implemented the optimiser in the main and training files - I  aim to revisit the optimiser later in the project




## Week 8: 11 - 17
### 14 Nov 2019
* *0.5 hour* advisor meeting, feedback given on the order in which to train the CAE and A3C files
### 15 Nov 2019
* *3.5 hour* rewriting CAE to fix some errors
* *3 hour* further progress on the train and test functions in CAE




## Week 9 : 18 - 24
### 19 Nov 2019
### 21 Nov 2019
* *0.5 hour* Meeting with advisor - asked questions related to how to save and load model weights between different actor critic files
## 22 Nov 2019
* *1.5 hour* Looking at PyTorch documentation regarding model weights and how to save / load
* *2 hour* ammending training files to save and load weights
* *2 hour* Fixing bugs relating to saving and loading model weights


## Week 10: 25 - 1
### 25 Nov 2019
* *3 hour* Looking through Gerardo's repo: mario-bm to understand how the mariodata file works
### 27 Nov 2019
* *2.5 hour* Attempting to implement mariodata.py for use in CAE
### 29 Nov 2019
* *1.5 hour* completed CAE train function using mario data
### 30 Nov 2019
* *2 hour* trained CAE and tested to see if working correctly

## Week 11
### 2 - 8 Dec 2019
* *0 hour* Time Spent this week working on machine learning and ai coursework & PSI exam, little to no time spent on the project



## Week 12: 9 - 15
### 9 - 11
* *0 hour* Time spent revising for and sitting PSI exam
### 13 Dec 2019
* *2 hour* planning how to carry out test experiment to record video - decided to leave implementing multithreading until semester 2, so opted for low iteration values
* *2 hour* began experimenting with a script to record the agent executing (post-training), but this quickly became too complicated, opted to use third party software to record the video.
### 15 Dec 2019
* *3 hour* general fixes solved
* *1 hour* research done on how to write an execution script for the agent, post training



## Week 13
### 16 Dec 2019
* *2 hour* writing the train all script to replace manually executing train scripts for each button.
### 17 Dec 2019
* *1 hour* began writing the executions script for the agent
* *0.5 hour* ammended all train.py files to reflect lower episode and step counts for training so that a running agent could be recorded before the 20th of Dec (training otherwise would take too long)
* *4 hour* fully training the agent for all 6 buttons. 
### 18 Dec 2019
* *2 hour* finished execution script
* *2 hour* agent was run for the first time after being trained. small experiments done to see if changing the order in which buttons were trained impacted the agents performance. video recorded for Gerado.


## Week 14 
### 13 Jan 2020
* *2 hour* researching multithreading / asynchronicity for A3C - currently training scripts are extrememly slow and multithreading would speed this up
### 17 Jan 2020
* *0.5 hour* advisor meeting - spoken about the structure of the project

## Week 14
### 22 Jan 2020
* *1 hour* started to implement multithreading training
### 23 Jan 2020
* *2.5 hour* implemented multitheading - need to test
### 24 Jan 2020
* *2hour* researching into how to implement LSTM functionality
* *0.5 hour* meeting with advisor re: lstm
### 26 Jan 2020
* *1.5 hour* Reading through papers re: lstm networks including simple-nn accompanying paper

## Week 15
### 27 Jan 2020
* *2 hour* making adjustments to a/ to see how the agent performs when training on all buttons at once
* *1.5 hour* training a/actor_critic
### 30 Jan 2020
* *3 hour* More research into LSTM and sharing weights, minor tweaks to saving / loading weights process implemented
### 31 Jan 2020
* *2 hour* started work on LSTM class
* *0.5 hour* Advisior meeting

## Week 16
### 3rd Feb 2020
* *2 hour* started to compile list of papers / sources for dissertation writing, making critical summaries of them.
* *2.5 hour* Writing the abstract and beginning of the introductory theory for the disseration
## 5 Feb 2020
* *2hour* more work on the lstm class object and methods
* *1hour* continuing with dissertation writing
## 7 Feb 2020
* *1 hour* evaluating the agent's peformance when trained on all buttons - this could possibly be used for comparisons later on.

TODO: finish 7,8,10

## Week 17
### 11 Feb 2020
* *1 hour* refactoring codebase to reflect implementation change regarding which button is being trained
