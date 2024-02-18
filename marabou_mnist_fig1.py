"""
This script will try to check if a ReLU pattern is good enough to prove the outcome of an ACAS network

"""

import json
from typing import Tuple, List
import numpy as np
from maraboupy import Marabou, MarabouCore, MarabouUtils
import logging
import pandas
import matplotlib.pyplot as plt 

PATH = '/home/allen/Desktop/VerifyNNE-master/Marabou/resources/nnet/mnist/mnist4x256.nnet'
EPSILON = 1.0
# PATTERN_PATH = '/home/allen/Desktop/VerifyNNE-master/MNIST_NAP/mnist_relu_patterns_test.json'
PATTERN_PATH = '/home/allen/Desktop/VerifyNNE-master/MNIST_NAP/mnist_relu_patterns_0.0.json'
# PATTERN_PATH = '/home/allen/Desktop/VerifyNNE-master/MNIST_NAP/mnist_relu_patterns_0.01.json'
# PATTERN_PATH = '/home/allen/Desktop/VerifyNNE-master/MNIST_NAP/mnist_relu_patterns_0.5.json'


MAX_TIME = 300 #in seconds
M_OPTIONS: MarabouCore.Options = Marabou.createOptions(verbosity=0, numWorkers=10, timeoutInSeconds=MAX_TIME)


with open(PATTERN_PATH, "r") as f:
    STABLE_PATTERNS = json.load(f)


FIG1_PATH = '/home/allen/Downloads/fig1.json'
with open(FIG1_PATH, "r") as f:
    res = json.load(f)
img1 = []
img2 = []
for i in range(28):
    img1 = img1 + res["target"][i]
    img2 = img2 + res["closest_neighbor"][i]  


#print(img1)

fig = np.array(img1).reshape(28,28)
#print(fig)

#plt.imshow(fig)

diff = []
for i in range(784):
    diff.append(abs(img1[i]-img2[i]))


def init_network()->Marabou.MarabouNetworkNNet:
    network:Marabou.MarabouNetworkNNet = Marabou.read_nnet(PATH)

    # print("output nodes:", network.outputVars)
    # print("input nodes:", network.inputVars)

    logging.debug("relu list:")
    for r in network.reluList:
        logging.debug(r)  

    for i in range(784):
        network.setLowerBound(i, img2[i]-EPSILON)
        network.setUpperBound(i, img2[i]+EPSILON)
        #network.setLowerBound(i, 0)
        #network.setUpperBound(i, 1)      
    # for i in range(784):

    #     network.setLowerBound(i, img1[i]-diff[i] )
    #     network.setUpperBound(i, img1[i]+diff[i] )

    # i = 687
    # network.setLowerBound(i, img1[i]-diff[i] )
    # network.setUpperBound(i, img1[i]+diff[i] )
    return network

def add_relu_constraints(network: Marabou.MarabouNetworkNNet, 
                        relu_check_list: List[int], 
                        relu_val: List[int])->Marabou.MarabouNetworkNNet:
    """
    Add stable relus constraints to the Marabou network
    """
    for i in range(len(relu_check_list)):
        layer, idx, marabou_idx = parse_raw_idx(relu_check_list[i])  
        #if relu_val[i] < 1000:   
        if relu_val[i] == 0:
            #constraint = MarabouUtils.Equation(MarabouCore.Equation.LE)
            #constraint.addAddend(1, marabou_idx)
            #constraint.setScalar(-0.001)
            
            #output of ReLU must be 0
            
            constraint = MarabouUtils.Equation(MarabouCore.Equation.LE)
            constraint.addAddend(1, marabou_idx)
            constraint.setScalar(0)
            
            
            network.setLowerBound(marabou_idx+256, 0)
            network.setUpperBound(marabou_idx+256, 0)

            #input of ReLU must be <=0
            # network.setUpperBound(marabou_idx, 0)            

        else:
            constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
            constraint.addAddend(1, marabou_idx)
            constraint.setScalar(0.001)
        network.addEquation(constraint)

    return network


    
def parse_raw_idx(raw_idx: int) -> Tuple[int, int, int]:
    """
    only for MNIST 256xk network:
    """
    n_relus = 256
    offset = 28*28+10
    layer = raw_idx // n_relus
    idx = raw_idx % n_relus
    marabou_idx = 2*n_relus*layer + idx + offset
    return layer, idx, marabou_idx
    


def find_one_assignment(relu_check_list: List[int], relu_val: List[int])->None:
    network = init_network()
    network = add_relu_constraints(network, relu_check_list, relu_val)    
    exitCode, vals, stats = network.solve()
    assert(exitCode=="sat")    
    for idx, r in enumerate(relu_check_list):
        marabou_idx = parse_raw_idx(r)[-1]
#        print(marabou_idx, vals[marabou_idx], relu_val[idx])

def check_pattern(relu_check_list: List[int], relu_val: List[int], label: int, other_label: int)->Tuple[str, int]:
    """
    In ACAS, the prediction is the label with smallest value.
    So we check that label - other_label < 0 forall input
    by finding assignments for label - other_label >=0
    """
    print("--------CHECK PATTERN: output_{} is always less than output_{} ? --------".format(label, other_label))
    network = init_network()
#    network = add_relu_constraints(network, relu_check_list, relu_val)    
    offset = network.outputVars[0][0][0]
    #print(offset)
    
    #add output constraint
    constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
    constraint.addAddend(1, other_label+offset)
    constraint.addAddend(-1, label+offset)
    constraint.setScalar(0.0001)
    network.addEquation(constraint)

    
    exit_code: str    
    exit_code, vals, stats = network.solve(options=M_OPTIONS)
    running_time:int = stats.getTotalTimeInMicro()
    for idx, r in enumerate(relu_check_list):
        marabou_idx = parse_raw_idx(r)[-1]
        # print(marabou_idx, vals[marabou_idx], relu_val[idx])

    # print("double check output")
    # for o in network.outputVars[0][0]:
    #     print(o, vals[o])
    # print(label+offset, vals[label+offset])
    # print(other_label+offset, vals[other_label+offset])

    # print("Running time:{}".format(running_time))
    return exit_code, running_time



    # try:
    #     exit_code: str    
    #     exit_code, vals, stats = network.solve(options=M_OPTIONS)
    #     running_time:int = stats.getTotalTimeInMicro()
    #     for idx, r in enumerate(relu_check_list):
    #         marabou_idx = parse_raw_idx(r)[-1]
    #         print(marabou_idx, vals[marabou_idx], relu_val[idx])

    #     print("double check output")
    #     for o in network.outputVars[0][0]:
    #         print(o, vals[o])
    #     print(label+offset, vals[label+offset])
    #     print(other_label+offset, vals[other_label+offset])

    #     print("Running time:{}".format(running_time))
    #     return exit_code, running_time
    # except Exception:
    #     if exit_code not in ["sat", "unsat"]:
    #         print("THE QUERY CANNOT BE SOLVED")
    #     return exit_code, -1
        
def main():
    res = []
    #res = [[-1.]*10 for i in range(10)]
    # print(res)
    print("For label 1, check if its stable RELU pattern guarantees the output")
    for other_label in range(10):
        if other_label == 1:
            res.append(-1)
            continue
        relu_check_list = STABLE_PATTERNS["1"]["stable_idx"]
        relu_val = STABLE_PATTERNS["1"]["val"] 
        exit_code, running_time = check_pattern(relu_check_list, relu_val, label=1, other_label = other_label)
        if exit_code=="sat":
            res.append("SAT:{}".format(running_time/10**6))
        elif exit_code=="unsat":
            res.append("UNS:{}".format(running_time/10**6))
        else:
            res.append(exit_code)


    res = pandas.DataFrame(res)
    print(res)
    
main()
