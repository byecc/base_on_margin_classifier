import numpy as np
import math
from processdata import Encode
from numpy import *
import time

class IntervalTrain:
    def __init__(self):
        self.weight_matrix = np.array((0,0))
        self.one = np.array((0,0))
        self.cor = 0
        self.total = 0
        self.loss = 0

    def create_weight_matrix(self,depth,width):
        self.weight_matrix = np.zeros((depth,width))
        self.one = np.array([1 for i in range(width)])

    def train(self,parameter,train_encodes,dev_encodes,test_encodes):
        for i in range(parameter.ap_iter_num):
            print('第%d轮迭代：'%(i+1))
            starttime = time.time()
            self.total = self.cor = self.loss = left_bound = 0
            right_bound = parameter.interval_batch_size
            max_len = len(train_encodes)
            if parameter.interval_batch_size <= 0:
                print("batch_size must be greater than zero")
                return None
            while left_bound<max_len:    #batch
                encodes = train_encodes[left_bound:right_bound]
                outputs,gold_indexes = self.forward(encodes,parameter.class_num)
                jude_list,output_indexes = self.margin_list(outputs,parameter.lamda,gold_indexes)
                self.backward(jude_list,output_indexes,encodes,gold_indexes,parameter.lamda)
                left_bound += parameter.interval_batch_size
                right_bound += parameter.interval_batch_size
                if right_bound >= max_len:
                    right_bound = max_len - 1
            print('训练时间：',time.time()-starttime)
            print('train accuarcy:',self.cor/self.total)
            print('loss:',self.loss )
            self.eval(dev_encodes, 'dev',parameter)
            # self.eval(test_encodes, 'test')
            if self.cor/self.total == 1.0:
                break
            train_encodes = self.encode_random(train_encodes)
        print('-------------------------')

    def forward(self,encodes,class_num):
        result_labels = []
        gold_indexes = []
        for encode in encodes:
            sum = np.array([0.0 for i in range(class_num)])
            for i in encode.code_list:
                sum += self.weight_matrix[i]
            gold_indexes.append(self.get_maxIndex(encode.label))
            result_labels.append(sum)
        return result_labels,gold_indexes

    def margin_list(self,outputs,lamda,gold_indexes):
        jude_list = [True for i in range(len(outputs))]
        output_indexes = [0 for i in range(len(outputs))]
        for i,output in enumerate(outputs):
            gold = output[gold_indexes[i]]
            for j in range(len(output)):
                if j != gold_indexes[i]:
                    output[j] += lamda
                    if output[j] >= gold:
                        jude_list[i] = False
                        output_indexes[i] = j
                else:
                    continue
            if jude_list[i] is True:
                output_indexes[i] = gold
        return jude_list,output_indexes

    def backward(self,jude_list,output_indexes,encodes,gold_indexes,lamda):
        for i,jude in enumerate(jude_list):
            if jude is False:
                encode = encodes[i]
                for e in encode.code_list:
                    self.weight_matrix[e][gold_indexes[i]] += lamda
                    self.weight_matrix[e][output_indexes[i]] -= lamda
                self.loss += 1
            else:
                self.cor+=1
            self.total+=1

    def margin(self,output,lamda,gold_index):
        for i in range(len(output)):
            if i != gold_index:
                if output[i]+lamda >= output[gold_index]:
                    return False
            else:
                continue
        return True

    # def hardmax_list(self,list,max_index):
    #     hm = []
    #     for i in range(len(list)):
    #         if i != max_index:
    #             hm.append(0)
    #         else:
    #             hm.append(1)
    #     return hm
    #
    # def hardmax(self,list):
    #     max = list[0]
    #     for i in range(len(list)):
    #         if list[i] > max:
    #             max = list[i]
    #     return max

    def get_maxIndex(self,list):
        max,index = list[0],0
        for i in range(len(list)):
            if list[i] > max:
                max,index = list[i],i
        return index

    def eval(self,encodes,dataset_name,parameter):
        cor =0
        total= 0
        for encode in encodes:
            sum = np.array([0.0 for i in range(parameter.class_num)])
            for ec in encode.code_list:
                sum+=self.weight_matrix[ec]
            gold_index = self.get_maxIndex(encode.label)
            # if self.margin(sum,parameter.lamda,gold_index):
            if self.get_maxIndex(sum) == gold_index:
                cor+=1
            total+=1
        if dataset_name=='dev' and cor/total > 0.39:
            print('*******')
        print(dataset_name+' accuracy:', cor/total)
        return cor/total

    def encode_random(self,o_encodes):
        index_list = []
        for i in range(len(o_encodes)):
            index_list.append(i)
        random.seed(200)
        random.shuffle(index_list)
        n_encodes = []
        for i in index_list:
            encode = Encode()
            encode.code_list = o_encodes[i].code_list
            encode.label = o_encodes[i].label
            n_encodes.append(encode)
        return n_encodes