"""
Created on 2016-12-11

class: RL4SRD

@author: Long Xia
@contact: xl.1988.life@gmail.com
"""
# !/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import json
import yaml
import copy
import math
import random
import numpy as np


class RL4SRD(object):
    """docstring for RL4SRD"""
    def __init__(self, fileQueryPermutaion, fileQueryRepresentation, fileDocumentRepresentation, fileQueryDocumentSubtopics, folder):
        super(RL4SRD, self).__init__()

        with open(fileQueryPermutaion) as self.fileQueryPermutaion:
            self.dictQueryPermutaion = json.load(self.fileQueryPermutaion)

        with open(fileQueryRepresentation) as self.fileQueryRepresentation:
            self.dictQueryRepresentation = json.load(self.fileQueryRepresentation)
        for query in self.dictQueryRepresentation:
            self.dictQueryRepresentation[query] = np.matrix([self.dictQueryRepresentation[query]], dtype=np.float)
            self.dictQueryRepresentation[query] = np.transpose(self.dictQueryRepresentation[query])

        with open(fileDocumentRepresentation) as self.fileDocumentRepresentation:
            self.dictDocumentRepresentation = json.load(self.fileDocumentRepresentation)
        for doc in self.dictDocumentRepresentation:
            self.dictDocumentRepresentation[doc] = np.matrix([self.dictDocumentRepresentation[doc]], dtype=np.float)
            self.dictDocumentRepresentation[doc] = np.transpose(self.dictDocumentRepresentation[doc])

        with open(fileQueryDocumentSubtopics) as self.fileQueryDocumentSubtopics:
            self.dictQueryDocumentSubtopics = json.load(self.fileQueryDocumentSubtopics)

        self.folder = folder
        with open(self.folder + '/config.yml') as self.confFile:
            self.dictConf = yaml.load(self.confFile)
        self.floatLearningRate = self.dictConf['learning_rate']
        self.listTestSet = self.dictConf['test_set']
        self.listValidationSet = self.dictConf['validation_set']
        self.lenTrainPermutation = self.dictConf['length_train_permutation']
        self.K = self.dictConf['K']
        self.gamma = self.dictConf['gamma']
        self.hidden_dim = self.dictConf['hidden_dim']
        
        self.fileResult = open(self.folder + '/RL_result.dat', 'w')
        self.fileValidation = open(self.folder + '/RL_validation.dat', 'w')

        self.floatTestTime = 0.0

        self.__RL_build__()

    def __del__(self):
        self.fileResult.close()
        self.fileValidation.close()

    def __RL_build__(self):
        #self.U = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (100, self.hidden_dim))
        #self.V_q = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, 100))
        #self.V = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, 100))
        #self.W = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))

        self.U = np.random.uniform(-1./self.hidden_dim, 1./self.hidden_dim, (100, self.hidden_dim))
        self.V_q = np.random.uniform(-1./self.hidden_dim, 1./self.hidden_dim, (self.hidden_dim, 100))
        self.V = np.random.uniform(-1./self.hidden_dim, 1./self.hidden_dim, (self.hidden_dim, 100))
        self.W = np.random.uniform(-1./self.hidden_dim, 1./self.hidden_dim, (self.hidden_dim, self.hidden_dim))

    def __RL_derive_assign(self):
        self.U_derive = np.zeros((100, self.hidden_dim))
        self.V_q_derive = np.zeros((self.hidden_dim, 100))
        self.V_derive = np.zeros((self.hidden_dim, 100))
        self.W_derive = np.zeros((self.hidden_dim, self.hidden_dim))

    def __RL_derive_list(self):
        self.list_U_derive = []
        self.list_V_derive = []
        self.list_V_q_derive = []
        self.list_W_derive = []
        self.list_h = []
        self.list_diag_h = []
        self.list_G_t = []

    def __sigmoid(self, arrayX):
        return 1 / (1+np.exp(-arrayX))

    def __sigmoid_derive(self, arrayX):
        return    arrayX * (1-arrayX)

    def __sigmoid_diag(self, arrayX):
        Ashape = arrayX.shape
        tmp = np.zeros(Ashape[0])
        for i in xrange(Ashape[0]):
            tmp[i] = arrayX[i][0] * ( 1 - arrayX[i][0] )
        return np.diag(tmp)

    def preference(self, o_list):
        tmp = random.random()
        for i in xrange(len(o_list)):
            if tmp < o_list[i]:
                return i

    def alphaDCG(self, alpha, query, docList, k):
        DCG = 0.0
        subtopics = []
        for i in xrange(20):
            subtopics.append(0)
        for i in xrange(k):
            G = 0.0
            if docList[i] not in self.dictQueryDocumentSubtopics[query]:
                continue
            listDocSubtopics = self.dictQueryDocumentSubtopics[query][docList[i]]
            if len(listDocSubtopics) == 0:
                    G = 0.0
            else:
                for subtopic in listDocSubtopics:
                    G += (1-alpha) ** subtopics[int(subtopic)-1]
                    subtopics[int(subtopic)-1] += 1
            DCG += G/math.log(i+2, 2)
        return DCG

    def subtopic_recall(self):
        pass

    def Train(self):
        listKeys = self.dictQueryPermutaion.keys()
        random.shuffle(listKeys)

        for query in listKeys:
            #if (int(query) in self.listTestSet) or (int(query) in self.listValidationSet):
            if int(query) in self.listTestSet:
                continue

            self.__RL_derive_assign()
            self.__RL_derive_list()

            q = self.dictQueryRepresentation[query]
            self.h_t = self.__sigmoid(np.dot(self.V_q, q))

            listPermutation = copy.deepcopy(self.dictQueryPermutaion[query]['permutation'])
            listSelectedSet = []

            for t in xrange(self.lenTrainPermutation):
                #store h_t and diag(h_t * (1-h_t))
                h_t_tmp = copy.deepcopy(self.h_t)
                self.list_h.append(h_t_tmp)
                self.list_diag_h.append(self.__sigmoid_diag(h_t_tmp))

                Z_sum = 0.0

                derive_U_sum = np.zeros((100, self.hidden_dim))
                derive_V_q_sum = np.zeros((self.hidden_dim, 100))
                derive_V_sum = np.zeros((self.hidden_dim, 100))
                derive_W_sum = np.zeros((self.hidden_dim, self.hidden_dim))

                s_t = np.dot(self.U, self.h_t)

                x_list = []
                x_prob = []
                x_prob_sum = 0.0
                x_prob_list = []

                list_derive_f_U = []
                list_derive_f_V_q = []
                list_derive_f_V = []
                list_derive_f_W = []
                for j in xrange(len(listPermutation)):
                    x_score = np.exp(np.dot(np.transpose(self.dictDocumentRepresentation[listPermutation[j]]), s_t))
                    x_score = np.array(x_score)
                    x_score = x_score[0][0]
                    Z_sum += x_score
                    x_list.append(x_score)

                    derive_f_U_tmp = np.dot(self.dictDocumentRepresentation[listPermutation[j]], np.transpose(self.h_t))
                    list_derive_f_U.append(derive_f_U_tmp)
                    derive_U_sum += derive_f_U_tmp * x_score

                    derive_f_h_tmp = np.dot(np.transpose(self.U), self.dictDocumentRepresentation[listPermutation[j]])

                    if t == 0:
                        derive_f_V_q_tmp = np.dot(self.list_diag_h[t], np.dot(derive_f_h_tmp, np.transpose(q)))
                        list_derive_f_V_q.append(derive_f_V_q_tmp)
                        derive_V_q_sum += derive_f_V_q_tmp * x_score
                    else:
                        derive_f_V_tmp = np.dot(self.list_diag_h[t], np.dot(derive_f_h_tmp, np.transpose(self.dictDocumentRepresentation[listSelectedSet[t-1]])))
                        derive_f_W_tmp = np.dot(self.list_diag_h[t], np.dot(derive_f_h_tmp, np.transpose(self.list_h[t-1])))
                        for i in xrange(t-1, 1, -1):
                            derive_f_h_tmp = np.dot(np.dot(np.transpose(self.W),self.list_diag_h[i]), derive_f_h_tmp)
                            derive_f_V_tmp += np.dot(self.list_diag_h[i-1], np.dot(derive_f_h_tmp, np.transpose(self.dictDocumentRepresentation[listSelectedSet[i-2]])))
                            derive_f_W_tmp += np.dot(self.list_diag_h[i-1], np.dot(derive_f_h_tmp, np.transpose(self.list_h[i-1])))

                        list_derive_f_V.append(derive_f_V_tmp)
                        list_derive_f_W.append(derive_f_W_tmp)
                        derive_V_sum += derive_f_V_tmp * x_score
                        derive_W_sum += derive_f_W_tmp * x_score

                #generage policy pi
                for item in x_list:
                    x_one = item/Z_sum
                    x_prob.append(x_one)
                    x_prob_sum += x_one
                    x_prob_list.append(x_prob_sum)

                #sample action
                preference_j = self.preference(x_prob_list)
                listSelectedSet.append(listPermutation[preference_j])

                R = self.alphaDCG(0.5, query, listSelectedSet, t+1) - self.alphaDCG(0.5, query, listSelectedSet, t)
                self.list_G_t.append(R)

                x_t = self.dictDocumentRepresentation[listPermutation[preference_j]]
                self.h_t = self.__sigmoid(np.dot(self.V, x_t) + np.dot(self.W, self.h_t))

                self.list_U_derive.append(list_derive_f_U[preference_j] - derive_U_sum/Z_sum)
                if t == 0:
                    self.list_V_q_derive.append(list_derive_f_V_q[preference_j] - derive_V_q_sum/Z_sum)
                else:
                    self.list_V_derive.append(list_derive_f_V[preference_j] - derive_V_sum/Z_sum)
                    self.list_W_derive.append(list_derive_f_W[preference_j] - derive_W_sum/Z_sum)

                #X = X/x_t
                del listPermutation[preference_j]

            for t in xrange(len(self.list_G_t)):
                G = 0.0
                for j in xrange(t, len(self.list_G_t)):
                    G += self.list_G_t[j] * (self.gamma ** j)
                self.V_q_derive += (self.gamma ** t) * G * self.list_V_q_derive[0]
                self.U_derive += (self.gamma ** t) * G * self.list_U_derive[t]

                if t > 0:
                    self.V_derive += (self.gamma ** t) * G * self.list_V_derive[t-1]
                    self.W_derive += (self.gamma ** t) * G * self.list_W_derive[t-1]

            self.U += self.floatLearningRate * self.U_derive
            self.V += self.floatLearningRate * self.V_derive
            self.V_q += self.floatLearningRate * self.V_q_derive
            self.W += self.floatLearningRate * self.W_derive

    def Prediction(self, listInput, boolTest):
        floatSumResultScore = 0.0
        dictResult = {}
        for query in listInput:
            if boolTest:
                fileRankingResult = open(self.folder + '/ranking/' + 'test' + str(query) + '.ranking', 'w')
            else:
                fileRankingResult = open(self.folder + '/ranking/' + 'val' + str(query) + '.ranking', 'w')
            listSelectedSet = []
            listTest = copy.deepcopy(self.dictQueryPermutaion[str(query)]['permutation'])
            idealScore = self.alphaDCG(0.5, str(query), listTest, self.K)
            if idealScore == 0:
                continue
            random.shuffle(listTest)

            self.s_t = self.__sigmoid(np.dot(self.V_q, self.dictQueryRepresentation[str(query)]) + np.dot(self.W, np.zeros((self.hidden_dim,1))))

            while len(listSelectedSet) < self.K:
                bestScore = -10000000.0
                bestDoc = ''
                s_tmp = self.s_t
                for doc in listTest:
                    o_tmp = np.dot(self.U, s_tmp)
                    doc_score = np.dot(np.transpose(o_tmp), self.dictDocumentRepresentation[doc])
                    if doc_score > bestScore and doc not in listSelectedSet:
                        bestScore = doc_score
                        bestDoc = doc
                        self.s_t = self.__sigmoid(np.dot(self.V, self.dictDocumentRepresentation[bestDoc]) + np.dot(self.W, s_tmp))
                listSelectedSet.append(bestDoc)
                fileRankingResult.write(bestDoc + '\n')

            resultScore = self.alphaDCG(0.5, str(query), listSelectedSet, self.K)
            dictResult[query] = resultScore/idealScore
            floatSumResultScore += resultScore/idealScore

            fileRankingResult.close()

        print dictResult
        return floatSumResultScore/len(dictResult.keys())

    def main(self):
        iteration = 1
        #while (iteration < 151):
        while True:
            print 'iteration:' + str(iteration)

            self.Train()

            #validation = self.Prediction(self.listValidationSet, False)
            #print round(validation, 4)
            #print '\n'
            #self.fileValidation.write(str(iteration) + ' ' + str(validation) + '\n')
            #self.fileValidation.flush()

            result = self.Prediction(self.listTestSet, True)
            print 'ndcg:' + str(round(result, 4))
            self.fileResult.write(str(iteration) + ' ' + str(result) + '\n')
            self.fileResult.flush()

            print '\n'

            iteration += 1


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print 'Error: params number is 5!'
        print 'Need: query permutation file, query representation file, document representation file, query document subtopics file, and folder!'
        sys.exit(-1)

    carpe_diem = RL4SRD(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    carpe_diem.main()
    del carpe_diem
    print 'Game over!'