# -*-coding:utf-8 -*-
import math
import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    '''
        遗传算法核心
        暂时用字符串存储基因
    '''
    def __init__(self):
        self.populationSize = 1000 # 种群大小
        self.chooseElite = True # 是否进行选择精英操作
        self.chromosomeLength = 17 # 染色体长度
        self.maxNumberIteration = 1000 # 最大迭代次数
        self.crossoverProbability = 0.6 # 交叉概率
        self.mutationProbability = 0.01 # 变异概率
        self.population = [] # 前一代种群
        self.populationNew = [] # 新一代种群
        self.lowerBound = 0 # 下限
        self.upperBound = 9 # 上限
        self.fitnessValue = [] # 每个个体的适应度值
        self.fitnessSum = [] # 总适应度函数
        self.fitnessAverage = [] # 每一代的平均适应度
        self.bestFitness = 0 # 最佳适应的值
        self.bestGen = 0 # 最佳代数
        self.bestIndividual = '' # 最佳基因
    def getMaxNumberIteration(self):
        return self.maxNumberIteration
    def getBestThreeAttr(self):
        return self.bestFitness, self.bestGen, self.bestIndividual
    def encode(self):
        pass
    def decode(self, binCode):
        '''
            f(x), x∈[lower_bound, upper_bound]
            x = lower_bound + decimal(chromosome)×(upper_bound-lower_bound)/(2^chromosome_size-1)
        '''
        decCode = int(binCode, 2)
        x = self.lowerBound + decCode * (self.upperBound - self.lowerBound) / (2**(self.chromosomeLength - 1))
        return x
    def targetFun(self, x):
        '''
            y = x+10*sin(5*x)+7*cos(4*x);
        '''
        return x+10*math.sin(5 * x) + 7 * math.cos(4 * x)
    def initPopulation(self):
        '''初始化种群 基本完成'''
        for i in range(self.populationSize):
            localStr = ''
            for j in range(self.chromosomeLength):
                localStr += str(round(random.random()))
            self.population.append(localStr)
    def fitness(self):
        '''适应度函数'''
        self.fitnessValue = []
        for i in range(self.populationSize):
            self.fitnessValue.append(0)
        # 先存储字符串转换出来的十进制的值， 然后带入目标函数中计算出真实的值
        for i in range(self.populationSize):
            for j in range(self.chromosomeLength):
                if (self.population[i][j] == "1"):
                    # print(self.population[i][j])
                    self.fitnessValue[i] = self.fitnessValue[i] + 2**(j)
            self.fitnessValue[i] = self.lowerBound + self.fitnessValue[i] * (self.upperBound - self.lowerBound) / (2**self.chromosomeLength - 1)
            # print('十进制值', self.fitnessValue[i])
            self.fitnessValue[i] = self.targetFun(self.fitnessValue[i]) # 因为这个直接调用了函数所以，求函数的最大值其实就是求适应值
            # print('函数值', self.fitnessValue[i])
    def rank(self):
        '''
        对个体的适应度大小进行排序，并保留最佳个体
        '''
        for i in range(self.populationSize):
            self.fitnessSum.append(0.)
        for i in range(self.populationSize):
            minIndex = i
            for j in range(i, self.populationSize):
                if (self.fitnessValue[j] < self.fitnessValue[minIndex]):
                    minIndex = j
            if (minIndex != i):
                tmp = self.fitnessValue[i]
                self.fitnessValue[i] = self.fitnessValue[minIndex]
                self.fitnessValue[minIndex] = tmp
            tmpChromosome = self.population[i]
            self.population[i] = self.population[minIndex]
            self.population[minIndex] = tmpChromosome
        for i in range(self.populationSize):
            if i == 0:
                self.fitnessSum[i] = self.fitnessSum[i] + self.fitnessValue[i]
            else:
                self.fitnessSum[i] = self.fitnessSum[i-1] + self.fitnessValue[i]
        self.fitnessAverage.append(self.fitnessSum[self.populationSize - 1] / self.populationSize)
        if(self.fitnessValue[self.populationSize - 1] > self.bestFitness):
            self.bestFitness = self.fitnessValue[self.populationSize - 1]
            self.bestGen = len(self.fitnessAverage)
            self.bestIndividual = self.population[self.populationSize - 1]
    def crossOver(self):
        '''
            交叉 改进 交换前一半的基因 = 交换后一半的基因
        '''
        for i in range(0, self.populationSize, 2):
            # 生成随机概率
            if (random.random() < self.crossoverProbability):
                crossPosition = round(random.random() * self.chromosomeLength)
                if (crossPosition == 0) or (crossPosition == 1):
                    continue
                # 对crossPosition及之后的二进制串进行交换
                tmp = self.population[i][crossPosition:]
                self.population[i] = self.population[i][0:crossPosition] + self.population[i+1][crossPosition:]
                self.population[i+1] = self.population[i+1][0:crossPosition] + tmp
    def mutation(self):
        '''
            单点变异操作
        '''
        for i in range(self.populationSize):
            if (random.random() < self.mutationProbability):
                mutationPosition = round(random.random() * self.chromosomeLength)
                if mutationPosition == self.chromosomeLength:
                    continue
                tmp = list(self.population[i])
                if ( tmp[mutationPosition] == '1'):
                    tmp[mutationPosition] = '0'
                else:
                    tmp[mutationPosition] = '1'
                self.population[i] = ''.join(tmp)
    def selection(self):
        '''
            轮盘赌选择操作,按照[0, 总适应度] 先求出结果这个是要改的 应为 这个放弃了 适应度为负的值
        '''
        self.populationNew = [None] * self.populationSize
        for i in range(self.populationSize):
            r = random.random() * self.fitnessSum[self.populationSize - 1]
            first = 0
            last = self.populationSize - 1
            mid = round((last + first) / 2)
            idx = -1
            # 原文是排序选择和 r 差不多的个体（排中法？）
            while(first <= last and idx==-1):
                if(r > self.fitnessSum[mid]):
                    first = mid
                elif( r < self.fitnessSum[mid]):
                    last = mid
                else:
                    idx = mid # 一般不会走到这个分支因为是浮点数
                    break
                mid = round((last + first) / 2)
                if(last - first == 1):
                    idx = last
                    break
            # 产生新一代的个体
            self.populationNew[i] = self.population[idx]
        if self.chooseElite:
            p = self.populationSize - 1
        else:
            p = self.populationSize
        for i in range(p):
            self.population[i] = self.populationNew[i]
    def debug(self, myStr):
        if (myStr == 'population'):
            print('初始化种群', self.population)
        elif (myStr == 'fitnessValue'):
            print('fitnessValue', self.fitnessValue)
        elif (myStr == 'rank'):
            print('fitnessSum', self.fitnessSum)
            print('fitnessValue', self.fitnessValue)
            print('population', self.population)
        print('--------------------------------------------------------------------')
    def plotGA(self):
        plt.figure(1)  # 创建图表1
        x = np.linspace(1, self.maxNumberIteration, self.maxNumberIteration)
        x = x.astype(int)
        # print(x)
        plt.plot(x, self.fitnessAverage)
    def testTarget(self):
        plt.figure(1)
        x = np.linspace(0, 9, 10000)
        x = x.astype('float')
        y = []
        for i in range(len(x)):
            y.append(self.targetFun(x[i]))
        print(x)
        print(y)
        plt.plot(x, y)
        plt.show()
        # plt.scatter(x, y, color='r', marker='+')
        # plt.show()





