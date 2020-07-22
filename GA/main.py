# -*-coding:utf-8 -*-
import ga
import time
import numpy as np
gaAlgorithm = ga.GeneticAlgorithm()
# 初始化种群   随机赋值 0110值
gaAlgorithm.initPopulation()
# debug
gaAlgorithm.debug('population')
maxGN = gaAlgorithm.getMaxNumberIteration()
start = time.process_time()
for i in range(maxGN):
    # 计算适应度函数
    gaAlgorithm.fitness()
    # gaAlgorithm.debug('fitnessValue')
    gaAlgorithm.rank()
    # gaAlgorithm.debug('rank')
    gaAlgorithm.selection()
    # gaAlgorithm.debug('population')
    gaAlgorithm.crossOver()
    gaAlgorithm.mutation()
    if (i % 10 == 0):
        print(i, end=' ')
elapsed = (time.process_time() - start)
print("Time used:",elapsed)
gaAlgorithm.plotGA()
bestFitness, bestGeneration, bestIndividual = gaAlgorithm.getBestThreeAttr()
# decCode = gaAlgorithm.decode(bestIndividual)
decCode = 0
for j in range(17):
    if (bestIndividual[j] == "1"):
        # print(self.population[i][j])
        decCode = decCode + 2 ** (j)
decCode = 0 + decCode * (9) / (2 ** 17 - 1)
print('最佳个体', bestIndividual)
print('最优适应度', bestFitness)
print('最优个体对应自变量值', decCode)
print('最优个体的迭代次数', bestGeneration)

# 最佳个体 10101111011111011
# 最优适应度 24.855362812666666
# 最优个体对应自变量值 12.339157104492188
# 最优个体的迭代次数 20
# print('开始校验\n-----------------\n')
# gaAlgorithm.testTarget()
# print('校验完毕')


