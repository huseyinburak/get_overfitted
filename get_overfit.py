# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:51:33 2023

@author: hbonen
"""
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r'D:\Burak\Hacattepe_Universitesi\Doktora\Dersler\CMP712_Machine_Learning\hw1\train.csv').sort_values("x")
df.head()
# dfSorted = df.sort_values("x")
df.to_csv("trainSorted.csv")
df = pd.read_csv(r'D:\Burak\Hacattepe_Universitesi\Doktora\Dersler\CMP712_Machine_Learning\hw1\trainSorted.csv')
xS = df.sort_values("x").x
tS = df.sort_values("x").t


#after 2.775.133 iteration 2.922.799
# w1 = np.zeros(2, dtype = float)
w1 = np.array([0.98684605, -1.06968138]) # Beacuse it reached saturaion, calculation was disabled
# w2 = np.zeros(3, dtype = float)
w2 = np.array([1.4037605,  -0.9821509,  -4.86863078]) # Beacuse it reached saturaion, calculation was disabled
# w3 = np.zeros(4, dtype = float)
# w3 = np.array([1.39804336, -0.35444795, -4.8194472,  -4.14491912])#after 799.133 iteration
w3 = np.array([ 1.44870396, -0.27653776, -7.06186617, -3.71495316])
# # w4 = np.zeros(5, dtype = float)
# w4 = np.array([1.40674265, -0.38061714, -5.23216182, -3.96726908,  2.0596894])#after 644.133 iteration
w4 = np.array([ 1.44870396, -0.27653776, -7.06186617, -3.71495316, 10.29207626])
# # w5 = np.zeros(6, dtype = float)
# w5 = np.array([1.40630502, -0.38163428, -5.23142312, -3.63893618,  2.07992633, -1.83058642])#after 644.133 iteration
w5 = np.array([ 1.44870396, -0.27653776, -7.06186617, -3.71495316, 10.29207626, -4.7466357])
# # w6 = np.zeros(7, dtype = float)
# w6 = np.array([ 1.40710283, -0.38066405, -5.25715698, -3.64474875,  2.06245537, -1.83257438, 0.71657703])#after 644.133 iteration
w6 = np.array([ 1.44870396, -0.27653776, -7.06186617, -3.71495316, 10.29207626, -4.7466357, 3.10631815])
# # w7 = np.zeros(8, dtype = float)
# w7 = np.array([1.40705948, -0.38142793, -5.25697034, -3.62038398,  2.06410781, -1.82463119, 0.71713585, -0.62187817])#after 644.133 iteration
w7 = np.array([ 1.44870396, -0.27653776, -7.06186617, -3.71495316, 10.29207626, -4.7466357, 3.10631815, -1.9773935])
# # w8 = np.zeros(9, dtype = float)
# w8 = np.array([1.40710971, -0.38136353, -5.25832294, -3.62079793,  2.06291208, -1.82477275, 0.71675143, -0.62191507,  0.18985186])#after 644.133 iteration
w8 = np.array([ 1.44870396, -0.27653776, -7.06186617, -3.71495316, 10.29207626, -4.7466357, 3.10631815, -1.9773935, 0.77951057])
# w9 = np.zeros(10, dtype = float)
w9 = np.array([ 1.44870396, -0.27653776, -7.06186617, -3.71495316, 10.29207626, -4.7466357, 3.10631815, -1.9773935, 0.77951057, -0.59576565])

learnRate = 0.01
iteration = 5
n = len(tS)

cost1 = np.array([])
cost2 = np.array([])
cost3 = np.array([])
cost4 = np.array([])
cost5 = np.array([])
cost6 = np.array([])
cost7 = np.array([])
cost8 = np.array([])
cost9 = np.array([])
for i in range(iteration):
    t1Pred = np.array([])
    t2Pred = np.array([])
    t3Pred = np.array([])
    t4Pred = np.array([])
    t5Pred = np.array([])
    t6Pred = np.array([])
    t7Pred = np.array([])
    t8Pred = np.array([])
    t9Pred = np.array([])
    for j in range(n):
        t1Pred = np.append(t1Pred, w1[0] + (w1[1]*xS[j]))
        t2Pred = np.append(t2Pred, w2[0] + (w2[1]*xS[j]) + (w2[2]*(xS[j]**2)))
        t3Pred = np.append(t3Pred, w3[0] + (w3[1]*xS[j]) + (w3[2]*(xS[j]**2))+(w3[3]*(xS[j]**3)))
        t4Pred = np.append(t4Pred, w4[0] + (w4[1]*xS[j]) + (w4[2]*(xS[j]**2))+(w4[3]*(xS[j]**3))+
                                  (w4[4]*(xS[j]**4)))
        t5Pred = np.append(t5Pred, w5[0] + (w5[1]*xS[j]) + (w5[2]*(xS[j]**2))+(w5[3]*(xS[j]**3))+
                                  (w5[4]*(xS[j]**4)) + (w5[5]*(xS[j]**5)))
        t6Pred = np.append(t6Pred, w6[0] + (w6[1]*xS[j]) + (w6[2]*(xS[j]**2))+(w6[3]*(xS[j]**3))+
                                  (w6[4]*(xS[j]**4)) + (w6[5]*(xS[j]**5)) + (w6[6]*(xS[j]**6)))
        t7Pred = np.append(t7Pred, w7[0] + (w7[1]*xS[j]) + (w7[2]*(xS[j]**2))+(w7[3]*(xS[j]**3))+
                                  (w7[4]*(xS[j]**4)) + (w7[5]*(xS[j]**5)) + (w7[6]*(xS[j]**6))+
                                  (w7[7]*(xS[j]**7)))
        t8Pred = np.append(t8Pred, w8[0] + (w8[1]*xS[j]) + (w8[2]*(xS[j]**2))+(w8[3]*(xS[j]**3))+
                                  (w8[4]*(xS[j]**4)) + (w8[5]*(xS[j]**5)) + (w8[6]*(xS[j]**6))+
                                  (w8[7]*(xS[j]**7)) + (w8[8]*(xS[j]**8)))
        t9Pred = np.append(t9Pred, w9[0] + (w9[1]*xS[j]) + (w9[2]*(xS[j]**2))+(w9[3]*(xS[j]**3))+
                                  (w9[4]*(xS[j]**4)) + (w9[5]*(xS[j]**5)) + (w9[6]*(xS[j]**6))+
                                  (w9[7]*(xS[j]**7)) + (w9[8]*(xS[j]**8)) + (w9[9]*(xS[j]**9)))
    
    w1[0] = w1[0] - (learnRate * ((1/n)*np.sum(t1Pred-tS)))
    w1[1] = w1[1] - (learnRate * ((1/n)*np.sum((t1Pred - tS)*xS)))
    c1 = (1/n)*(np.sum(t1Pred -tS)**2)    
    cost1 = np.append(cost1, c1)
    
    w2[0] = w2[0] - (learnRate * ((1/n)*np.sum(t2Pred-tS)))
    w2[1] = w2[1] - (learnRate * ((1/n)*np.sum((t2Pred - tS)*xS)))
    w2[2] = w2[2] - (learnRate * ((1/n)*np.sum((t2Pred - tS)*(xS**2))))
    c2 = (1/n)*(np.sum(t2Pred -tS)**2)     
    cost2 = np.append(cost2, c2)
    
    w3[0] = w3[0] - (learnRate * ((1/n)*np.sum(t3Pred-tS)))
    w3[1] = w3[1] - (learnRate * ((1/n)*np.sum((t3Pred - tS)*xS)))
    w3[2] = w3[2] - (learnRate * ((1/n)*np.sum((t3Pred - tS)*(xS**2))))
    w3[3] = w3[3] - (learnRate * ((1/n)*np.sum((t3Pred - tS)*(xS**3))))
    c3 = (1/n)*(np.sum(t3Pred - tS)**2)     
    cost3 = np.append(cost3, c3)
    
    w4[0] = w4[0] - (learnRate * ((1/n)*np.sum(t4Pred-tS)))
    w4[1] = w4[1] - (learnRate * ((1/n)*np.sum((t4Pred - tS)*xS)))
    w4[2] = w4[2] - (learnRate * ((1/n)*np.sum((t4Pred - tS)*(xS**2))))
    w4[3] = w4[3] - (learnRate * ((1/n)*np.sum((t4Pred - tS)*(xS**3))))
    w4[4] = w4[4] - (learnRate * ((1/n)*np.sum((t4Pred - tS)*(xS**4))))
    c4 = (1/n)*(np.sum(t4Pred -tS)**2)     
    cost4 = np.append(cost4, c4)
    
    w5[0] = w5[0] - (learnRate * ((1/n)*np.sum(t5Pred-tS)))
    w5[1] = w5[1] - (learnRate * ((1/n)*np.sum((t5Pred - tS)*xS)))
    w5[2] = w5[2] - (learnRate * ((1/n)*np.sum((t5Pred - tS)*(xS**2))))
    w5[3] = w5[3] - (learnRate * ((1/n)*np.sum((t5Pred - tS)*(xS**3))))
    w5[4] = w5[4] - (learnRate * ((1/n)*np.sum((t5Pred - tS)*(xS**4))))
    w5[5] = w5[5] - (learnRate * ((1/n)*np.sum((t5Pred - tS)*(xS**5))))
    c5 = (1/n)*(np.sum(t5Pred -tS)**2)     
    cost5 = np.append(cost5, c5)
    
    w6[0] = w6[0] - (learnRate * ((1/n)*np.sum(t6Pred-tS)))
    w6[1] = w6[1] - (learnRate * ((1/n)*np.sum((t6Pred - tS)*xS)))
    w6[2] = w6[2] - (learnRate * ((1/n)*np.sum((t6Pred - tS)*(xS**2))))
    w6[3] = w6[3] - (learnRate * ((1/n)*np.sum((t6Pred - tS)*(xS**3))))
    w6[4] = w6[4] - (learnRate * ((1/n)*np.sum((t6Pred - tS)*(xS**4))))
    w6[5] = w6[5] - (learnRate * ((1/n)*np.sum((t6Pred - tS)*(xS**5))))
    w6[6] = w6[6] - (learnRate * ((1/n)*np.sum((t6Pred - tS)*(xS**6))))
    c6 = (1/n)*(np.sum(t6Pred -tS)**2)     
    cost6 = np.append(cost6, c6)
    
    w7[0] = w7[0] - (learnRate * ((1/n)*np.sum(t7Pred-tS)))
    w7[1] = w7[1] - (learnRate * ((1/n)*np.sum((t7Pred - tS)*xS)))
    w7[2] = w7[2] - (learnRate * ((1/n)*np.sum((t7Pred - tS)*(xS**2))))
    w7[3] = w7[3] - (learnRate * ((1/n)*np.sum((t7Pred - tS)*(xS**3))))
    w7[4] = w7[4] - (learnRate * ((1/n)*np.sum((t7Pred - tS)*(xS**4))))
    w7[5] = w7[5] - (learnRate * ((1/n)*np.sum((t7Pred - tS)*(xS**5))))
    w7[6] = w7[6] - (learnRate * ((1/n)*np.sum((t7Pred - tS)*(xS**6))))
    w7[7] = w7[7] - (learnRate * ((1/n)*np.sum((t7Pred - tS)*(xS**7))))
    c7 = (1/n)*(np.sum(t7Pred -tS)**2)     
    cost7 = np.append(cost7, c7)
    
    w8[0] = w8[0] - (learnRate * ((1/n)*np.sum(t8Pred-tS)))
    w8[1] = w8[1] - (learnRate * ((1/n)*np.sum((t8Pred - tS)*xS)))
    w8[2] = w8[2] - (learnRate * ((1/n)*np.sum((t8Pred - tS)*(xS**2))))
    w8[3] = w8[3] - (learnRate * ((1/n)*np.sum((t8Pred - tS)*(xS**3))))
    w8[4] = w8[4] - (learnRate * ((1/n)*np.sum((t8Pred - tS)*(xS**4))))
    w8[5] = w8[5] - (learnRate * ((1/n)*np.sum((t8Pred - tS)*(xS**5))))
    w8[6] = w8[6] - (learnRate * ((1/n)*np.sum((t8Pred - tS)*(xS**6))))
    w8[7] = w8[7] - (learnRate * ((1/n)*np.sum((t8Pred - tS)*(xS**7))))
    w8[8] = w8[8] - (learnRate * ((1/n)*np.sum((t8Pred - tS)*(xS**8))))
    c8 = (1/n)*(np.sum(t8Pred -tS)**2)     
    cost8 = np.append(cost8, c8)
    
    w9[0] = w9[0] - (learnRate * ((1/n)*np.sum(t9Pred-tS)))
    w9[1] = w9[1] - (learnRate * ((1/n)*np.sum((t9Pred - tS)*xS)))
    w9[2] = w9[2] - (learnRate * ((1/n)*np.sum((t9Pred - tS)*(xS**2))))
    w9[3] = w9[3] - (learnRate * ((1/n)*np.sum((t9Pred - tS)*(xS**3))))
    w9[4] = w9[4] - (learnRate * ((1/n)*np.sum((t9Pred - tS)*(xS**4))))
    w9[5] = w9[5] - (learnRate * ((1/n)*np.sum((t9Pred - tS)*(xS**5))))
    w9[6] = w9[6] - (learnRate * ((1/n)*np.sum((t9Pred - tS)*(xS**6))))
    w9[7] = w9[7] - (learnRate * ((1/n)*np.sum((t9Pred - tS)*(xS**7))))
    w9[8] = w9[8] - (learnRate * ((1/n)*np.sum((t9Pred - tS)*(xS**8))))
    w9[9] = w9[9] - (learnRate * ((1/n)*np.sum((t9Pred - tS)*(xS**9))))
    c9 = (1/n)*(np.sum(t9Pred -tS)**2)     
    cost9 = np.append(cost9, c9)
    
    iteration -= 1
    if iteration % 1000 ==0 : print(iteration)

f1 = plt.figure('Linear fit to train data')
plt.scatter(xS,tS, color ='b')
plt.plot(xS, t1Pred, color = 'r')
f1.show()

f2 = plt.figure('2nd degree polynominal fit to train data')
plt.scatter(xS,tS, color ='b')
plt.plot(xS, t2Pred, color = 'r')
f2.show()

f3 = plt.figure('3rd degree polynominal fit to train data')
plt.scatter(xS,tS, color ='b')
plt.plot(xS, t3Pred, color = 'r')
f3.show()

f4 = plt.figure('4th degree polynominal fit to train data')
plt.scatter(xS,tS, color ='b')
plt.plot(xS, t4Pred, color = 'r')
f4.show()

f5 = plt.figure('5th degree polynominal fit to train data')
plt.scatter(xS,tS, color ='b')
plt.plot(xS, t5Pred, color = 'r')
f5.show()

f6 = plt.figure('6th degree polynominal fit to train data')
plt.scatter(xS,tS, color ='b')
plt.plot(xS, t6Pred, color = 'r')
f6.show()

f7 = plt.figure('7th degree polynominal fit to train data')
plt.scatter(xS,tS, color ='b')
plt.plot(xS, t7Pred, color = 'r')
f7.show()

f8 = plt.figure('8th degree polynominal fit to train data')
plt.scatter(xS,tS, color ='b')
plt.plot(xS, t8Pred, color = 'r')
f8.show()

f9 = plt.figure('9th degree polynominal fit to train data')
plt.scatter(xS,tS, color ='b')
plt.plot(xS, t9Pred, color = 'r')
f9.show()

# h1 = plt.figure('cost1')
# plt.plot(cost1)
# h1.show()

# h2 = plt.figure('cost2')
# plt.plot(cost2)
# h2.show()

# h3 = plt.figure('cost3')
# plt.plot(cost3)
# h3.show()

# h9 = plt.figure('cost9')
# plt.plot(cost9)
# h9.show()

dfT = pd.read_csv(r'D:\Burak\Hacattepe_Universitesi\Doktora\Dersler\CMP712_Machine_Learning\hw1\test.csv').sort_values("x")
dfT.head()
# dfSorted = df.sort_values("x")
dfT.to_csv("testSorted.csv")
dfT = pd.read_csv(r'D:\Burak\Hacattepe_Universitesi\Doktora\Dersler\CMP712_Machine_Learning\hw1\testSorted.csv')
xT = dfT.sort_values("x").x
tT = dfT.sort_values("x").t

prediction1 = np.array([])
prediction2 = np.array([])
prediction3 = np.array([])
prediction4 = np.array([])
prediction5 = np.array([])
prediction6 = np.array([])
prediction7 = np.array([])
prediction8 = np.array([])
prediction9 = np.array([])
for j in range(len(xT)):
    prediction1 = np.append(prediction1, w1[0] + (w1[1]*xT[j]))
    prediction2 = np.append(prediction2, w2[0] + (w2[1]*xT[j]) + (w2[2]*(xT[j]**2)))
    prediction3 = np.append(prediction3, w3[0] + (w3[1]*xT[j]) + (w3[2]*(xT[j]**2))+(w3[3]*(xT[j]**3)))
    prediction4 = np.append(prediction4, w4[0] + (w4[1]*xT[j]) + (w4[2]*(xT[j]**2))+(w4[3]*(xT[j]**3))+
                               (w4[4]*(xT[j]**4)))
    prediction5 = np.append(prediction5, w5[0] + (w5[1]*xT[j]) + (w5[2]*(xT[j]**2))+(w5[3]*(xT[j]**3))+
                               (w5[4]*(xT[j]**4)) + (w5[5]*(xT[j]**5)))
    prediction6 = np.append(prediction6, w6[0] + (w6[1]*xT[j]) + (w6[2]*(xT[j]**2))+(w6[3]*(xT[j]**3))+
                               (w6[4]*(xT[j]**4)) + (w6[5]*(xT[j]**5)) + (w6[6]*(xT[j]**6)))
    prediction7 = np.append(prediction7, w7[0] + (w7[1]*xT[j]) + (w7[2]*(xT[j]**2))+(w7[3]*(xT[j]**3))+
                               (w7[4]*(xT[j]**4)) + (w7[5]*(xT[j]**5)) + (w7[6]*(xT[j]**6))+
                               (w7[7]*(xT[j]**7)))
    prediction8 = np.append(prediction8, w8[0] + (w8[1]*xT[j]) + (w8[2]*(xT[j]**2))+(w8[3]*(xT[j]**3))+
                               (w8[4]*(xT[j]**4)) + (w8[5]*(xT[j]**5)) + (w8[6]*(xT[j]**6))+
                               (w8[7]*(xT[j]**7)) + (w8[8]*(xT[j]**8)))
    prediction9 = np.append(prediction9, w9[0] + (w9[1]*xT[j]) + (w9[2]*(xT[j]**2))+(w9[3]*(xT[j]**3))+
                               (w9[4]*(xT[j]**4)) + (w9[5]*(xT[j]**5)) + (w9[6]*(xT[j]**6))+
                               (w9[7]*(xT[j]**7)) + (w9[8]*(xT[j]**8)) + (w9[9]*(xT[j]**9)))

MSE1 = (1/n)*(np.sum(prediction1 - tT)**2)
print("1st order regression MSE : " + str(MSE1))
RMSE1 = math.sqrt(MSE1)
print("1st order regression RMSE : " + str(RMSE1))
testPlot1 = plt.figure('1st order fit to test data')
plt.scatter(xT, tT, color ='b')
plt.plot(xT, prediction1, color = 'r')
testPlot1.show()

MSE2 = (1/n)*(np.sum(prediction2 - tT)**2)
print("2st order regression MSE : " + str(MSE2))
RMSE2 = math.sqrt(MSE2)
print("2st order regression RMSE : " + str(RMSE2))
testPlot2 = plt.figure('2st order fit to test data')
plt.scatter(xT, tT, color ='b')
plt.plot(xT, prediction2, color = 'r')
testPlot2.show()

MSE3 = (1/n)*(np.sum(prediction3 - tT)**2)
print("3st order regression MSE : " + str(MSE3))
RMSE3 = math.sqrt(MSE3)
print("3st order regression RMSE : " + str(RMSE3))
testPlot3 = plt.figure('3st order fit to test data')
plt.scatter(xT, tT, color ='b')
plt.plot(xT, prediction3, color = 'r')
testPlot3.show()

MSE4 = (1/n)*(np.sum(prediction4 - tT)**2)
print("4st order regression MSE : " + str(MSE4))
RMSE4 = math.sqrt(MSE4)
print("4st order regression RMSE : " + str(RMSE4))
testPlot4 = plt.figure('4st order fit to test data')
plt.scatter(xT, tT, color ='b')
plt.plot(xT, prediction4, color = 'r')
testPlot4.show()

MSE5 = (1/n)*(np.sum(prediction5 - tT)**2)
print("5st order regression MSE : " + str(MSE5))
RMSE5 = math.sqrt(MSE5)
print("5st order regression RMSE : " + str(RMSE5))
testPlot5 = plt.figure('5st order fit to test data')
plt.scatter(xT, tT, color ='b')
plt.plot(xT, prediction5, color = 'r')
testPlot5.show()

MSE6 = (1/n)*(np.sum(prediction6 - tT)**2)
print("6st order regression MSE : " + str(MSE6))
RMSE6 = math.sqrt(MSE6)
print("6st order regression RMSE : " + str(RMSE6))
testPlot6 = plt.figure('6st order fit to test data')
plt.scatter(xT, tT, color ='b')
plt.plot(xT, prediction6, color = 'r')
testPlot6.show()

MSE7 = (1/n)*(np.sum(prediction7 - tT)**2)
print("7st order regression MSE : " + str(MSE7))
RMSE7 = math.sqrt(MSE7)
print("7st order regression RMSE : " + str(RMSE7))
testPlot7 = plt.figure('7st order fit to test data')
plt.scatter(xT, tT, color ='b')
plt.plot(xT, prediction7, color = 'r')
testPlot7.show()

MSE8 = (1/n)*(np.sum(prediction8 - tT)**2)
print("8st order regression MSE : " + str(MSE8))
RMSE8 = math.sqrt(MSE8)
print("8st order regression RMSE : " + str(RMSE8))
testPlot8 = plt.figure('8st order fit to test data')
plt.scatter(xT, tT, color ='b')
plt.plot(xT, prediction8, color = 'r')
testPlot8.show()

MSE9 = (1/n)*(np.sum(prediction9 - tT)**2)
print("9st order regression MSE : " + str(MSE9))
RMSE9 = math.sqrt(MSE9)
print("9st order regression RMSE : " + str(RMSE9))
testPlot9 = plt.figure('9st order fit to test data')
plt.scatter(xT, tT, color ='b')
plt.plot(xT, prediction9, color = 'r')
testPlot9.show()
