# encoding=utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import math
import pickle
import numbers
import mpl_toolkits.mplot3d.art3d as art3d
import csv
import sys

#deserialized_a = pickle.load(open("Output22b.txt", "rb"))

if len(sys.argv) < 2:
	sys.exit('input not specified')

datas = []
with open(sys.argv[1], newline='') as csvfile:
    adata = csv.reader(csvfile, delimiter=',')
    for row in adata:
        datarow = []
        datarow.append(0)
        fls=[]
        for item in list(row)[:-1]:
            a=item.strip()
            #print(f"<{a}>")
            b=float(a)
            fls.append(b)
        datarow.append(fls[0:10])
        datarow.append(fls[10:20])
        datarow.append(fls[20:30])
        datas.append(datarow)

def collapsedata(data, i, j):
    if (data[1][i] == 0) or (data[1][j]== 0):
        return data
    if data[1][i]<data[1][j]:
        i,j = j,i
    print("before collapse {0}:{1}, {2}:{3}\n".format(data[1][i],data[3][i],data[3][j],data[1][j]))
    data[3][i] = (data[3][i]*data[1][i] + data[3][j]*data[1][j])/(data[1][i]+data[1][j])
    print("after collapse {0}:{1}, {2}:{3}\n".format(data[1][i],data[3][i],data[3][j],data[1][j]))
    data[1][i] = data[1][i]+data[1][j]
    data[1][j] = 0
    return data

"""
delta = 0.15
for idx, data in enumerate(datas):
    if not isinstance(data[0], numbers.Number):
        continue
    for i in range(len(data[1])):
        for j in range(i+1,len(data[1])):
            if abs(data[3][i]-data[3][j]) <= delta:
                datas[idx] = collapsedata(data, i, j)
"""

vdiff=[]
vdynn=[]
for idx, data in enumerate(datas):
    vdif =0
    for i in range(len(data[1])):
        vdif += data[1][i]*(data[3][i]**2)
    vdiff.append(vdif)

    amid=0
    for i in range(len(data[1])):
        amid += data[1][i]*data[2][i]
    amid /= len(data[1])

    vdyn=0
    for i in range(len(data[1])):
        vdyn += data[1][i]*((data[2][i]-amid)**2)
    vdynn.append(vdyn)

x = np.arange(0, len(datas), 1)
fig, ax = plt.subplots()
ax.plot(x, vdiff)

ax2 = ax.twinx()
ax2.plot(x, vdynn)
plt.show()
