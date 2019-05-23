# encoding=utf8
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import math
import pickle
import numbers
import mpl_toolkits.mplot3d.art3d as art3d
import csv

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


delta = 0.15

for idx, data in enumerate(datas):
    if not isinstance(data[0], numbers.Number):
        continue
    for i in range(len(data[1])):
        for j in range(i+1,len(data[1])):
            if abs(data[3][i]-data[3][j]) <= delta:
                datas[idx] = collapsedata(data, i, j)


x=[]
y=[]
z=[]
mu=[]
def addpoint(data,i,n):
    x.append(n)
    y.append(data[3][i])
    z.append(data[1][i])
    mu.append(data[2][i])
    return

treshold = 0.1
for idx,data in enumerate(datas):
    for i in range(len(data[1])):
        if data[1][i]>treshold:
            addpoint(data,i, idx)

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

import matplotlib.cm as cm
colors = cm.jet(np.linspace(0,1,101))
maxz=max(z)

for xi, yi, zi in zip(x, y, z):
    line=art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), color=colors[int(zi/maxz*100)], linewidth=0.1, alpha=0.5, marker='o', markevery=(1, 1))
    ax.add_line(line)
ax.set_xlim3d(0, max(x))
ax.set_ylim3d(0, max(y))
ax.set_zlim3d(0, 2)
ax.view_init(50)
plt.draw()
plt.show()
"""

"""
#df = pd.read_csv('RI.IMOEX_180401_190401_daily.csv', sep=';')
#vol = df['<VOL>'].to_numpy()
#close = df['<CLOSE>'].to_numpy()

plt.close('all')
#plt.scatter(x, y, s=1)
plt.scatter(x, mu, s=1)
plt.show()
"""

f, axarr = plt.subplots()
axarr.scatter(x, y, 1, alpha=0.5, c=z, cmap="jet")
#axarr[1].scatter(x, mu, 1, alpha=0.5)
axarr.set(xlabel='Номер окна (n)', ylabel='σ',title='Диффузная составляющая')
#axarr[1].set(xlabel='Номер окна  (n)', ylabel='μ',title='Динамическая составляющая')
#axarr[2].plot(range(257), vol)
#axarr[3].plot(range(257), close)
plt.show()

