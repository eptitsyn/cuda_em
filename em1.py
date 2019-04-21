# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import math


df=pd.read_csv('RI.IMOEX_171001_181001.csv', sep=';')

def strtofloat(s):
    s = s.replace(",","")
    return float(s)

data = np.array([])
prevopen = 0
for index,row in df.iterrows():
    if prevopen != 0:
        adata = strtofloat(row[4])-prevopen
        if (abs(adata)< 30):
            data = np.append(data, adata)
    prevopen = strtofloat(row[4])
data = np.append(data,strtofloat(df.iloc[-1][7])-prevopen)
'''
dataa = np.random.normal(0, 6, 500)
datab = np.random.normal(15, 8, 500)
data = np.concatenate((dataa, datab))
'''

def pdfcombine(x, pis, mus, sigmas):
    return pis[0]*sts.norm(mus[0], sigmas[0]).pdf(x)+pis[1]*sts.norm(mus[1], sigmas[1]).pdf(x)


u, s = 0., 100
x = np.linspace(u-s,u+s,1000)
plt.hist(data, bins=300, normed=True)
plt.ylabel('fraction of samples')
plt.xlabel('$x$')
plt.legend(loc='upper right')


print ('EM Start\n')

pis = np.array([0.5, 0.5])
mus = np.array([0,0])
sigmas = np.array([6,9])

k = 2
n = len(data)
tol = 0.001
p = 1

#ем алг
ll_old = 0
for i in range(100):
	print 'EM iteration {0}'.format(i)
	ll_new = 0
    #E step
	ws = np.zeros((k, n))
	for j in range(len(mus)):
		for i in range(n):
			ws[j, i] = pis[j] * sts.norm(mus[j],  sigmas[j]).pdf(data[i])
	ws /= ws.sum(0)

    # M-step
        pis = np.zeros(k)
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j, i]
        pis /= n

	mus = np.zeros(k)
	for j in range(k):
		for i in range(n):
			mus[j] += ws[j, i] * data[i]
		mus[j] /= ws[j, :].sum()

	sigmas = np.zeros(k)
	for j in range(k):
		for i in range(n):
			sigmas[j] += ws[j, i] * ((data[i]- mus[j])**2)
		sigmas[j] /= ws[j,:].sum()
		sigmas[j] = math.sqrt(sigmas[j])

	print 'pis {0}, {1}\n'.format(pis[0], pis[1])
	print 'mus {0}, {1}\n'.format(mus[0], mus[1])
	print 'sigmas {0}, {1}\n'.format(sigmas[0], sigmas[1])

	#
    # update complete log likelihoood
	ll_new = 0.0
	for i in range(n):
		s = 0
		for j in range(k):
			s += pis[j] * sts.norm(mus[j],  sigmas[j]).pdf(data[i])
		ll_new += np.log(s)

	if np.abs(ll_new - ll_old) < tol:
		break
	ll_old = ll_new


import sys
sys.stdout.flush()
#plt.plot(x, sts.norm(mus[0], sigmas[0]).pdf(x), label='asd')
#plt.plot(x, sts.norm(mus[1], sigmas[1]).pdf(x), label='asd')
plt.plot(x, pdfcombine(x, pis, mus, sigmas), label='asd')
plt.show()