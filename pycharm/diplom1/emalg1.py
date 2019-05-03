import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import math
import itertools
import pickle
import datetime
from numba import jit, cuda, float32, float64


def set_initial_guess(xdata, k):
    np.random.seed(1234)
# set pi mu sigma
    pis = np.random.random(k)
    pis *= 0.9
    pis += 0.1
    pis /= np.sum(pis)
# mu
    mus = np.zeros(k)
# sigma
    sigmas = np.random.random(k)
    sigmas *= 1.5
    sigmas += 0.25
    sigmas *= np.std(xdata)
    return pis, mus, sigmas

df = pd.read_csv('RI.IMOEX_180323_180424_5min.csv', sep=';')
# drop last from every day
df = df.groupby('<DATE>').apply(lambda x: x.iloc[:-2] if len(x) > 1 else x).reset_index(drop=True)

data = []
prevopen = 0
for index, row in df.iterrows():
    if prevopen != 0:
        adata = row[4] - prevopen
        data = np.append(data, adata)
    prevopen = row[4]

data = data[0:10]
k = 10

pis = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
mus = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
sigmas = [0.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0]

set_initial_guess(data, 10)

tol = 0.01
max_iterations = 100
n = len(data)
# ем алг
ll_old = 0
ll_new = 0
for i in range(max_iterations):
    print(f'EM iteration {i}')
# E step
    ws = np.zeros((k, n))
    for j in range(len(mus)):
        for l in range(n):
            if i == 28 and j == 6:
                print(f" w={ws[j,l]}, u={mus[j]} d={data[l]}")
            ws[j, l] = pis[j] * sts.norm(mus[j], sigmas[j]).pdf(data[l])
            if ws[j,l] == 0:
                print(f" ws = {ws[j, l]} j={j} l={l}")
    ws /= ws.sum(0)

# M-step
    pis = np.zeros(k)
    for j in range(len(mus)):
        for l in range(n):
            pis[j] += ws[j, l]
    pis /= n

    mus = np.zeros(k)
    for j in range(k):
        for l in range(n):
            mus[j] += ws[j, l] * data[l]
        mus[j] /= ws[j, :].sum()
    if i == 28:
        print("A")
    sigmas = np.zeros(k)
    for j in range(k):
        for l in range(n):
            sigmas[j] += ws[j, l] * ((data[l] - mus[j]) ** 2)
        sigmas[j] /= ws[j, :].sum()
        sigmas[j] = math.sqrt(sigmas[j])

    print(f'pis {pis}')
    print(f'mus {mus}')
    print(f'sigmas {sigmas}')

    #
    # update complete log likelihoood
    ll_new = 0.0
    for l in range(n):
        s = 0.0
        for j in range(k):
            s += pis[j] * sts.norm(mus[j], sigmas[j]).pdf(data[l])
        ll_new += np.log(s)

    if np.abs(ll_new - ll_old) < tol:
        break
    # print 'll = {0}\n'.format(ll_new)
    ll_old = ll_new