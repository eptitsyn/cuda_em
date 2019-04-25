# -*- coding: utf-8 -*-
from multiprocessing import Pool, TimeoutError
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import math
import itertools
import pickle
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

@jit
def em_alg_decomposition(iter, data, k, pis, mus, sigmas, tol = 0.01, max_iterations = 2):
    n = len(data)
    # ем алг
    ll_old = 0
    ll_new = 0
    for i in range(max_iterations):
        print('[{0}:{1}]EM iteration {2}'.format(iter,os.getpid(),i))
# E step
        ws = np.zeros((k, n))
        for j in range(len(mus)):
            for i in range(n):
                ws[j, i] = pis[j] * sts.norm(mus[j], sigmas[j]).pdf(data[i])
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
                sigmas[j] += ws[j, i] * ((data[i] - mus[j]) ** 2)
            sigmas[j] /= ws[j, :].sum()
            sigmas[j] = math.sqrt(sigmas[j])

        #print 'pis {0}'.format(pis)
        #print 'mus {0}'.format(mus)
        #print 'sigmas {0}'.format(sigmas)

        #
        # update complete log likelihoood
        ll_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pis[j] * sts.norm(mus[j], sigmas[j]).pdf(data[i])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < tol:
            break
        # print 'll = {0}\n'.format(ll_new)
        ll_old = ll_new
    return ll_new, pis, mus, sigmas


def worker_task(iter, zdata, k, pis, mus, sigmas):
    print("start worker {0}:{1}\n".format(iter, os.getpid()))
    pis, mus, sigmas = set_initial_guess(zdata, k)
    ll, pi, mu, sigma = em_alg_decomposition(iter, zdata, k, pis, mus, sigmas)
    return [ll, pi, mu, sigma]


def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return worker_task(*a_b)

@jit
def preparedatas():
    datas = []
    for x in range(len(a) - slswindowlen):
        print(x)
        xf = df.loc[df['<DATE>'].isin(a[x:x + slswindowlen])]
        data = np.array([])
        prevopen = 0
        for index, row in xf.iterrows():
            if prevopen != 0:
                adata = row[4] - prevopen
                #            if (abs(adata)< 30):
                data = np.append(data, adata)
            prevopen = row[4]
        #    data = np.append(data,df.iloc[-1][7]-prevopen)
        datas.append(data)
    return datas


# main
if __name__ == '__main__':
    slswindowlen = 7
    pool = Pool(processes=8)  # start 8 worker processes

# init
    df = pd.read_csv('RI.IMOEX_140101_140131.csv', sep=';')
    # drop last from every day
    df = df.groupby('<DATE>').apply(lambda x: x.iloc[:-1] if len(x) > 1 else x).reset_index(drop=True)
    a = df['<DATE>'].unique()

# prepare data
    datas = preparedatas()


# start
    k = 10

# set initial guess
    pis, mus, sigmas = set_initial_guess(datas[0], k)
    result = pool.map(func_star, zip(range(len(datas)), datas, itertools.repeat(k), itertools.repeat(pis), itertools.repeat(mus), itertools.repeat(sigmas)))

    print(result)
    serialized = pickle.dump(result, open("Output1.txt", "wb"))
