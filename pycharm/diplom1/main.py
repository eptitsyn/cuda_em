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
import datetime
import math
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
def em_alg_decomposition(iter, data, k, pis, mus, sigmas, tol = 0.01, max_iterations = 100):
    n = len(data)
    # ем алг
    ll_old = 0
    for i in range(max_iterations):
        #print('{3} [{0}:{1}]EM iteration {2}'.format(iter, os.getpid(), i, str(datetime.datetime.now())))
# E step
        ws = np.zeros((k, n))
        for j in range(len(mus)):
            for i in range(n):
                if sigmas[j] != 0:
                    ws[j, i] = pis[j] * sts.norm(mus[j], sigmas[j]).pdf(data[i])
                elif mus[j]-data[i] == 0:
                    ws[j, i] = pis[j]
                else:
                    ws[j, i] = 0
        #
        # update complete log likelihoood

        ll_new = sum(np.log(ws.sum(0)))
        if np.abs(ll_new - ll_old) < tol:
            break
        # print 'll = {0}\n'.format(ll_new)
        ll_old = ll_new

        b = ws.sum(0)
        ws = np.divide(ws, b, out=np.zeros_like(ws), where=b!=0)
        #ws /= ws.sum(0)
# M-step
        pis = np.zeros(k)
        mus = np.zeros(k)
        sigmas = np.zeros(k)

        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j, i]
        if pis[j] != 0:
            for j in range(k):
                for i in range(n):
                    mus[j] += ws[j, i] * data[i]
                mus[j] /= pis[j]
            for j in range(k):
                for i in range(n):
                    sigmas[j] += ws[j, i] * ((data[i] - mus[j]) ** 2)
                sigmas[j] /= pis[j]
                sigmas[j] = math.sqrt(sigmas[j])

            pis /= n
    #print('pis {0}'.format(pis))
    #print('mus {0}'.format(mus))
    #print('sigmas {0}'.format(sigmas))

    print("{2} end wkr (iter={3}) {0}:{1}\n".format(iter, os.getpid(), str(datetime.datetime.now()), i))

    return ll_new, pis, mus, sigmas


def worker_task(iter, zdata, k, pis, mus, sigmas):
    print("{2} start worker {0}:{1}\n".format(iter, os.getpid(), str(datetime.datetime.now())))
    pis, mus, sigmas = set_initial_guess(zdata, k)
    ll, pi, mu, sigma = em_alg_decomposition(iter, zdata, k, pis, mus, sigmas)
    return [ll, pi, mu, sigma]


def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return worker_task(*a_b)

@jit
def preparedatas():
    winlen = 336
    datas = []
    #for x in range(len(df) - winlen):
    for x in range(178, 312):
        print(x)
        xf = df[x:x+winlen]
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

def preparedatas2(data):
    winlen = 1040
    winstep = 1
    datas = []
    for x in range(0, len(data)-winlen, winstep):
        print(x/winstep)
        datas.append(data[x:x+winlen])
    return datas


# main
if __name__ == '__main__':
    pool = Pool(processes=16)  # start 8 worker processes

    """
# init
    df = pd.read_csv('RI.IMOEX_180323_180424_5min.csv', sep=';')
    # drop last from every day
    df = df.groupby('<DATE>').apply(lambda x: x.iloc[:-2] if len(x) > 1 else x).reset_index(drop=True)

# prepare data
    datas = preparedatas()
    """
    f = open('d://tmp//base_2.txt', 'r')
    x = f.readlines()
    f.close()
    data = []
    for i in range(len(x)):
        data = np.append(data, float(x[i][:-1]))

    datas = preparedatas2(data)


# start
    k = 10

# set initial guess
    pis, mus, sigmas = set_initial_guess(datas[0], k)
    result = pool.map(func_star, zip(range(len(datas)), datas, itertools.repeat(k), itertools.repeat(pis), itertools.repeat(mus), itertools.repeat(sigmas)))

    print(result)
    serialized = pickle.dump(result, open("Output_test.txt", "wb"))
