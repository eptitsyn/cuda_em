from __future__ import division
from numba import cuda, float32
import numpy as np
import math

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


@cuda.jit
def em_alg(xs, pis, mus, sigmas):
    sXS = cuda.shared.array((len(xs), 1), dtype=float32)


slswindowlen = 7

# init
df = pd.read_csv('RI.IMOEX_140101_140131.csv', sep=';')
# drop last from every day
df = df.groupby('<DATE>').apply(lambda x: x.iloc[:-1] if len(x) > 1 else x).reset_index(drop=True)
a = df['<DATE>'].unique()

# prepare data
data = []

xf = df.loc[df['<DATE>'].isin(a[x:x + slswindowlen])]
data = np.array([])

prevopen = 0
for index, row in xf.iterrows():
    if prevopen != 0:
        data = np.append(data, row[4] - prevopen)
    prevopen = row[4]
datas.append(data)

# start
k = 10

# set initial guess
pis, mus, sigmas = set_initial_guess(data, k)
em_alg(x, pis,mus, sigmas)
print(result)
serialized = pickle.dumps(result, protocol=0)

with open("Output.txt", "wb") as text_file:
    text_file.write(serialized)
