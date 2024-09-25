import numpy as np
from scipy.stats import norm
import torch
def num2vect(x, sigma):
    '''
    x: list of scalar (0-35) 
    sigma = 0 -> hard label (one-hot encoding) 
    sigma > 0 -> soft label (softmax)
    '''

    start = 0
    end = 30
    if len(x) == 1:
        x = int(x)
        v = np.zeros((end - start))
        centers = (np.arange(end) + 0.5)
        if np.isscalar(x):
            if sigma == 0:
                v[x] = 1
                return torch.from_numpy(v) 
            elif sigma > 0:
                for i in range(end):
                    x1 = centers[i] - 0.5
                    x2 = centers[i] + 0.5
                    cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                    v[i] = cdfs[1] - cdfs[0]
                return torch.from_numpy(v) 
        else:
            raise("x needs to be a scaler !")
    else:
        v = np.zeros((len(x), (end - start)))
        for idx, q in enumerate(x):
            q = int(q)
            centers = (np.arange(end) + 0.5)
            if np.isscalar(q):
                if sigma == 0:
                    v[idx, q] = 1
                    
                elif sigma > 0:
                    for i in range(end):
                        x1 = centers[i] - 0.5
                        x2 = centers[i] + 0.5
                        cdfs = norm.cdf([x1, x2], loc=q, scale=sigma)
                        v[idx, i] = cdfs[1] - cdfs[0]
            else:
                raise("x needs to be a scaler !")
        return torch.from_numpy(v) 
        

if __name__ == "__main__":

    print(num2vect(4, sigma=1))
    '''
    [1.31822679e-03 2.14002339e-02 1.35905122e-01 3.41344746e-01
    3.41344746e-01 1.35905122e-01 2.14002339e-02 1.31822679e-03
    3.13845903e-05 2.85664984e-07 9.85307835e-10 1.27919897e-12
    6.66133815e-16 0.00000000e+00 0.00000000e+00 0.00000000e+00
    0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    0.00000000e+00 0.00000000e+00 0.00000000e+00]
    '''