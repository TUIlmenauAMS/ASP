# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st
from distributions import logCauchy, weibull

def showHist(x, bins =  250):
    plt.close()
    plt.figure()
    hist, bins = np.histogram(x, bins=bins, density = True, normed = False)
    width = 0.3 * (bins[1] - bins[0])
    #center = bins[:-1]
    center = (bins[:-2] + bins[2:]) / 2
    hist = hist[1:]
    fc = np.trapz(hist)
    dist = getattr(st, 'gamma')

    params = (0.95, 0., 0.17)   # gamma voice
    params = (1., 0., 0.14)     # gamma bass
    params = (0.65, 0., 0.2)    # gamma drums
    params = (1.2, 0., 0.1)     # gamma other
    params = (1.8, 0., 0.15)    # gamma mixture

    params = (0.65, 0.22)       # gaussian! magnitude mixture
    params = (8.73, 0.074)      # gamma approximation

    pdf_fitted = dist.pdf(center, *params)
    plt.bar(center, hist, align='center', width=width)
    plt.plot(center, pdf_fitted, 'r')
    plt.show(block = False)