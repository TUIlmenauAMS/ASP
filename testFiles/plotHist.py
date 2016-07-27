# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import matplotlib.pyplot as plt
import numpy as np

def showHist(x, bins =  250):
    plt.figure()
    hist, bins = np.histogram(x, bins=bins, density = False)
    width = 0.25 * (bins[1] - bins[0])
    #center = bins[:-1]
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist/np.trapz(hist), align='center', width=width)
    #plt.plot(center, hist)
    plt.show(block = False)