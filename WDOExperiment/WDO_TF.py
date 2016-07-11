# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import IOMethods as IO
import TFMethods as TF
import numpy as np
import scipy.signal as sig
import os
import QMF.qmf_realtime_class as qrf
import matplotlib.pyplot as plt
import spams, math
from multiprocessing import Process
from MaskingMethods import FrequencyMasking as fm
from matplotlib import rcParams

eps = np.finfo(np.float32).tiny

# STFT
def eval_stft_hann(alpha, mixKaraoke, vox):
    MixmX, _ = TF.TimeFrequencyDecomposition.STFT(mixKaraoke, w = sig.hanning(1024, False), N = 1024, hop = 512)
    VoxmX, _ = TF.TimeFrequencyDecomposition.STFT(vox, w = sig.hanning(1024, False), N = 1024, hop = 512)

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(MixmX + VoxmX), np.abs(VoxmX), np.abs(MixmX), [], [], alpha = alpha, method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(VoxmX))
    SIR = TF.WDODisjointness.SIR(M, np.abs(VoxmX), np.abs(MixmX))

    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def eval_stft_bt(alpha, mixKaraoke, vox):
    MixmX, _ = TF.TimeFrequencyDecomposition.STFT(mixKaraoke, w = sig.bartlett(1024, False), N = 1024, hop = 512)
    VoxmX, _ = TF.TimeFrequencyDecomposition.STFT(vox, w = sig.bartlett(1024, False), N = 1024, hop = 512)

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(MixmX + VoxmX), np.abs(VoxmX), np.abs(MixmX), [], [], alpha = alpha, method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(VoxmX))
    SIR = TF.WDODisjointness.SIR(M, np.abs(VoxmX), np.abs(MixmX))

    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def eval_stft_nt(alpha, mixKaraoke, vox):
    MixmX, _ = TF.TimeFrequencyDecomposition.STFT(mixKaraoke, w = TF.TimeFrequencyDecomposition.nuttall4b(1024, False), N = 1024, hop = 256)
    VoxmX, _ = TF.TimeFrequencyDecomposition.STFT(vox, w = TF.TimeFrequencyDecomposition.nuttall4b(1024, False), N = 1024, hop = 256)

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(MixmX + VoxmX), np.abs(VoxmX), np.abs(MixmX), [], [], alpha = alpha, method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(VoxmX))
    SIR = TF.WDODisjointness.SIR(M, np.abs(VoxmX), np.abs(MixmX))

    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def eval_stft_ntB(alpha, mixKaraoke, vox):
    MixmX, _ = TF.TimeFrequencyDecomposition.STFT(mixKaraoke, w = TF.TimeFrequencyDecomposition.nuttall4b(1024, False), N = 1024, hop = 512)
    VoxmX, _ = TF.TimeFrequencyDecomposition.STFT(vox, w = TF.TimeFrequencyDecomposition.nuttall4b(1024, False), N = 1024, hop = 512)

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(MixmX + VoxmX), np.abs(VoxmX), np.abs(MixmX), [], [], alpha = alpha, method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(VoxmX))
    SIR = TF.WDODisjointness.SIR(M, np.abs(VoxmX), np.abs(MixmX))

    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def spc_eval_stft_hann(alpha, mixKaraoke, vox):
    MixmX, _ = TF.TimeFrequencyDecomposition.STFT(mixKaraoke + vox, w = sig.hanning(1024, False), N = 1024, hop = 512)

    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index(MixmX))

    return SPCMix

def spc_eval_stft_bt(alpha, mixKaraoke, vox):
    MixmX, _ = TF.TimeFrequencyDecomposition.STFT(mixKaraoke + vox, w = sig.bartlett(1024, False), N = 1024, hop = 512)

    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index(MixmX))

    return SPCMix

def spc_eval_stft_nt(alpha, mixKaraoke, vox):
    MixmX, _ = TF.TimeFrequencyDecomposition.STFT(mixKaraoke + vox, w = TF.TimeFrequencyDecomposition.nuttall4b(1024, False), N = 1024, hop = 256)

    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index(MixmX))

    return SPCMix

def spc_eval_stft_ntB(alpha, mixKaraoke, vox):
    MixmX, _ = TF.TimeFrequencyDecomposition.STFT(mixKaraoke + vox, w = TF.TimeFrequencyDecomposition.nuttall4b(1024, False), N = 1024, hop = 512)

    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index(MixmX))

    return SPCMix

# MDCT
def eval_mdct(alpha, mixKaraoke, vox):

    MixmX = TF.TimeFrequencyDecomposition.complex_analysis(mixKaraoke, N = 1024)
    VoxmX = TF.TimeFrequencyDecomposition.complex_analysis(vox, N = 1024)

    MixmX = np.real(MixmX)
    VoxmX = np.real(VoxmX)

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(MixmX + VoxmX), np.abs(VoxmX), np.abs(MixmX), [], [], alpha = alpha, method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(VoxmX))
    SIR = TF.WDODisjointness.SIR(M, np.abs(VoxmX), np.abs(MixmX))

    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def eval_mdcst_complex(alpha, mixKaraoke, vox):

    MixmX = TF.TimeFrequencyDecomposition.complex_analysis(mixKaraoke, N = 1024)
    VoxmX = TF.TimeFrequencyDecomposition.complex_analysis(vox, N = 1024)

    # Acquire Magnitude from Complex
    MixmX = np.abs(MixmX)
    VoxmX = np.abs(VoxmX)

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(MixmX + VoxmX), np.abs(VoxmX), np.abs(MixmX), [], [], alpha = alpha, method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(VoxmX))
    SIR = TF.WDODisjointness.SIR(M, np.abs(VoxmX), np.abs(MixmX))

    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def spc_eval_mdct(alpha, mixKaraoke, vox):

    MixmX = TF.TimeFrequencyDecomposition.complex_analysis(mixKaraoke + vox, N = 1024)

    MixmX = np.abs(np.real(MixmX))

    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index(MixmX))

    return SPCMix

def spc_eval_mdcst_complex(alpha, mixKaraoke, vox):

    MixmX = TF.TimeFrequencyDecomposition.complex_analysis(mixKaraoke + vox, N = 1024)

    # Acquire Magnitude from Complex
    MixmX = np.abs(MixmX)

    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index((MixmX)))
    return SPCMix

# PQMF
def eval_pqmf_cos(alpha, mixKaraoke, vox):

    N = 1024
    timeSlots = len(mixKaraoke)/N
    karms = np.empty((timeSlots, N), dtype = np.float32)
    vs = np.empty((timeSlots, N), dtype = np.float32)

    qrf.reset_rt()
    for indx in xrange(timeSlots):
            karms[indx, :] = qrf.PQMFAnalysis.analysisqmf_realtime(mixKaraoke[indx * N : (indx + 1) * N], N)

    qrf.reset_rt()
    for indx in xrange(timeSlots):
            vs[indx, :] = qrf.PQMFAnalysis.analysisqmf_realtime(vox[indx * N : (indx + 1) * N], N)

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(karms + vs), np.abs(vs), np.abs(karms), [], [], alpha = alpha, method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(vs))
    SIR = TF.WDODisjointness.SIR(M, np.abs(vs), np.abs(karms))

    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def eval_pqmf_complex(alpha, mixKaraoke, vox):

    N = 1024
    timeSlots = len(mixKaraoke)/N
    karms = np.empty((timeSlots, N), dtype = np.complex64)
    vs = np.empty((timeSlots, N), dtype = np.complex64)

    qrf.reset_rt()
    for indx in xrange(timeSlots):
            karms[indx, :] = qrf.PQMFAnalysis.complex_analysis_realtime(mixKaraoke[indx * N : (indx + 1) * N], N)

    qrf.reset_rt()
    for indx in xrange(timeSlots):
            vs[indx, :] = qrf.PQMFAnalysis.complex_analysis_realtime(vox[indx * N : (indx + 1) * N], N)

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(karms + vs), np.abs(vs), np.abs(karms), [], [], alpha = alpha, method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(vs))
    SIR = TF.WDODisjointness.SIR(M, np.abs(vs), np.abs(karms))

    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def spc_eval_pqmf_cos(alpha, mixKaraoke, vox):

    N = 1024
    timeSlots = len(mixKaraoke)/N
    karms = np.empty((timeSlots, N), dtype = np.float32)
    vs = np.empty((timeSlots, N), dtype = np.float32)
    mix = mixKaraoke + vox

    qrf.reset_rt()
    for indx in xrange(timeSlots):
            karms[indx, :] = qrf.PQMFAnalysis.analysisqmf_realtime(mix[indx * N : (indx + 1) * N], N)


    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index(np.abs(ms)))

    return SPCMix

def spc_eval_pqmf_complex(alpha, mixKaraoke, vox):

    N = 1024
    timeSlots = len(mixKaraoke)/N
    karms = np.empty((timeSlots, N), dtype = np.complex64)
    vs = np.empty((timeSlots, N), dtype = np.complex64)
    mix = mixKaraoke + vox

    qrf.reset_rt()
    for indx in xrange(timeSlots):
            karms[indx, :] = qrf.PQMFAnalysis.complex_analysis_realtime(mix[indx * N : (indx + 1) * N], N)

    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index(np.abs(ms)))

    return SPCMix

# OMP
def OLAnalysis(x, w, N, hop):

    # Analysis Parameters
    wsz = w.size
    print(wsz)
    hw1 = int(math.floor((wsz+1)/2))
    hw2 = int(math.floor(wsz/2))

    # Add some zeros at the start and end of the signal to avoid window smearing
    x = np.append(np.zeros(3*hop),x)
    x = np.append(x, np.zeros(3*hop))

    xol = np.empty((len(x)/hop, N))

    # Initialize sound pointers
    pin = 0

    pend = x.size - wsz

    indx = 0

    while pin <= pend:
        xSeg = x[pin:pin+wsz] * w
        xol[indx, :] = xSeg

        indx += 1
        pin += hop

    return xol

def OLAPSynth(xmX, wsz, hop):

    # Acquire half window sizes
    hw1 = int(math.floor((wsz+1)/2))
    hw2 = int(math.floor(wsz/2))

    # Acquire the number of STFT frames
    numFr = xmX.shape[0]

    # Initialise output array with zeros
    y = np.zeros(numFr * hop + hw1 + hw2)

    # Initialise sound pointer
    pin = 0

    # Main Synthesis Loop
    for indx in range(numFr):
        # Inverse Discrete Fourier Transform
        ybuffer = xmX[indx, :]

        # Overlap and Add
        y[pin:pin+wsz] += ybuffer

        # Advance pointer
        pin += hop

    # Delete the extra zeros that the analysis had placed
    y = np.delete(y, range(np.int(3*hop)))
    y = np.delete(y, range(np.int(y.size-(3*hop + 1)), y.size))

    return y

def eval_dctdst_mp(mixKaraoke, vox):
    nSamples = 2048
    N = 512
    winsamples = N * 2

    # Compute union of various DCT lengths for Dictionary
    win = np.bartlett(winsamples)
    cCos, cSin = TF.TimeFrequencyDecomposition.coreModulation(win, N)
    cCos = cCos.T
    cSin = cSin.T
    cbCos = np.zeros((nSamples, cCos.shape[1]))
    cbCos[:cCos.shape[0], :] = cCos
    cbSin = np.zeros((nSamples, cSin.shape[1]))
    cbSin[:cSin.shape[0], :] = cSin
    U = np.hstack((cbCos, cbSin))

    # Create Dictionary
    CS2 = np.asfortranarray(U, (np.float32))

    # Ovelapped analysis
    win = np.bartlett(nSamples)
    karmix = np.asfortranarray(OLAnalysis(mixKaraoke, win, nSamples, nSamples/4).T, np.float32)
    vox = np.asfortranarray(OLAnalysis(vox, win, nSamples, nSamples/4).T, np.float32)

    # Acquire coefficients using orthogonal matching pursuit
    ms = np.asarray(spams.omp(karmix, CS2, 210, 1e-16).todense()).T
    vs = np.asarray(spams.omp(vox, CS2, 210, 1e-16).todense()).T

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(ms + vs), np.abs(vs), np.abs(ms), [], [], alpha = 1., method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(vs))
    SIR = TF.WDODisjointness.SIR(M, np.abs(vs), np.abs(ms))
    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def spc_eval_dctdst_mp(mixKaraoke, vox):
    nSamples = 2048
    N = 512
    winsamples = N * 2

    # Compute union of various DCT lengths for Dictionary
    win = np.bartlett(winsamples)
    cCos, cSin = TF.TimeFrequencyDecomposition.coreModulation(win, N)
    cCos = cCos.T
    cSin = cSin.T
    cbCos = np.zeros((nSamples, cCos.shape[1]))
    cbCos[:cCos.shape[0], :] = cCos
    cbSin = np.zeros((nSamples, cSin.shape[1]))
    cbSin[:cSin.shape[0], :] = cSin
    U = np.hstack((cbCos, cbSin))

    # Create Dictionary
    CS2 = np.asfortranarray(U, (np.float32))

    # Ovelapped analysis
    win = np.bartlett(nSamples)
    karmix = np.asfortranarray(OLAnalysis(mixKaraoke + vox, win, nSamples, nSamples/4).T, np.float32)

    # Acquire coefficients using orthogonal matching pursuit
    ms = np.asarray(spams.omp(karmix, CS2, 210, 1e-16).todense()).T

    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index(np.abs(ms)))

    return SPCMix

def eval_dct_union_mp_ovdt(mixKaraoke, vox):
    # Union overdetermined
    nSamples = 2048
    N = 2 ** np.arange(6,10)
    N = np.hstack((64,N))
    winsamples = N * 2

    # Compute union of various DCT lengths for Dictionary
    for indx in xrange(N.size) :
        win = sig.cosine(winsamples[indx], False)
        cCos, _ = TF.TimeFrequencyDecomposition.coreModulation(win, N[indx])
        cCos = cCos.T
        cbCos = np.zeros((nSamples, cCos.shape[1]))
        if indx == 0:
            cbCos[:cCos.shape[0], :] = cCos
            Cos = cbCos[:, :]
        else :
            cbCos[cCos.shape[0]:2 * cCos.shape[0], :] = cCos
            Cos = np.hstack((Cos, cbCos))

    CS2 = np.asfortranarray(Cos, (np.float32))
    N = 1024
    winsamples = nSamples
    win = sig.cosine(winsamples, False)
    win /= np.sum(win)

    # Ovelapped analysis
    karmix = np.asfortranarray(OLAnalysis(mixKaraoke, win, winsamples, N/8).T, np.float32)
    vox = np.asfortranarray(OLAnalysis(vox, win, winsamples, N/8).T, np.float32)

    # Acquire coefficients using orthogonal matching pursuit
    ms = np.asarray(spams.omp(karmix, CS2, 360, 0.).todense()).T
    vs = np.asarray(spams.omp(vox, CS2, 360, 0.).todense()).T

    # Compute Upper Bound Binary Mask
    mask = fm(np.abs(ms + vs), np.abs(vs), np.abs(ms), [], [], alpha = 1., method = 'UBBM')

    # Activate the method to acquire the mask
    vsf = mask()
    M = mask._mask

    # Compute the measures used in WDO
    PSR = TF.WDODisjointness.PSR(M, np.abs(vs))
    SIR = TF.WDODisjointness.SIR(M, np.abs(vs), np.abs(ms))

    WDO = TF.WDODisjointness.WDO(PSR, SIR)

    return WDO

def spc_eval_dct_union_mp_ovdt(mixKaraoke, vox):
    # Union overdetermined
    nSamples = 2048
    N = 2 ** np.arange(6,10)
    N = np.hstack((64,N))
    winsamples = N * 2

    # Compute union of various DCT lengths for Dictionary
    for indx in xrange(N.size) :
        win = sig.cosine(winsamples[indx], False)
        cCos, _ = TF.TimeFrequencyDecomposition.coreModulation(win, N[indx])
        cCos = cCos.T
        cbCos = np.zeros((nSamples, cCos.shape[1]))
        if indx == 0:
            cbCos[:cCos.shape[0], :] = cCos
            Cos = cbCos[:, :]
        else :
            cbCos[cCos.shape[0]:2 * cCos.shape[0], :] = cCos
            Cos = np.hstack((Cos, cbCos))

    CS2 = np.asfortranarray(Cos, (np.float32))
    N = 1024
    winsamples = nSamples
    win = sig.cosine(winsamples, False)
    win /= np.sum(win)

    # Ovelapped analysis
    karmix = np.asfortranarray(OLAnalysis(mixKaraoke + vox, win, winsamples, N/8).T, np.float32)

    # Acquire coefficients using orthogonal matching pursuit
    ms = np.asarray(spams.omp(karmix, CS2, 360, 0.).todense()).T

    # Compute Sparsity Criteria
    SPCMix = np.mean(TF.WDODisjointness.gini_index(np.abs(ms)))

    return SPCMix

# Result Loading
def load_results():
    # STFT
    # Hanning
    stft_hann_a1 = np.load('WDOExperiment/stft_hann_a1.npy')

    # Bartlett
    stft_bt_a1 = np.load('WDOExperiment/stft_bt_a1.npy')

    # Nuttall-4b
    stft_nt_a1 = np.load('WDOExperiment/stft_nt_a1.npy')
    stft_ntB_a1 = np.load('WDOExperiment/stft_ntB_a1.npy')

    # MDCT
    mdct_a1 = np.load('WDOExperiment/mdct_a1.npy')
    mdcst_a1 = np.load('WDOExperiment/mdcst_a1.npy')

    # PQMF
    pqmf_cos_a1 = np.load('WDOExperiment/pqmf_cos_a1.npy')
    pqmf_compl_a1 = np.load('WDOExperiment/pqmf_compl_a1.npy')

    # Matching Pursuit
    mdct_union_mp_ovdt = np.load('WDOExperiment/mdct_union_mp_ovdt.npy')
    mdctdst_union_mp_ovdt = np.load('WDOExperiment/mdctdst_union_mp_ovdt.npy')

    return stft_hann_a1,stft_bt_a1, stft_nt_a1, stft_ntB_a1, mdct_a1, mdcst_a1, pqmf_cos_a1,\
           pqmf_compl_a1, mdct_union_mp_ovdt, mdctdst_union_mp_ovdt

def load_SPCresults():
    # STFT
    # Hanning
    spc_stft_hann = np.load('WDOExperiment/spc_stft_hann_a1.npy')

    # Bartlett
    spc_stft_bt = np.load('WDOExperiment/spc_stft_bt_a1.npy')

    # Nuttall-4b
    spc_stft_nt = np.load('WDOExperiment/spc_stft_nt_a1.npy')
    spc_stft_ntB = np.load('WDOExperiment/spc_stft_ntB_a1.npy')

    # MDCT
    spc_mdct = np.load('WDOExperiment/spc_mdct_a1.npy')
    spc_mdcst = np.load('WDOExperiment/spc_mdcst_a1.npy')

    # PQMF
    spc_pqmf = np.load('WDOExperiment/spc_pqmf_cos_a1.npy')
    spc_pqmf_comp = np.load('WDOExperiment/spc_pqmf_compl_a1.npy')

    # Matching Pursuit
    #mdct_union_mp_ovdt = np.load('WDOExperiment/spc_mdct_union_mp_ovdt.npy')
    #mdctdst_union_mp_ovdt = np.load('WDOExperiment/spc_mdctdst_union_mp_ovdt.npy')

    return spc_stft_hann, spc_stft_bt, spc_stft_nt, spc_stft_ntB, spc_mdct,\
           spc_mdcst, spc_pqmf, spc_pqmf_comp#, mdct_union_mp_ovdt, mdctdst_union_mp_ovdt

# Main Operations
### WDO
def mainWDO(selection):
    # Paths & Names
    MixturesPath = '/home/avdata/audio/own/dsd100/DSD100/Mixtures/'
    SourcesPath = '/home/avdata/audio/own/dsd100/DSD100/Sources/'
    foldersList = ['Dev', 'Test']
    # Usage of full dataset
    #keywords = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav', 'mixture.wav']
    # Usage of segmented dataset
    keywords = ['bass_seg.wav', 'drums_seg.wav', 'other_seg.wav', 'vocals_seg.wav', 'mixture_seg.wav']

    # Generate full paths for dev and test
    DevMixturesList = sorted(os.listdir(MixturesPath + foldersList[0]))
    DevMixturesList = [MixturesPath+foldersList[0] + '/' + i for i in DevMixturesList]
    DevSourcesList = sorted(os.listdir(SourcesPath + foldersList[0]))
    DevSourcesList = [SourcesPath+foldersList[0] + '/' + i for i in DevSourcesList]

    TestMixturesList = sorted(os.listdir(MixturesPath + foldersList[1]))
    TestMixturesList = [MixturesPath+foldersList[1] + '/' + i for i in TestMixturesList]
    TestSourcesList = sorted(os.listdir(SourcesPath + foldersList[1]))
    TestSourcesList = [SourcesPath+foldersList[1] + '/' + i for i in TestSourcesList]

    # Extend Lists for full validation
    DevMixturesList.extend(TestMixturesList)
    DevSourcesList.extend(TestSourcesList)

    # Saving Index for storing the results
    saveIndx = 0

    # Number of sub-bands
    N = 1024

    # Initialize Results storing
    # Alpha = 1.
    stft_hann_a1 = np.zeros(100)
    stft_bt_a1 = np.zeros(100)
    stft_nt_a1 = np.zeros(100)
    stft_ntB_a1 = np.zeros(100)

    mdct_a1 = np.zeros(100)
    mdcst_a1 = np.zeros(100)
    pqmf_cos_a1 = np.zeros(100)
    pqmf_compl_a1 = np.zeros(100)

    # Matching Pursuit based decompositions
    dctdst_union_mp_ovdt = np.zeros(100)
    dct_union_mp_ovdt = np.zeros(100)

    for folderIndx in xrange(len(DevMixturesList)):

        #print('Reading:' + DevMixturesList[folderIndx])
        bss, fs = IO.AudioIO.wavRead(os.path.join(DevSourcesList[folderIndx], keywords[0]), mono = True)
        drm, fs = IO.AudioIO.wavRead(os.path.join(DevSourcesList[folderIndx], keywords[1]), mono = True)
        oth, fs = IO.AudioIO.wavRead(os.path.join(DevSourcesList[folderIndx], keywords[2]), mono = True)
        mixKaraoke = bss + drm + oth
        del bss, drm, oth
        vox, fs = IO.AudioIO.wavRead(os.path.join(DevSourcesList[folderIndx], keywords[3]), mono = True)
        mix, fs = IO.AudioIO.wavRead(os.path.join(DevMixturesList[folderIndx], keywords[4]), mono = True)

        # Evaluation inside file loop
        if selection == 'stft':
            print('STFT')
            # Alpha = 1.
            stft_hann_a1[saveIndx] = eval_stft_hann(1., mixKaraoke, vox)
            stft_bt_a1[saveIndx] = eval_stft_bt(1., mixKaraoke, vox)
            stft_nt_a1[saveIndx] = eval_stft_nt(1., mixKaraoke, vox)
            stft_ntB_a1[saveIndx] = eval_stft_ntB(1., mixKaraoke, vox)

            np.save('WDOExperiment/stft_hann_a1.npy', stft_hann_a1)
            np.save('WDOExperiment/stft_bt_a1.npy', stft_bt_a1)
            np.save('WDOExperiment/stft_nt_a1.npy', stft_nt_a1)
            np.save('WDOExperiment/stft_ntB_a1.npy', stft_ntB_a1)

        elif selection == 'mdct':
            print('MDCT')
            # Alpha = 1.
            mdct_a1[saveIndx] = eval_mdct(1., mixKaraoke, vox)
            mdcst_a1[saveIndx] = eval_mdcst_complex(1., mixKaraoke, vox)

            np.save('WDOExperiment/mdct_a1.npy', mdct_a1)
            np.save('WDOExperiment/mdcst_a1.npy', mdcst_a1)

        elif selection == 'pqmf':
            print('PQMF')
            # Alpha = 1.
            pqmf_cos_a1[saveIndx] = eval_pqmf_cos(1., mixKaraoke, vox)
            np.save('WDOExperiment/pqmf_cos_a1.npy', pqmf_cos_a1)

        elif selection == 'pqmf_complex':
            print('PQMF Complex')
            # Alpha = 1.
            pqmf_compl_a1[saveIndx] = eval_pqmf_complex(1., mixKaraoke, vox)
            np.save('WDOExperiment/pqmf_compl_a1.npy', pqmf_compl_a1)

        elif selection == 'mp':
            print('Matching Pursuit Based')
            #dctdst_union_mp_ovdt[saveIndx] = eval_dctdst_mp(mixKaraoke, vox)
            dct_union_mp_ovdt[saveIndx] = eval_dct_union_mp_ovdt(mixKaraoke, vox)

            np.save('WDOExperiment/mdctdst_union_mp_ovdt.npy', dctdst_union_mp_ovdt)
            np.save('WDOExperiment/mdct_union_mp_ovdt.npy', dct_union_mp_ovdt)

        else :
            assert('Unknown Selection!')

        # Update Storing Index
        saveIndx += 1

    return None

### SPC
def mainSPC(selection):
    # Paths & Names
    MixturesPath = '/home/avdata/audio/own/dsd100/DSD100/Mixtures/'
    SourcesPath = '/home/avdata/audio/own/dsd100/DSD100/Sources/'
    foldersList = ['Dev', 'Test']
    # Usage of full dataset
    #keywords = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav', 'mixture.wav']
    # Usage of segmented dataset
    keywords = ['bass_seg.wav', 'drums_seg.wav', 'other_seg.wav', 'vocals_seg.wav', 'mixture_seg.wav']

    # Generate full paths for dev and test
    DevMixturesList = sorted(os.listdir(MixturesPath + foldersList[0]))
    DevMixturesList = [MixturesPath+foldersList[0] + '/' + i for i in DevMixturesList]
    DevSourcesList = sorted(os.listdir(SourcesPath + foldersList[0]))
    DevSourcesList = [SourcesPath+foldersList[0] + '/' + i for i in DevSourcesList]

    TestMixturesList = sorted(os.listdir(MixturesPath + foldersList[1]))
    TestMixturesList = [MixturesPath+foldersList[1] + '/' + i for i in TestMixturesList]
    TestSourcesList = sorted(os.listdir(SourcesPath + foldersList[1]))
    TestSourcesList = [SourcesPath+foldersList[1] + '/' + i for i in TestSourcesList]

    # Extend Lists for full validation
    DevMixturesList.extend(TestMixturesList)
    DevSourcesList.extend(TestSourcesList)

    # Saving Index for storing the results
    saveIndx = 0

    # Number of sub-bands
    N = 1024

    # Initialize Results storing
    # Alpha = 1.
    stft_hann_a1 = np.zeros((100))
    stft_bt_a1 = np.zeros((100))
    stft_nt_a1 = np.zeros((100))
    stft_ntB_a1 = np.zeros((100))

    mdct_a1 = np.zeros((100))
    mdcst_a1 = np.zeros((100))
    pqmf_cos_a1 = np.zeros((100))
    pqmf_compl_a1 = np.zeros((100))


    mdct_union_mp_ovdt = np.zeros((100))
    dctdst_union_mp_ovdt = np.zeros((100))

    for folderIndx in xrange(len(DevMixturesList)):

        print('Reading:' + DevMixturesList[folderIndx])
        bss, fs = IO.AudioIO.wavRead(os.path.join(DevSourcesList[folderIndx], keywords[0]), mono = True)
        drm, fs = IO.AudioIO.wavRead(os.path.join(DevSourcesList[folderIndx], keywords[1]), mono = True)
        oth, fs = IO.AudioIO.wavRead(os.path.join(DevSourcesList[folderIndx], keywords[2]), mono = True)
        mixKaraoke = bss + drm + oth
        del bss, drm, oth
        vox, fs = IO.AudioIO.wavRead(os.path.join(DevSourcesList[folderIndx], keywords[3]), mono = True)
        mix, fs = IO.AudioIO.wavRead(os.path.join(DevMixturesList[folderIndx], keywords[4]), mono = True)

        # Evaluation inside file loop
        if selection == 'stft':
            print('STFT')
            # Alpha = 1.
            stft_hann_a1[saveIndx]= spc_eval_stft_hann(1., mixKaraoke, vox)
            stft_bt_a1[saveIndx] = spc_eval_stft_bt(1., mixKaraoke, vox)
            stft_nt_a1[saveIndx] = spc_eval_stft_nt(1., mixKaraoke, vox)
            stft_ntB_a1[saveIndx] = spc_eval_stft_ntB(1., mixKaraoke, vox)
            print(stft_hann_a1[saveIndx], stft_bt_a1[saveIndx], stft_nt_a1[saveIndx], stft_ntB_a1[saveIndx])

            np.save('WDOExperiment/spc_stft_hann_a1.npy', stft_hann_a1)
            np.save('WDOExperiment/spc_stft_bt_a1.npy', stft_bt_a1)
            np.save('WDOExperiment/spc_stft_nt_a1.npy', stft_nt_a1)
            np.save('WDOExperiment/spc_stft_ntB_a1.npy', stft_ntB_a1)


        elif selection == 'mdct':
            print('MDCT')
            # Alpha = 1.
            mdct_a1[saveIndx] = spc_eval_mdct(1., mixKaraoke, vox)
            mdcst_a1[saveIndx] = spc_eval_mdcst_complex(1., mixKaraoke, vox)
            print(mdct_a1[saveIndx], mdcst_a1[saveIndx])

            np.save('WDOExperiment/spc_mdct_a1.npy', mdct_a1)
            np.save('WDOExperiment/spc_mdcst_a1.npy', mdcst_a1)


        elif selection == 'pqmf':
            print('PQMF')
            # Alpha = 1.
            pqmf_cos_a1[saveIndx] = spc_eval_pqmf_cos(1., mixKaraoke, vox)
            print(pqmf_cos_a1[saveIndx])
            np.save('WDOExperiment/spc_pqmf_cos_a1.npy', pqmf_cos_a1)


        elif selection == 'pqmf_complex':
            print('PQMF Complex')
            # Alpha = 1.
            pqmf_compl_a1[saveIndx] = spc_eval_pqmf_complex(1., mixKaraoke, vox)
            print(pqmf_compl_a1[saveIndx])
            np.save('WDOExperiment/spc_pqmf_compl_a1.npy', pqmf_compl_a1)

        elif selection == 'mp':
            print('Matching Pursuit')
            mdct_union_mp_ovdt[saveIndx] = spc_eval_dct_union_mp_ovdt(mixKaraoke, vox)
            dctdst_union_mp_ovdt[saveIndx] = spc_eval_dctdst_mp(mixKaraoke, vox)
            print(mdct_union_mp_ovdt[saveIndx], dctdst_union_mp_ovdt[saveIndx])
            np.save('WDOExperiment/spc_mdctdst_union_mp_ovdt.npy', dctdst_union_mp_ovdt)
            np.save('WDOExperiment/spc_mdct_union_mp_ovdt.npy', mdct_union_mp_ovdt)

        else :
            assert('Unknown Selection!')

        # Update Storing Index
        saveIndx += 1

    return None

if __name__ == "__main__":

    # Define Operation
    # Plot acquired results
    #operation = 'Results'
    # Run the disjointness experiment
    #operation = 'WDO'
    # Run the sparsity experiment
    operation = 'Sparsity'

    print(operation)
    # Select multi processing feature
    multi_processing = False

    if operation == 'WDO' :
        if multi_processing == True :
            p1 = Process(target = mainWDO, args = ('stft',))
            p1.start()
            p2 = Process(target = mainWDO, args = ('mdct',))
            p2.start()
            p3 = Process(target = mainWDO, args = ('pqmf',))
            p3.start()
            p4 = Process(target = mainWDO, args = ('pqmf_complex',))
            p4.start()
            p5 = Process(target = mainWDO, args = ('mdcst_oc',))
            p5.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
        else :
            mainWDO('stft')
            mainWDO('mdct')
            mainWDO('pqmf')
            mainWDO('pqmf_complex')
            mainWDO('mp')

    if operation == 'Sparsity' :
        if multi_processing == True :
            p1 = Process(target = mainSPC, args = ('stft',))
            p1.start()
            p2 = Process(target = mainSPC, args = ('mdct',))
            p2.start()
            p3 = Process(target = mainSPC, args = ('pqmf',))
            p3.start()
            p4 = Process(target = mainSPC, args = ('pqmf_complex',))
            p4.start()
            p5 = Process(target = mainSPC, args = ('mdcst_oc',))
            p5.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()

        else :
            mainSPC('stft')
            mainSPC('mdct')
            mainSPC('pqmf')
            mainSPC('pqmf_complex')
            #mainSPC('mp')                  #TODO : Investigation for better dictionaries

    if operation == 'Results':
        print('Loading WDO Results')

        stft_hann_a1,stft_bt_a1, stft_nt_a1, stft_ntB_a1, mdct_a1, mdcst_a1, pqmf_cos_a1,\
        pqmf_compl_a1, mdct_union_mp_ovdt, mdctdst_union_mp_ovdt = load_results()

        data = np.vstack((stft_hann_a1, stft_bt_a1, stft_nt_a1, stft_ntB_a1, mdct_a1, mdcst_a1,
                          pqmf_cos_a1, pqmf_compl_a1, mdct_union_mp_ovdt)).T

        data = np.delete(data, (85,87), 0)

        colors = ['cyan', 'cyan', 'cyan', 'cyan', 'lightgreen', 'lightgreen', 'pink',
                  'pink', 'gray', 'gray', 'pink']

        labels = ['STFT-H 50%', 'STFT-Bt 50%', 'STFT-Nt 75%', 'STFT-Nt 50%', 'MDCT', 'MDCST',
                  'PQMF-cos', 'PQMF-complex', 'union']

        meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')

        # Configure Boxplot
        plt.figure(1)
        rcParams['xtick.labelsize'] = 19
        rcParams.update({'font.size': 22})
        box = plt.boxplot(data, notch = True, patch_artist = True, labels = labels, meanprops=meanpointprops, meanline=False,
                   showmeans=True)

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        plt.ylabel('W-DO Measure', fontsize=22)
        plt.show(block = False)

        print('Loading Sparsity Results')
        spc_stft_hann_mix, spc_stft_bt_mix, spc_stft_nt_mix, spc_stft_ntB_mix, spc_mdct_mix,\
        spc_mdcst_mix, spc_pqmf_mix, spc_pqmf_comp_mix = load_SPCresults()


        dataMIX = np.vstack((spc_stft_hann_mix, spc_stft_bt_mix, spc_stft_nt_mix, spc_stft_ntB_mix,
                             spc_mdct_mix, spc_mdcst_mix, spc_pqmf_mix, spc_pqmf_comp_mix)).T

        dataMIX = np.delete(dataMIX, (85,87), 0)

        colors = ['cyan', 'cyan', 'lightblue', 'lightblue', 'lightgreen', 'lightgreen',
                  'pink', 'pink', 'gray', 'gray', 'gray']

        labels = ['STFT-H 50%', 'STFT-Bt 50%', 'STFT-Nt 75%', 'STFT-Nt 50%', 'MDCT', 'MDCST',
                  'PQMF-cos', 'PQMF-complex']

        meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')

        # Configure Boxplot
        plt.figure(2)
        box2 = plt.boxplot(dataMIX, notch = True, patch_artist = True, labels = labels, meanprops=meanpointprops, meanline=False,
                   showmeans=True)
        for patch, color in zip(box2['boxes'], colors):
            patch.set_facecolor(color)
        rcParams['xtick.labelsize'] = 19
        rcParams.update({'font.size': 22})
        plt.ylabel('Sparsity Measure', fontsize=22)
        plt.show(block = False)