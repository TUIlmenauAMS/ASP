# -*- coding: utf-8 -*-
"""
        Class to handle the implementations of a real time Pseudo Quandrature Mirror Filter Bank.
        "Real time" means: it reads a block of N samples in and generates a block of N spectral samples from the QMF.
        The QMF implementaion uses internal memory to accomodate the window overlap. To use it out of the box, see
        "qmf_comp_call.py" file.
"""
__author__ = 'G. Schuller, S.I. Mimilakis'
__copyright__ = "Fraunhofer IDMT, TU Ilmenau, MacSeNet"

import numpy as np

try :
    import numpy.core._dotblas
    print('Fast BLAS found!')
except ImportError:
    print('Slow BLAS will be used...')

import scipy.fftpack as spfft
from scipy import io, sparse, signal
import cPickle as pickle
import IOMethods as IO


# Utilities for Analysis
def ha2Pa3d(ha, N = 1024):
        """
            Method to produce the analysis polyphase Matrix "Pa"

            Arguments :
                ha    :  (1D Array) Windowing function (Cosine Modulation)
                N     :  (int) Number of subbands

            Returns :
                Pa  :  (3D Array) Analysis polyphase matrix


            Usage : Pa=ha2Pa3d(ha, N)
            Author : Gerald Schuller ('shl'), Dec/2/15
        """

        # Initialize
        L=len(ha)
        blocks=int(np.ceil(L/N))
        Pa=np.zeros((N,N,blocks), dtype = np.float32)

        # Construction loops
        for k in range(N): # Over subbands
          for m in range(blocks):  # Over block numbers
            for nphase in range(N): # Over Phase
              n=m*N+nphase

              # Indexing like impulse response, phase index is reversed (N-np):
              Pa[N-1-nphase,k,m]=ha[n]*np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(blocks*N-1-n-N/2.0+0.5))

	    return Pa

def ha2Fa3d_fast(qmfwin,N):
        """
            Method to produce the analysis polyphase Matrix "Pa".
            An optimized version

            Arguments  :
                qmfwin :  (1D Array) Windowing function (Cosine Modulation)
                N      :  (int) Number of subbands

            Returns  :
                Fa   :  (3D Array) Analysis polyphase matrix


            Usage : Pa = ha2Fa3d_fast(qmfwin, N)
            Author : Gerald Schuller ('shl'), Jan/23/16
        """

        # Initialization
        Fa=np.zeros((N,N,overlap), dtype = np.float32)

        # Construction loop
        for m in range(overlap/2):
           Fa[:,:,2*m]+=np.fliplr(np.diag(np.flipud(-qmfwin[m*2*N:(m*2*N+N/2)]*((-1)**m)),k=-N/2))
           Fa[:,:,2*m]+=(np.diag(np.flipud(qmfwin[m*2*N+N/2:(m*2*N+N)]*((-1)**m)),k=N/2))
           Fa[:,:,2*m+1]+=(np.diag(np.flipud(qmfwin[m*2*N+N : np.int(m*2*N+1.5*N)]*((-1)**m)),k=-N/2))
           Fa[:,:,2*m+1]+=np.fliplr(np.diag(np.flipud(qmfwin[np.int(m*2*N+1.5*N) :(m*2*N+2*N)]*((-1)**m)),k=N/2))

        return Fa

def ha2Fa3d_sinmod_fast(qmfwin, N):
        """
            Method to produce the analysis polyphase Matrix "Pa".
            An optimized version

            Arguments  :
                qmfwin :  (1D Array) Windowing function (Cosine Modulation)
                N      :  (int) Number of subbands

            Returns  :
                Fa   :  (3D Array) Analysis polyphase matrix


            Usage : Pa = ha2Fa3d_sinmod_fast(qmfwin, N)
            Author : Gerald Schuller ('shl'), Jan/29/16
        """

        # Initialization
        Fa=np.zeros((N,N,overlap), dtype = np.float32)

        # Construction loop
        for m in range(overlap/2):
           Fa[:,:,2*m]+=np.fliplr(np.diag(np.flipud(qmfwin[m*2*N:(m*2*N+N/2)]*((-1)**m)),k=-N/2))
           Fa[:,:,2*m]+=(np.diag(np.flipud(qmfwin[m*2*N+N/2:(m*2*N+N)]*((-1)**m)),k=N/2))
           Fa[:,:,2*m+1]+=(np.diag(np.flipud(qmfwin[m*2*N+N:np.int(m*2*N+1.5*N)]*((-1)**m)),k=-N/2))
           Fa[:,:,2*m+1]+=np.fliplr(np.diag(np.flipud(-qmfwin[np.int(m*2*N+1.5*N):(m*2*N+2*N)]*((-1)**m)),k=N/2))

        return Fa

    # Utilities for Synthesis

# Utilities for Synthesis
def hs2Ps3d(hs,N):
        """
            Method to produce the synthesis polyphase Matrix "Pa"

            Arguments :
                hs    :  (1D Array) Windowing function (Cosine Modulation for synthesis)
                N     :  (int) Number of subbands

            Returns :
                Pa  :  (3D Array) Analysis polyphase matrix


            Usage : Pa=hs2Ps3d(ha, N)
            Author : Gerald Schuller ('shl'), Dec/2/15
        """

        # Initialize
        L=len(hs);
        blocks=int(np.ceil(L/N))
        Ps=np.zeros((N,N,blocks), dtype = np.float32)

        # Construction loops
        for k in range(N): # Over subbands
          for m in range(blocks):  # Over block numbers
            for nphase in range(N): # Over Phase
              n=m*N+nphase

              # Synthesis
              Ps[k,nphase,m]=hs[n]*np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(n-N/2.0+0.5))

        return Ps

def hs2Fs3d_fast(qmfwin, N):
    """
        Method to produce the synthesis polyphase Matrix "Pa".
        An optimized version

        Arguments  :
            qmfwin :  (1D Array) Windowing function (Cosine Modulation)
            N      :  (int) Number of subbands

        Returns  :
            Fs   :  (3D Array) Synthesis polyphase matrix


        Usage : Fs = hs2Fs3d_fast(qmfwin, N)
        Author : Gerald Schuller ('shl'), Jan/23/15

    """

    # Initialize
    global overlap
    Fs = np.zeros((N,N,overlap), dtype = np.float32)

    # Construction loop
    for m in range(overlap/2):
        Fs[:,:,2*m]+=np.fliplr(np.diag(np.flipud(qmfwin[m*2*N:(m*2*N+N/2)]*((-1)**m)),k=N/2))
        Fs[:,:,2*m]+=(np.diag((qmfwin[m*2*N+N/2:np.int(m*2*N+N)]*((-1)**m)),k=N/2))
        Fs[:,:,2*m+1]+=(np.diag((qmfwin[m*2*N+N:np.int(m*2*N+1.5*N)]*((-1)**m)),k=-N/2))
        Fs[:,:,2*m+1]+=np.fliplr(np.diag(np.flipud(-qmfwin[np.int(m*2*N+1.5*N):(m*2*N+2*N)]*((-1)**m)),k=-N/2))


    # Avoid sign change after reconstruction
    return -Fs

def hs2Fs3d_sinmod_fast(qmfwin,N):
    """
        Method to produce the synthesis polyphase Matrix "Pa".
        An optimized version

        Arguments  :
            qmfwin :  (1D Array) Windowing function (Cosine Modulation)
            N      :  (int) Number of subbands

        Returns    :
            Fs     :  (3D Array) Synthesis polyphase matrix

           Usage : Fs = hs2Fs3d_sinmod_fast(qmfwin, N)
           Author : Gerald Schuller ('shl'), Jan/29/16
    """

    # Initialize
    global overlap
    Fs=np.zeros((N,N,overlap), dtype = np.float32)

    # Construction loop
    for m in range(overlap/2):
        Fs[:,:,2*m]+=np.fliplr(np.diag(np.flipud(-qmfwin[m*2*N:(m*2*N+N/2)]*((-1)**m)),k=N/2))
        Fs[:,:,2*m]+=(np.diag((qmfwin[m*2*N+N/2:(m*2*N+N)]*((-1)**m)),k=N/2))
        Fs[:,:,2*m+1]+=(np.diag((qmfwin[m*2*N+N : np.int(m*2*N+1.5*N)]*((-1)**m)),k=-N/2))
        Fs[:,:,2*m+1]+=np.fliplr(np.diag(np.flipud(qmfwin[ np.int(m*2*N+1.5*N) : (m*2*N+2*N)]*((-1)**m)),k=-N/2))

    # Avoid sign change after reconstruction
    return -Fs

    # Other Utilities

# Matrix Utilities
def DCToMatrix(N):
    """
        Method to create an odd DCT Matrix.

        Arguments  :
            N      :  (int) Number of subbands

        Returns    :
            y      :  (3D Array) Odd DCT Mmtrix

           Author  : Gerald Schuller ('shl'), Dec. 2015
    """

    # Initialize
    y=np.zeros((N,N,1), dtype = np.float32);

    # Construction loops
    for n in range(N):
        for k in range(N):
            y[n,k,0]=np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(n+0.5))
            #y(n,k)=cos(pi/N*(k-0.5)*(n-1))

    return y

def DSToMatrix(N):
    """
        Method to create an odd DST Matrix.

        Arguments  :
               N   :  (int) Number of subbands

        Returns    :
               y   :  (3D Array) Odd DST Mmtrix

        Author : Gerald Schuller ('shl'), Jan. 2016
    """

    # Initialize
    y=np.zeros((N,N,1), dtype = np.float32);

    # Construction Loops
    for n in range(N):
        for k in range(N):
            y[n,k,0]=np.sqrt(2.0/N)*np.sin(np.pi/N*(k+0.5)*(n+0.5));
            #y(n,k)=cos(pi/N*(k-0.5)*(n-1));

    return y

def polmatmult(A, B):
    """
        Method for multiplying 2 polynomial matrices A and B,
        where each matrix entry is a polynomial, e.g. in z^-1.
        Those polynomial entries are in the 3rd dimension.
        The third dimension can also be interpreted as containing
        the (2D) coefficient matrices for each exponent of z^-1.
        Result is C=A*B;

            Arguments  :
                A      :  (ND Array) Matrix
                B      :  (ND Array) Matrix

            Returns    :
                C      :  (3D Array) Matrix

            Author : Gerald Schuller ('shl'), Dec. 2015
    """

    # Initialize
    [NAx,NAy,NAz]=A.shape
    [NBx,NBy,NBz]=B.shape
    #Degree +1 of resulting polynomial, with NAz-1 and NBz-1 beeing the degree of the input  polynomials:
    Deg=NAz+NBz-1
    C=np.zeros((NAx,NBy,Deg), dtype = np.float32)

    for n in range(Deg):
        for m in range(n+1):
            if ((n-m)<NAz and m<NBz):
                C[:,:,n]=C[:,:,n]+ A[:,:,(n-m)].dot(B[:,:,m])
                #sparse version:
                #C[:,:,n]=C[:,:,n]+ (sparse.csr_matrix(A[:,:,(n-m)]).dot(sparse.csr_matrix(B[:,:,m]))).todense()
    return C

def reset_rt():
    """
        Method to reset the block buffers
    """

    global blockmemorysyn
    global blockmemory
    global overlap

    blockmemory = np.zeros((overlap,N))
    blockmemorysyn = np.zeros((overlap,N))
    blockmemory_sin = np.zeros((overlap, N))
    blockmemorysyn_sin = np.zeros((overlap, N))

    print('Block Memories Resetted')

    return None

#The DCT4 transform:
def DCT4(samples):
    """
        Method to create DCT4 transformation using DCT3

        Arguments   :

            samples : (1D Array) Input samples to be transformed

        Returns     :

            y       :  (1D Array) Transformed output samples

    """

    # Initialize
    samplesup=np.zeros(2*N, dtype = np.float32)
    # Upsample signal:
    samplesup[1::2]=samples

    y=spfft.dct(samplesup,type=3,norm='ortho')*np.sqrt(2)#/2

    return y[0:N]

#The DST4 transform:
def DST4(samples):
    """
        Method to create DST4 transformation using DST3

        Arguments   :
            samples : (1D Array) Input samples to be transformed

        Returns     :
            y       :  (1D Array) Transformed output samples

    """

    # Initialize
    samplesup=np.zeros(2*N, dtype = np.float32)

    # Upsample signal
    # Reverse order to obtain DST4 out of DCT4:
    #samplesup[1::2]=np.flipud(samples)
    samplesup[0::2] = samples
    y = spfft.dst(samplesup,type=3,norm='ortho')*np.sqrt(2)#/2

    # Flip sign of every 2nd subband to obtain DST4 out of DCT4
    #y=(y[0:N])*(((-1)*np.ones(N, dtype = np.float32))**range(N))

    return y[0: N]

def x2polyphase(x, N):
    """
        Method to convert input signal x (a row vector) into a polphase row vector
	    of blocks with length N.

        Arguments   :
            x       : (1D Array) Input samples
            N       : (int)  Number of subbands

        Returns     :
            y       :  (3D Array) Polyphase representation of the input signal

        Author : Gerald Schuller ('shl'), Dec/2/15
    """

    #Number of blocks in the signal:
    L=int(np.floor(len(x)/N))

    xp=np.zeros((1,N,L), dtype = np.float32)

    for m in range(L):
        xp[0,:,m] = x[m*N: (m*N+N)]

    return xp

def polyphase2x(xp):
    """
        Method to convert a polyphase input signal xp (a row vector) into a contiguous row vector.

        Arguments   :
            xp      : (3D Array) Input Polyphase representation

        Returns     :
            x       : (1D Array) Output row vector signal

        Author : Gerald Schuller ('shl'), Aug/24/11
    """

    #Number of blocks in the signal:
    [r,N,b] = xp.shape
    L = b

    x = np.zeros(N*L, dtype = np.float32);

    for m in range(L):
        #print x[(m*N):((m+1)*N)].shape, xp[0,:,m].shape
        x[(m*N):(m+1)*N] = xp[0,:,m]

    return x

# Parameters
# Number of subbands of the QMF filter bank
N = 1024

# QMF Window
qmfwin = np.loadtxt('QMF/qmf1024qn.mat').astype(np.float32)
qmfwin = np.hstack((qmfwin, qmfwin[::-1]))

# Overlapped frames
overlap = 8

# Block frames for analysis and synthesis (Cosine)
blockmemory = np.zeros((overlap, N), dtype = np.float32)
blockmemorysyn = np.zeros((overlap, N), dtype = np.float32)

# Block frames for analysis and synthesis (Sine)
blockmemory_sin = np.zeros((overlap, N), dtype = np.float32)
blockmemorysyn_sin = np.zeros((overlap, N), dtype = np.float32)

# Analysis/Synthesis Matrices
FsCos = hs2Fs3d_fast(qmfwin, N)
FsSin = hs2Fs3d_sinmod_fast(qmfwin, N)
FaCos = ha2Fa3d_fast(qmfwin, N)
FaSin = ha2Fa3d_sinmod_fast(qmfwin, N)

class PQMFAnalysis():
    """
        Implements a real time Pseudo Quadrature Mirror Filter Bank (Analysis part).
        "Real time" means: it reads a block of N samples in and generates a block of N spectral samples from the QMF.
        The QMF implementaion uses internal memory to accomodate the window overlap.
        Gerald Schuller, gerald.schuller@tu-ilmenau.de, January 2016.
    """
    @staticmethod
    def analysisqmf_realtime(xrt, N = 1024):
        """
            Method to compute the QMF subband samples for each real time input block xrt.
            Conducts an implicit polynomial multiplication with folding matrix Fa of
            the polyphase matrix of the QMF filter bank, using
            internal memory of the past input blocks.

            Arguments   :
                xrt     : (1D Array) Input signal (block-based)
                N       : (int)      Number of sub-bands

            Returns     :
                y       : (2D Array) Output of QMF analysis matrix (Cosine)

            Author : Gerald Schuller ('shl')
        """

        global blockmemory
        global overlap

        # Push down old blocks:
        blockmemory[0:(overlap-1),:]=blockmemory[1:(overlap),:]

        # Write new block into top of memory stack:
        blockmemory[overlap-1,:]=xrt;
        y=np.zeros((1,N), dtype = np.float32);

        for m in range(overlap):
           y += np.dot(np.array([blockmemory[overlap-1-m,:]]), FaCos[:,:,m])
           #y+= (sparse.csr_matrix(blockmemory[overlap-1-m,:]).dot(sparse.csr_matrix(Fa[:,:,m]))).todense()

        # Fast DCT4:
        y = DCT4(y)
        return y

    @staticmethod
    def analysisqmf_sinmod_realtime(xrt, N = 1024):
        """
            Method to compute the QMF subband samples for each real time input block xrt.
            Conducts an implicit polynomial multiplication with folding matrix Fa of
            the polyphase matrix of the QMF filter bank, using
            internal memory of the past input blocks.

            Arguments   :
                xrt     : (1D Array) Input signal (block-based)
                N       : (int)      Number of sub-bands

            Returns     :
                y       : (2D Array) Output of QMF analysis matrix (Sine)

            Author : Gerald Schuller ('shl')
        """

        global blockmemory_sin
        global overlap

        # Push down old blocks:
        blockmemory_sin[0:(overlap-1),:] = blockmemory_sin[1:(overlap),:]

        # Write new block into top of memory stack

        blockmemory_sin[overlap-1,:] = xrt
        y=np.zeros((1, N), dtype = np.float32)

        for m in range(overlap):
           y += np.dot(np.array([blockmemory_sin[overlap-1-m,:]]), FaSin[:,:,m])
           #y+= (sparse.csr_matrix(blockmemory_sin[overlap-1-m,:]).dot(sparse.csr_matrix(Fa[:,:,m]))).todense()

        #Fast DST4
        y = DST4(y)

        return y

    @staticmethod
    def complex_analysis_realtime(xrt, N = 1024):
        """
            Method to compute the QMF subband samples for each real time input block xrt.
            Conducts an implicit polynomial multiplication with folding matrix Fa of
            the polyphase matrix of the QMF filter bank, using
            internal memory of the past input blocks. A complex output matrix will be
            computed using DCT and DST.

            Arguments   :
                xrt     : (1D Array) Input signal (block-based)
                N       : (int)      Number of sub-bands

            Returns     :
                y       : (2D Array) Complex output of QMF analysis matrix (Cosine)

            Authors     : Gerald Schuller('shl'), S.I. Mimilakis ('mis')
        """

        global blockmemory_sin, blockmemory
        global overlap

        # Push down old blocks:
        blockmemory[0:(overlap-1),:] = blockmemory[1:(overlap),:]
        blockmemory_sin[0:(overlap-1),:] = blockmemory_sin[1:(overlap),:]

        # Write new block into top of memory stack
        blockmemory_sin[overlap-1,:] = xrt
        blockmemory[overlap-1,:] = xrt

        # Initialize storing vectors
        y = np.empty((1, N), dtype = complex)
        ycos = np.zeros((1, N), dtype = np.float32)
        ysin = np.zeros((1, N), dtype = np.float32)

        for m in range(overlap):
            ycos += np.dot(np.array([blockmemory[overlap-1-m,:]]), FaCos[:,:,m])
            ysin += np.dot(np.array([blockmemory_sin[overlap-1-m,:]]), FaSin[:,:,m])

        # Fast DCT4
        ycos = DCT4(ycos)
        # Fast DST4
        ysin = DST4(ysin)

        y = ycos + (1j*ysin)

        return y

    @staticmethod
    def compute_pseudo_magnitude(ms):
        """
            Method to compute the regularized magnitude spectrum for real-valued
            (cosine modulated) signal representations.

            References :
                [1] L. Daudet and M. Sandler, "MDCT analysis of sinusoids: exact
                results and applications to coding artifacts reduction,"
                in IEEE Transactions on Speech and Audio Processing, vol. 12,
                no. 3, pp. 302-312, May 2004.


            Arguments   :
                ms      : (2D Array)  Input Time-Frequency representation (timeframes, sub-bands).

            Returns     :
                mX      : (2D Array)  Regularized Magnitude Coefficients (timeframes, sub-bands).

            Authors     : S.I. Mimilakis ('mis')
        """
        mX = np.zeros((ms.shape[0], ms.shape[1]), dtype=np.float32)

        for indx in xrange(ms.shape[0]):
            cfr = np.zeros((ms.shape[1],), dtype=np.float32)
            for csamples in xrange(1, len(cfr) - 1):
                cfr[csamples] = ms[indx, csamples] ** 2. + (ms[indx, csamples - 1] - ms[indx, csamples + 1]) ** 2.
            mX[indx, :] = cfr[:]

        return mX

    @staticmethod
    def analyseNStore(filePath, N = 1024, saveStr = 'Analysis_QMF.p'):
        """
            PQMF analysis and storing into pickle.

            Arguments   :
                filePath: (string)   Path for the input wave file
                N       : (int)      Number of sub-bands
                saveStr : (string)   Output filename

            Authors     : Gerald Schuller('shl'), S.I. Mimilakis ('mis')
        """

        x, fs = IO.AudioIO.wavRead(filePath, mono = False)

        if x.shape[1] == 1 :
            timeSlots = len(x)/N

            ms = np.empty((timeSlots, N), dtype = complex)
            for indx in xrange(timeSlots):
                ms[indx, :] = PQMFAnalysis.complex_analysis_realtime(x[indx * N : (indx + 1) * N, 0], N)

        elif x.shape[1] == 2 :
            timeSlots = len(x[:, 0])/N
            ms = np.empty((2, timeSlots, N), dtype = complex)

            for channelIndx in xrange(2) :
                for indx in xrange(timeSlots):
                    ms[channelIndx, indx, :] = PQMFAnalysis.complex_analysis_realtime(x[indx * N : (indx + 1) * N, channelIndx], N)


        if saveStr[-1] == 'p' :
            pickle.dump(ms, open(saveStr, 'wb'))
        else :
            io.savemat(saveStr, dict(analysis = ms))

        reset_rt()
        return None

class PQMFSynthesis() :

    """
        Implements a real time Pseudo Quandrature Mirror Filter Bank (Synthesis part).
        "Real time" means: it reads a block of N samples in and generates a block of N spectral samples from the QMF.
        The QMF implementaion uses internal memory to accomodate the window overlap.
        Gerald Schuller, gerald.schuller@tu-ilmenau.de, January 2016
    """

    @staticmethod
    def synthesisqmf_realtime(y, N = 1024):
        """
            Method to compute the inverse QMF for each subband block y,
            conducts an implicit polynomial multiplication with synthesis folding matrix Fs
            of the synthesis polyphase matrix of the QMF filter bank, using
            internal memory for future output blocks.

            Arguments   :
                y       : (2D Array) Analysed QMF Matrix
                N       : (int)      Number of sub-bands

            Returns     :
                xrek    : (1D Array) Reconstructed signal

            Author : Gerald Schuller ('shl')
        """
        global blockmemorysyn
        global overlap

        # Push down old blocks:
        blockmemorysyn[0:(overlap-1),:]=blockmemorysyn[1:(overlap), :]

        # Avoid leaving previous values in top of memory
        blockmemorysyn[overlap-1,:]=np.zeros((1, N), dtype = np.float32)

        #print "memory content synthesis: ", np.sum(np.abs(blockmemorysyn))
        #print "Fs.shape =", Fs.shape
        #print "y.shape= ", y.shape

        #Overlap-add after fast (inverse) DCT4::
        for m in range(overlap):
           blockmemorysyn[m,:] += np.dot(DCT4(y), FsCos[:,:,m])
           #y+= (sparse.csr_matrix(blockmemory[overlap-1-m,:]).dot(sparse.csr_matrix(Fa[:,:,m]))).todense()

        xrek = blockmemorysyn[0,:]

        return xrek

    @staticmethod
    def synthesisqmf_sinmod_realtime(y, N = 1024):
        """
            Method to compute the inverse QMF for each subband block y,
            conducts an implicit polynomial multiplication with synthesis folding matrix Fs
            of the synthesis polyphase matrix of the QMF filter bank, using
            internal memory for future output blocks.

            Arguments   :
                y       : (2D Array) Analysed QMF Matrix
                N       : (int)      Number of sub-bands

            Returns     :
                xrek    : (1D Array) Reconstructed signal

            Author : Gerald Schuller ('shl')
        """

        global blockmemorysyn_sin
        global overlap

        #print "overlap= ", overlap

        # Push down old blocks:
        blockmemorysyn_sin[0:(overlap-1),:]=blockmemorysyn_sin[1:(overlap),:]
        blockmemorysyn_sin[overlap-1,:]=np.zeros((1,N)) #avoid leaving previous values in top of memory.

        #print "memory content synthesis: ", np.sum(np.abs(blockmemorysyn))
        #print "Fs.shape =", Fs.shape
        #print "y.shape= ", y.shape

        # Overlap-add after fast (inverse) DCT4::
        for m in range(overlap):
           blockmemorysyn_sin[m,:] += np.dot(DST4(y), FsSin[:,:,m])
           #y+= (sparse.csr_matrix(blockmemorysyn_sin[overlap-1-m,:]).dot(sparse.csr_matrix(Fa[:,:,m]))).todense()
        xrek = blockmemorysyn_sin[0,:]

        return xrek

    @staticmethod
    def complex_synthesis_realtime(ycomp, N = 1024):
        """
            Method to compute the inverse QMF for each subband block y,
            conducts an implicit polynomial multiplication with synthesis folding matrix Fs
            of the synthesis polyphase matrix of the QMF filter bank, using
            internal memory for future output blocks. A complex input matrix will be
            expected.

             Arguments   :
                 xrt     : (1D Array) Input signal (block-based)
                 N       : (int)      Number of sub-bands
            Returns     :
                 xrek   : (1D Array) Reconstructed signal

            Authors     : Gerald Schuller('shl'), S.I. Mimilakis ('mis')
        """

        global blockmemorysyn_sin, blockmemorysyn
        global overlap

        # Push down old blocks:
        blockmemorysyn[0:(overlap-1),:] = blockmemorysyn[1:(overlap), :]
        blockmemorysyn_sin[0:(overlap-1),:] = blockmemorysyn_sin[1:(overlap),:]
        # Avoid leaving previous values in top of memory.
        blockmemorysyn_sin[overlap-1,:] = np.zeros((1,N), dtype = np.float32)
        blockmemorysyn[overlap-1,:] = np.zeros((1, N), dtype = np.float32)

        # Overlap-add after fast (inverse) DCT4:
        for m in range(overlap):
            blockmemorysyn[m,:] += np.dot(DCT4(np.real(ycomp)), FsCos[:,:,m])
            blockmemorysyn_sin[m,:] += np.dot(DST4(np.imag(ycomp)), FsSin[:,:,m])

        xrek = 0.5 * (blockmemorysyn[0,:] + blockmemorysyn_sin[0, :])
        #xrek = blockmemorysyn_sin[0, :]
        #xrek = blockmemorysyn[0,:]

        return xrek

    @staticmethod
    def loadNRes(filePath, fs = 44100, saveStr = 'SynthesisQMF.wav'):
        """
            PQMF resynthesis from stored into pickle.

            Arguments   :
                filePath: (string)   Path for the input analysis pickle file
                fs      : (int)      Sampling Frequency
                saveStr : (string)   Output filename (wave)

            Authors     : Gerald Schuller('shl'), S.I. Mimilakis ('mis')
        """

        ms = pickle.load(open(filePath, 'rb'))

        if len(ms.shape) == 2:
            timeSlots = ms.shape[0]
            N = ms.shape[1]

            yrec = np.empty((timeSlots*N, 1))
            for indx in xrange(timeSlots):
                yrec[indx * N : (indx + 1) * N, 0] = PQMFSynthesis.complex_synthesis_realtime(ms[indx, :], N)

            IO.AudioIO.wavWrite(yrec, fs, 16, saveStr)
            reset_rt()

        elif len(ms.shape) == 3:
            timeSlots = ms.shape[1]
            N = ms.shape[1]
            channels = ms.shape[0]

            yrec = np.empty((timeSlots*N, channels))
            for channelIndx in xrange(channels):
                for indx in xrange(timeSlots):
                    yrec[indx * N : (indx + 1) * N, channelIndx] = PQMFSynthesis.complex_synthesis_realtime(ms[channelIndx, indx, :], N)
                reset_rt()

            IO.AudioIO.wavWrite(yrec, fs, 16, saveStr)

        return None

if __name__ == '__main__':

    # Examples using the class
    import qmf_realtime_class as qmf
    import IOMethods as IO
    import numpy as np
    import cPickle  as pickle
    import os
    import matplotlib.pyplot as plt

    x, fs = IO.AudioIO.wavRead('mixed.wav', mono = True)

    N = 1024
    timeSlots = len(x)/1024

    # Initialize analysis matrix
    ms = np.empty((timeSlots, N), dtype = complex)

    # Complex analysis, with internal storing
    for indx in xrange(timeSlots):
        ms[indx, :] = qmf.PQMFAnalysis.complex_analysis_realtime(x[indx * N : (indx + 1) * N, 0], N)

    # Magnitude
    plt.figure(1)
    plt.imshow(np.abs((ms[:500,:].T)), aspect='auto', interpolation='nearest', origin='lower', cmap='jet')
    plt.draw()
    plt.show()

    # Phase
    plt.figure(2)
    plt.imshow((np.angle(ms[:500,:].T)), aspect='auto', interpolation='nearest', origin='lower', cmap='jet')
    plt.draw()
    plt.show()

    # Complex synthesis, from internal variable
    yrec = np.empty((timeSlots*N, 1))
    for indx in xrange(timeSlots):
        yrec[indx * N : (indx + 1) * N, 0] = qmf.PQMFSynthesis.complex_synthesis_realtime(ms[indx, :], N)

    # Reset Internal buffers!!!
    qmf.reset_rt()

    filePathWave = 'mixed.wav'
    # Complex analysis, with external storing using pickle
    qmf.PQMFAnalysis.analyseNStore(filePathWave, N = 1024, saveStr = 'Analysis_QMF.p')

    filePathPickle = 'Analysis_QMF.p'
    # Load complex matrix from the analysis
    ycomp = pickle.load(open(filePathPickle, 'rb'))

    # Magnitude
    plt.figure(3)
    plt.imshow(np.abs((ycomp[:500,:].T)), aspect='auto', interpolation='nearest', origin='lower', cmap='jet')
    plt.draw()
    plt.show()

    # Phase
    plt.figure(4)
    plt.imshow(np.angle((ycomp[:500,:].T)), aspect='auto', interpolation='nearest', origin='lower', cmap='jet')
    plt.draw()
    plt.show()

    # Complex synthesis, from external file using pickle
    qmf.PQMFSynthesis.loadNRes(filePathPickle, fs = 44100, saveStr = 'SynthesisQMF.wav')

    os.remove('Analysis_QMF.p')
    os.remove('SynthesisQMF.wav')