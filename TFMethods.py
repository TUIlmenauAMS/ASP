# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import math
import numpy as np
from scipy.fftpack import fft, ifft, dct, dst
from scipy.signal import firwin2, freqz, cosine, hanning, fftconvolve
from scipy.interpolate import InterpolatedUnivariateSpline as uspline

eps = np.finfo(np.float32).tiny

class TimeFrequencyDecomposition:
    """ A Class that performs time-frequency decompositions by means of a
        Discrete Fourier Transform, using Fast Fourier Transform algorithm
        by SciPy, MDCT-type IV and Fractional Fast Fourier Transform.
    """
    @staticmethod
    def DFT(x, w, N):
        """ Discrete Fourier Transformation(Analysis) of a given real input signal
        via an FFT implementation from scipy. Single channel is being supported.
        Args:
            x       : (array) Real time domain input signal
            w       : (array) Desired windowing function
            N       : (int)   FFT size
        Returns:
            magX    : (2D ndarray) Magnitude Spectrum
            phsX    : (2D ndarray) Phase Spectrum
        """

        # Half spectrum size containing DC component
        hlfN = (N/2)+1

        # Half window size. Two parameters to perform zero-phase windowing technique
        hw1 = int(math.floor((w.size+1)/2))
        hw2 = int(math.floor(w.size/2))

        # Window the input signal
        winx = x*w

        # Initialize FFT buffer with zeros and perform zero-phase windowing
        fftbuffer = np.zeros(N)
        fftbuffer[:hw1] = winx[hw2:]
        fftbuffer[-hw2:] = winx[:hw2]

        # Compute DFT via scipy's FFT implementation
        X = fft(fftbuffer)

        # Acquire magnitude and phase spectrum
        magX = (np.abs(X[:hlfN]))
        phsX = (np.angle(X[:hlfN]))

        return magX, phsX

    @staticmethod
    def iDFT(magX, phsX, wsz):
        """ Discrete Fourier Transformation(Synthesis) of a given spectral analysis
        via an inverse FFT implementation from scipy.
        Args:
            magX    : (2D ndarray) Magnitude Spectrum
            phsX    : (2D ndarray) Phase Spectrum
            wsz     :  (int)   Synthesis window size
        Returns:
            y       : (array) Real time domain output signal
        """

        # Get FFT Size
        hlfN = magX.size;
        N = (hlfN-1)*2

        # Half of window size parameters
        hw1 = int(math.floor((wsz+1)/2))
        hw2 = int(math.floor(wsz/2))

        # Initialise synthesis buffer with zeros
        fftbuffer = np.zeros(N)
        # Initialise output spectrum with zeros
        Y = np.zeros(N, dtype = complex)
        # Initialise output array with zeros
        y = np.zeros(wsz)

        # Compute complex spectrum(both sides) in two steps
        Y[0:hlfN] = magX * np.exp(1j*phsX)
        Y[hlfN:] = magX[-2:0:-1] * np.exp(-1j*phsX[-2:0:-1])

        # Perform the iDFT
        fftbuffer = np.real(ifft(Y))

        # Roll-back the zero-phase windowing technique
        y[:hw2] = fftbuffer[-hw2:]
        y[hw2:] = fftbuffer[:hw1]

        return y

    @staticmethod
    def STFT(x, w, N, hop):
        """ Short Time Fourier Transform analysis of a given real input signal,
        via the above DFT method.
        Args:
            x   : 	(array)  Magnitude Spectrum
            w   :   (array)  Desired windowing function
            N   :   (int)    FFT size
            hop :   (int)    Hop size
        Returns:
            sMx :   (2D ndarray) Stacked arrays of magnitude spectra
            sPx :   (2D ndarray) Stacked arrays of phase spectra
        """

        # Analysis Parameters
        wsz = w.size

        # Add some zeros at the start and end of the signal to avoid window smearing
        x = np.append(np.zeros(3*hop),x)
        x = np.append(x, np.zeros(3*hop))

        # Initialize sound pointers
        pin = 0
        pend = x.size - wsz
        indx = 0

        # Normalise windowing function
        w = w / sum(w)

        # Initialize storing matrix
        xmX = np.zeros((len(x)/hop, N/2 + 1), dtype = np.float32)
        xpX = np.zeros((len(x)/hop, N/2 + 1), dtype = np.float32)

        # Analysis Loop
        while pin <= pend:
            # Acquire Segment
            xSeg = x[pin:pin+wsz]

            # Perform DFT on segment
            mcX, pcX = TimeFrequencyDecomposition.DFT(xSeg, w, N)

            xmX[indx, :] = mcX
            xpX[indx, :] = pcX

            # Update pointers and indices
            pin += hop
            indx += 1

        return xmX, xpX

    @staticmethod
    def iSTFT(xmX, xpX, wsz, hop) :
        """ Short Time Fourier Transform synthesis of given magnitude and phase spectra,
        via the above iDFT method.
        Args:
            xmX :   (2D ndarray)  Magnitude Spectrum
            xpX :   (2D ndarray)  Phase Spectrum
            wsz :   (int)    Synthesis Window size
            hop :   (int)    Hop size
        Returns :
            y   :   (array) Synthesised time-domain real signal.
        """

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
            ybuffer = TimeFrequencyDecomposition.iDFT(xmX[indx, :], xpX[indx, :], wsz)

            # Overlap and Add
            y[pin:pin+wsz] += ybuffer*hop

            # Advance pointer
            pin += hop

        # Delete the extra zeros that the analysis had placed
        y = np.delete(y, range(3*hop))
        y = np.delete(y, range(y.size-(3*hop + 1), y.size))

        return y

    @staticmethod
    def nuttall4b(M, sym=False):
        """
        Returns a minimum 4-term Blackman-Harris window according to Nuttall.
        The typical Blackman window famlity define via "alpha" is continuous
        with continuous derivative at the edge. This will cause some errors
        to short time analysis, using odd length windows.

        Args    :
            M   :   (int)   Number of points in the output window.
            sym :   (array) Synthesised time-domain real signal.

        Returns :
            w   :   (ndarray) The windowing function

        References :
            [1] Heinzel, G.; Rüdiger, A.; Schilling, R. (2002). Spectrum and spectral density
               estimation by the Discrete Fourier transform (DFT), including a comprehensive
               list of window functions and some new flat-top windows (Technical report).
               Max Planck Institute (MPI) für Gravitationsphysik / Laser Interferometry &
               Gravitational Wave Astronomy, 395068.0

            [2] Nuttall A.H. (1981). Some windows with very good sidelobe behaviour. IEEE
               Transactions on Acoustics, Speech and Signal Processing, Vol. ASSP-29(1):
               84-91.
        """

        if M < 1:
            return np.array([])
        if M == 1:
            return np.ones(1, 'd')
        if not sym :
            M = M + 1

        a = [0.355768, 0.487396, 0.144232, 0.012604]
        n = np.arange(0, M)
        fac = n * 2 * np.pi / (M - 1.0)

        w = (a[0] - a[1] * np.cos(fac) +
             a[2] * np.cos(2 * fac) - a[3] * np.cos(3 * fac))

        if not sym:
            w = w[:-1]

        return w

    @staticmethod
    def coreModulation(win, N):
        """
            Method to produce Analysis and Synthesis matrices for the offline
            complex PQMF class.

            Arguments  :
                win    :  (1D Array) Windowing function
                N      :  (int) Number of subbands

            Returns  :
                Cos   :   (2D Array) Cosine Modulated Polyphase Matrix
                Sin   :   (2D Array) Sine Modulated Polyphase Matrix


            Usage  : Cos, Sin = TimeFrequencyDecomposition.coreModulation(qmfwin, N)
        """

        lfb = len(win)
        # Initialize Storing Variables
        Cos = np.zeros((N,lfb), dtype = np.float32)
        Sin = np.zeros((N,lfb), dtype = np.float32)

        # Generate Matrices
        for k in xrange(0, lfb):
            for n in xrange(0, N):
                Cos[n, k] = win[k] * np.cos(np.pi/N * (n + 0.5) * (k + 0.5 + N/2)) * np.sqrt(2. / N)
                Sin[n, k] = win[k] * np.sin(np.pi/N * (n + 0.5) * (k + 0.5 + N/2)) * np.sqrt(2. / N)

        return Cos, Sin

    @staticmethod
    def complex_analysis(x, N = 1024):
        """
            Method to compute the subband samples from time domain signal x.
            A complex output matrix will be computed using DCT and DST.

            Arguments   :
                x       : (1D Array) Input signal
                N       : (int)      Number of sub-bands

            Returns     :
                y       : (2D Array) Complex output of QMF analysis matrix (Cosine)

            Usage       : y = TimeFrequencyDecomposition.complex_analysis(x, N)

        """
        # Parameters and windowing function design
        win = cosine(2*N, True)
        win /= np.sum(win)
        lfb = len(win)
        nTimeSlots = len(x)/N - 2

        # Initialization
        ycos = np.zeros((len(x)/N, N), dtype = np.float32)
        ysin = np.zeros((len(x)/N, N), dtype = np.float32)

        # Analysis Matrices
        Cos, Sin = TimeFrequencyDecomposition.coreModulation(win, N)

        # Perform Complex Analysis
        for m in xrange(0, nTimeSlots):
            ycos[m, :] = np.dot(x[m * N : m * N + lfb], Cos.T)
            ysin[m, :] = np.dot(x[m * N : m * N + lfb], Sin.T)

        y = ycos + 1j *  ysin

        return y

    @staticmethod
    def complex_synthesis(y, N = 1024):
        """
            Method to compute the resynthesis of the MDCST.
            A complex input matrix is asummed as input, derived from DCT and DST.

            Arguments   :
                y       : (2D Array) Complex Representation

            Returns     :
                xrec    : (1D Array) Time domain reconstruction

            Usage       : xrec = TimeFrequencyDecomposition.complex_synthesis(y, N)

        """
        # Parameters and windowing function design
        win = cosine(2*N, True)
        win *= np.sum(win)
        lfb = len(win)
        nTimeSlots = y.shape[0]
        SignalLength = nTimeSlots * N + 2 * N

        # Synthesis matrices
        Cos, Sin = TimeFrequencyDecomposition.coreModulation(win, N)

        # Initialization
        zcos = np.zeros((1, SignalLength), dtype = np.float32)
        zsin = np.zeros((1, SignalLength), dtype = np.float32)

        # Perform Complex Synthesis
        for m in xrange(0, nTimeSlots):
            zcos[0, m * N : m * N + lfb] += np.dot(np.real(y[m, :]).T, Cos)
            zsin[0, m * N : m * N + lfb] += np.dot(np.imag(y[m, :]).T, Sin)

        xrec = 0.5 * (zcos + zsin)

        return xrec.T

    @staticmethod
    def real_analysis(x, N = 1024):
        """
            Method to compute the subband samples from time domain signal x.
            A complex output matrix will be computed using DCT and DST.

            Arguments   :
                x       : (1D Array) Input signal
                N       : (int)      Number of sub-bands

            Returns     :
                y       : (2D Array) Complex output of QMF analysis matrix (Cosine)

            Usage       : y = TimeFrequencyDecomposition.complex_analysis(x, N)

        """
        # Parameters and windowing function design
        win = cosine(2*N, True)
        win /= np.sum(win)
        lfb = len(win)
        nTimeSlots = len(x)/N - 2

        # Initialization
        ycos = np.zeros((len(x)/N, N), dtype = np.float32)
        ysin = np.zeros((len(x)/N, N), dtype = np.float32)

        # Analysis Matrices
        Cos, _ = TimeFrequencyDecomposition.coreModulation(win, N)

        # Perform Complex Analysis
        for m in xrange(0, nTimeSlots):
            ycos[m, :] = np.dot(x[m * N : m * N + lfb], Cos.T)

        return ycos

    @staticmethod
    def real_synthesis(y, N = 1024):
        """
            Method to compute the resynthesis of the MDCT.
            A complex input matrix is asummed as input, derived from DCT typeIV.

            Arguments   :
                y       : (2D Array) Real Representation

            Returns     :
                xrec    : (1D Array) Time domain reconstruction

            Usage       : xrec = TimeFrequencyDecomposition.complex_synthesis(y, N)

        """
        # Parameters and windowing function design
        win = cosine(2*N, True)
        win *= np.sum(win)
        lfb = len(win)
        nTimeSlots = y.shape[0]
        SignalLength = nTimeSlots * N + 2 * N

        # Synthesis matrices
        Cos, _ = TimeFrequencyDecomposition.coreModulation(win, N)

        # Initialization
        zcos = np.zeros((1, SignalLength), dtype = np.float32)

        # Perform Complex Synthesis
        for m in xrange(0, nTimeSlots):
            zcos[0, m * N : m * N + lfb] += np.dot((y[m, :]).T, Cos)

        xrec = zcos
        return xrec.T

    @staticmethod
    def frft(f, a):
        """
        Fractional Fourier transform. As appears in :
        -https://nalag.cs.kuleuven.be/research/software/FRFT/
        -https://github.com/audiolabs/frft/
        Args:
            f       : (array) Input data
            a       : (float) Alpha factor
        Returns:
            ret    : (array) Complex valued analysed data

        """
        ret = np.zeros_like(f, dtype=np.complex)
        f = f.copy().astype(np.complex)
        N = len(f)
        shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
        sN = np.sqrt(N)
        a = np.remainder(a, 4.0)

        # Special cases
        if a == 0.0:
            return f
        if a == 2.0:
            return np.flipud(f)
        if a == 1.0:
            ret[shft] = np.fft.fft(f[shft]) / sN
            return ret
        if a == 3.0:
            ret[shft] = np.fft.ifft(f[shft]) * sN
            return ret

        # reduce to interval 0.5 < a < 1.5
        if a > 2.0:
            a = a - 2.0
            f = np.flipud(f)
        if a > 1.5:
            a = a - 1
            f[shft] = np.fft.fft(f[shft]) / sN
        if a < 0.5:
            a = a + 1
            f[shft] = np.fft.ifft(f[shft]) * sN

        # the general case for 0.5 < a < 1.5
        alpha = a * np.pi / 2
        tana2 = np.tan(alpha / 2)
        sina = np.sin(alpha)
        f = np.hstack((np.zeros(N - 1), TimeFrequencyDecomposition.sincinterp(f), np.zeros(N - 1))).T

        # chirp premultiplication
        chrp = np.exp(-1j * np.pi / N * tana2 / 4 * np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
        f = chrp * f

        # chirp convolution
        c = np.pi / N / sina / 4
        ret = fftconvolve(np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2), f)
        ret = ret[4 * N - 4:8 * N - 7] * np.sqrt(c / np.pi)

        # chirp post multiplication
        ret = chrp * ret

        # normalizing constant
        ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]

        return ret

    @staticmethod
    def ifrft(f, a):
        """
        Inverse fractional Fourier transform. As appears in :
        -https://nalag.cs.kuleuven.be/research/software/FRFT/
        -https://github.com/audiolabs/frft/
        ----------
        Args:
            f       : (array) Complex valued input array
            a       : (float) Alpha factor
        Returns:
            ret     : (array) Real valued synthesised data
        """
        return TimeFrequencyDecomposition.frft(f, -a)

    @staticmethod
    def sincinterp(x):
        """
        Sinc interpolation for computation of fractional transformations.
        As appears in :
        -https://github.com/audiolabs/frft/
        ----------
        Args:
            f       : (array) Complex valued input array
            a       : (float) Alpha factor
        Returns:
            ret     : (array) Real valued synthesised data
        """
        N = len(x)
        y = np.zeros(2 * N - 1, dtype=x.dtype)
        y[:2 * N:2] = x
        xint = fftconvolve( y[:2 * N], np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),)
        return xint[2 * N - 3: -2 * N + 3]

    @staticmethod
    def stfrft(x, w, hop, a):
        """ Short Time Fractional Fourier Transform analysis of a given real input signal,
        via the above DFT method.
        Args:
            x   : 	(array)  Magnitude Spectrum
            w   :   (array)  Desired windowing function determining the analysis size
            hop :   (int)    Hop size
            a   :   (float)  Alpha factor
        Returns:
            sMx :   (2D ndarray) Stacked arrays of magnitude spectra
            sPx :   (2D ndarray) Stacked arrays of phase spectra
        """

        # Analysis Parameters
        wsz = w.size

        # Add some zeros at the start and end of the signal to avoid window smearing
        x = np.append(np.zeros(3*hop),x)
        x = np.append(x, np.zeros(3*hop))

        # Initialize sound pointers
        pin = 0
        pend = x.size - wsz
        indx = 0

        # Initialize storing matrix
        xmX = np.zeros((len(x)/hop, wsz), dtype = np.float32)
        xpX = np.zeros((len(x)/hop, wsz), dtype = np.float32)

        # Analysis Loop
        while pin <= pend:
            # Acquire Segment
            xSeg = x[pin:pin+wsz] * w

            # Perform frFT on segment
            cX = TimeFrequencyDecomposition.frft(xSeg, a)

            xmX[indx, :] = np.abs(cX)
            xpX[indx, :] = np.angle(cX)

            # Update pointers and indices
            pin += hop
            indx += 1

        return xmX, xpX

    @staticmethod
    def istfrft(xmX, xpX, hop, a):
        """ Short Time Fractional Fourier Transform synthesis of given magnitude and phase spectra.
        Args:
            xmX :   (2D ndarray)  Magnitude Spectrum
            xpX :   (2D ndarray)  Phase Spectrum
            hop :   (int)         Hop Size
            a   :   (float)       Alpha factor
        Returns :
            y   :   (array) Synthesised time-domain real signal.
        """
        # Acquire the number of frames
        numFr = xmX.shape[0]
        # Amount of samples
        wsz = xmX.shape[1]


        # Initialise output array with zeros
        y = np.zeros(numFr * hop + wsz)

        # Initialise sound pointer
        pin = 0

        # Main Synthesis Loop
        for indx in range(numFr):
            # Inverse FrFT
            cX = xmX[indx, :] * np.exp(1j*xpX[indx, :])
            ybuffer = TimeFrequencyDecomposition.ifrft(cX, a)

            # Overlap and Add
            y[pin:pin+wsz] += np.real(ybuffer)

            # Advance pointer
            pin += hop

        # Delete the extra zeros that the analysis had placed
        y = np.delete(y, range(3*hop))
        y = np.delete(y, range(y.size-(3*hop + 1), y.size))

        return y

class CepstralDecomposition:
    """ A Class that performs a cepstral decomposition based on the
        logarithmic observed magnitude spectrogram. As appears in:
        "A Novel Cepstral Representation for Timbre Modelling of
        Sound Sources in Polyphonic Mixtures", Z.Duan, B.Pardo, L. Daudet.
    """
    @staticmethod
    def computeUDCcoefficients(freqPoints = 2049, points = 2049, fs = 44100, melOption = False):
        """ Computation of M matrix that contains the coefficients for
        cepstral modelling architecture.
        Args:
            freqPoints   :     (int)  Number of frequencies to model
            points       : 	   (int)  The cepstum order (number of coefficients)
            fs           :     (int)  Sampling frequency
            melOption    :     (bool) Compute Mel-uniform discrete cepstrum
        Returns:
            M            :     (ndarray) Matrix containing the coefficients
        """
        M = np.empty((freqPoints, points), dtype = np.float32)
        # Array obtained by the order number of cepstrum
        p = np.arange(points)
        # Array with frequncy bin indices
        f = np.arange(freqPoints)

        if (freqPoints % 2 == 0):
            fftsize = (freqPoints)*2
        else :
            fftsize = (freqPoints-1)*2

        # Creation of frequencies from frequency bins
        farray = f * float(fs) / fftsize
        if (melOption):
            melf = 2595.0 * np.log10(1.0 + farray * fs/700.0)
            melHsr = 2595.0 * np.log10(1.0 + (fs/2.0) * fs/700.0)
            farray = (0.5 * melf) / (melHsr)
        else:
            farray = farray/(fs)

        twoSqrt = np.sqrt(2.0)

        for indx in range(M.shape[0]):
            M[indx, :] = (np.cos(2.0 * np.pi * p * farray[indx]))
            M[indx, 1:] *= twoSqrt

        return M

class PsychoacousticModel:
    """ Class that performs a very basic psychoacoustic model.
        Bark scaling is based on Perceptual-Coding-In-Python, [Online] :
        https://github.com/stephencwelch/Perceptual-Coding-In-Python
    """
    def __init__(self, N = 4096, fs = 44100, nfilts=24, type = 'rasta', width = 1.0, minfreq=0, maxfreq=22050):

        self.nfft = N
        self.fs = fs
        self.nfilts = nfilts
        self.width = width
        self.min_freq = minfreq
        self.max_freq = maxfreq
        self.max_freq = fs/2
        self.nfreqs = N/2
        self._LTeq = np.zeros(nfilts, dtype = np.float32)

        # Type of transformation
        self.type = type

        # Computing the matrix for forward Bark transformation
        self.W = self.mX2Bark(type)

        # Computing the inverse matrix for backward Bark transformation
        self.W_inv = self.bark2mX()

        # Non-linear superposition parameters
        self._alpha = 0.6
        self._maxb = 1./self.nfilts
        self._fa = 1./(10 ** (14.5/20.) * 10 ** (12./20.))
        self._fb = 1./(10**(7.5/20.))
        self._fbb = 1./(10**(26./20.))
        self._fd = 1./self._alpha

    def mX2Bark(self, type):
        """ Method to perform the transofrmation.
        Args :
            type : (str)        String denoting the type of transformation. Can be either
                                'rasta' or 'peaq'.
        Returns  :
            W    : (ndarray)    The transformation matrix.

        """
        if type == 'rasta':
            W = self.fft2bark_rasta()
        elif type == 'peaq':
            W = self.fft2bark_peaq()
        else:
            assert('Unknown method')

        return W

    def fft2bark_peaq(self):
        """ Method construct the weight matrix.
        Returns  :
            W    : (ndarray)    The transformation matrix, used in PEAQ evaluation.
        """

        nfft = self.nfft
        nfilts  = self.nfilts
        fs = self.fs

        # Acquire frequency analysis
        df = float(fs)/nfft

        # Acquire filter responses
        fc, fl, fu = self.CB_filters()

        W = np.zeros((nfilts, nfft))

        for k in range(nfft/2+1):
            for i in range(nfilts):
                temp = (np.amin([fu[i], (k+0.5)*df]) - np.amax([fl[i], (k-0.5)*df])) / df
                W[i,k] = np.amax([0, temp])

        return W

    def fft2bark_rasta(self):
        """ Method construct the weight matrix.
        Returns  :
            W    : (ndarray)    The transformation matrix, used in PEAQ evaluation.
        """
        minfreq = self.min_freq
        maxfreq = self.max_freq
        nfilts = self.nfilts
        nfft = self.nfft
        fs = self.fs
        width = self.width

        min_bark = self.hz2bark(minfreq)
        nyqbark = self.hz2bark(maxfreq) - min_bark

        if (nfilts == 0):
          nfilts = np.ceil(nyqbark)+1

        W = np.zeros((nfilts, nfft))

        # Bark per filter
        step_barks = nyqbark/(nfilts-1)

        # Frequency of each FFT bin in Bark
        binbarks = self.hz2bark(np.linspace(0,(nfft/2),(nfft/2)+1)*fs/nfft)

        for i in xrange(nfilts):
          f_bark_mid = min_bark + (i)*step_barks

          # Compute the absolute threshold
          self._LTeq[i] = 3.64 * (self.bark2hz(f_bark_mid + 1) / 1000.) ** -0.8 - \
          6.5*np.exp( -0.6 * (self.bark2hz(f_bark_mid + 1) / 1000. - 3.3) ** 2.) + 1e-3*((self.bark2hz(f_bark_mid + 1) / 1000.) ** 4.)

          # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
          lof = np.add(binbarks, (-1*f_bark_mid - 0.5))
          hif = np.add(binbarks, (-1*f_bark_mid + 0.5))
          W[i,0:(nfft/2)+1] = 10**(np.minimum(0, np.minimum(np.divide(hif,width), np.multiply(lof,-2.5/width))))

        return W

    def bark2mX(self):
        """ Method construct the inverse weight matrix, to map back to FT domain.
        Returns  :
            W    : (ndarray)    The inverse transformation matrix.
        """
        # Fix up the weight matrix by transposing and "normalizing"
        W_short = self.W[:,0:self.nfreqs + 1]
        WW = np.dot(W_short.T,W_short)

        WW_mean_diag = np.maximum(np.mean(np.diag(WW)), sum(WW,1))
        WW_mean_diag = np.reshape(WW_mean_diag,(WW_mean_diag.shape[0],1))
        W_inv_denom = np.tile(WW_mean_diag,(1,self.nfilts))

        W_inv = np.divide(W_short.T, W_inv_denom)

        return W_inv

    def hz2bark(self, f):
        """ Method to compute Bark from Hz.
        Args     :
            f    : (ndarray)    Array containing frequencies in Hz.
        Returns  :
            Brk  : (ndarray)    Array containing Bark scaled values.
        """

        Brk = 6. * np.arcsinh(f/600.)

        return Brk

    def bark2hz(self, Brk):
        """ Method to compute Hz from Bark scale.
        Args     :
            Brk  : (ndarray)    Array containing Bark scaled values.
        Returns  :
            Fhz  : (ndarray)    Array containing frequencies in Hz.
        """
        Fhz = 650. * np.sinh(Brk/7.)

        return Fhz

    def CB_filters(self):
        """ Method to acquire critical band filters for creation of the PEAQ FFT model.
        Returns         :
            fc, fl, fu  : (ndarray)    Arrays containing the values in Hz for the
                                       bandwidth and centre frequencies used in creation
                                       of the transformation matrix.
        """

        fl = np.array([  80.000,   103.445,   127.023,   150.762,   174.694, \
               198.849,   223.257,   247.950,   272.959,   298.317, \
               324.055,   350.207,   376.805,   403.884,   431.478, \
               459.622,   488.353,   517.707,   547.721,   578.434, \
               609.885,   642.114,   675.161,   709.071,   743.884, \
               779.647,   816.404,   854.203,   893.091,   933.119, \
               974.336,  1016.797,  1060.555,  1105.666,  1152.187, \
              1200.178,  1249.700,  1300.816,  1353.592,  1408.094, \
              1464.392,  1522.559,  1582.668,  1644.795,  1709.021, \
              1775.427,  1844.098,  1915.121,  1988.587,  2064.590, \
              2143.227,  2224.597,  2308.806,  2395.959,  2486.169, \
              2579.551,  2676.223,  2776.309,  2879.937,  2987.238, \
              3098.350,  3213.415,  3332.579,  3455.993,  3583.817, \
              3716.212,  3853.817,  3995.399,  4142.547,  4294.979, \
              4452.890,  4616.482,  4785.962,  4961.548,  5143.463, \
              5331.939,  5527.217,  5729.545,  5939.183,  6156.396, \
              6381.463,  6614.671,  6856.316,  7106.708,  7366.166, \
              7635.020,  7913.614,  8202.302,  8501.454,  8811.450, \
              9132.688,  9465.574,  9810.536, 10168.013, 10538.460, \
             10922.351, 11320.175, 11732.438, 12159.670, 12602.412, \
             13061.229, 13536.710, 14029.458, 14540.103, 15069.295, \
             15617.710, 16186.049, 16775.035, 17385.420 ])

        fc = np.array([  91.708,   115.216,   138.870,   162.702,   186.742, \
               211.019,   235.566,   260.413,   285.593,   311.136, \
               337.077,   363.448,   390.282,   417.614,   445.479, \
               473.912,   502.950,   532.629,   562.988,   594.065, \
               625.899,   658.533,   692.006,   726.362,   761.644, \
               797.898,   835.170,   873.508,   912.959,   953.576, \
               995.408,  1038.511,  1082.938,  1128.746,  1175.995, \
              1224.744,  1275.055,  1326.992,  1380.623,  1436.014, \
              1493.237,  1552.366,  1613.474,  1676.641,  1741.946, \
              1809.474,  1879.310,  1951.543,  2026.266,  2103.573, \
              2183.564,  2266.340,  2352.008,  2440.675,  2532.456, \
              2627.468,  2725.832,  2827.672,  2933.120,  3042.309, \
              3155.379,  3272.475,  3393.745,  3519.344,  3649.432, \
              3784.176,  3923.748,  4068.324,  4218.090,  4373.237, \
              4533.963,  4700.473,  4872.978,  5051.700,  5236.866, \
              5428.712,  5627.484,  5833.434,  6046.825,  6267.931, \
              6497.031,  6734.420,  6980.399,  7235.284,  7499.397, \
              7773.077,  8056.673,  8350.547,  8655.072,  8970.639, \
              9297.648,  9636.520,  9987.683, 10351.586, 10728.695, \
             11119.490, 11524.470, 11944.149, 12379.066, 12829.775, \
             13294.850, 13780.887, 14282.503, 14802.338, 15341.057, \
             15899.345, 16477.914, 17077.504, 17690.045 ])

        fu = np.array([ 103.445,   127.023,   150.762,   174.694,   198.849, \
               223.257,   247.950,   272.959,   298.317,   324.055, \
               350.207,   376.805,   403.884,   431.478,   459.622, \
               488.353,   517.707,   547.721,   578.434,   609.885, \
               642.114,   675.161,   709.071,   743.884,   779.647, \
               816.404,   854.203,   893.091,   933.113,   974.336, \
              1016.797,  1060.555,  1105.666,  1152.187,  1200.178, \
              1249.700,  1300.816,  1353.592,  1408.094,  1464.392, \
              1522.559,  1582.668,  1644.795,  1709.021,  1775.427, \
              1844.098,  1915.121,  1988.587,  2064.590,  2143.227, \
              2224.597,  2308.806,  2395.959,  2486.169,  2579.551, \
              2676.223,  2776.309,  2879.937,  2987.238,  3098.350, \
              3213.415,  3332.579,  3455.993,  3583.817,  3716.212, \
              3853.348,  3995.399,  4142.547,  4294.979,  4452.890, \
              4643.482,  4785.962,  4961.548,  5143.463,  5331.939, \
              5527.217,  5729.545,  5939.183,  6156.396,  6381.463, \
              6614.671,  6856.316,  7106.708,  7366.166,  7635.020, \
              7913.614,  8202.302,  8501.454,  8811.450,  9132.688, \
              9465.574,  9810.536, 10168.013, 10538.460, 10922.351, \
             11320.175, 11732.438, 12159.670, 12602.412, 13061.229, \
             13536.710, 14029.458, 14540.103, 15069.295, 15617.710, \
             16186.049, 16775.035, 17385.420, 18000.000 ])

        return fc, fl, fu

    def forward(self, spc):
        """ Method to transform FT domain to Bark.
        Args         :
            spc      : (ndarray)    2D Array containing the magnitude spectra.
        Returns      :
            Brk_spc  : (ndarray)    2D Array containing the Bark scaled magnitude spectra.
        """
        W_short = self.W[:, 0:self.nfreqs]
        Brk_spc = np.dot(W_short,spc)
        return Brk_spc

    def backward(self, Brk_spc):
        """ Method to reconstruct FT domain from Bark.
        Args         :
            Brk_spc  : (ndarray)    2D Array containing the Bark scaled magnitude spectra.
        Returns      :
            Xhat     : (ndarray)    2D Array containing the reconstructed magnitude spectra.
        """
        Xhat = np.dot(self.W_inv,Brk_spc)
        return Xhat

    def OutMidCorrection(self, correctionType, firOrd, fs):
        """
        Method to "correct" the middle outer ear transfer function.
            As appears in :
            - A. Härmä, and K. Palomäki, ''HUTear – a free Matlab toolbox for modeling of human hearing'',
            in Proceedings of the Matlab DSP Conference, pp 96-99, Espoo, Finland 1999.
        """
        # Lookup tables for correction
        f1 = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, \
        125, 150, 177, 200, 250, 300, 350, 400, 450, 500, 550, \
        600, 700, 800, 900, 1000, 1500, 2000, 2500, 2828, 3000, \
        3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, \
        12748, 15000])

        ELC = np.array([ 31.8, 26.0, 21.7, 18.8, 17.2, 15.4, 14.0, 12.6, 11.6, 10.6, \
            9.2, 8.2, 7.7, 6.7, 5.3, 4.6, 3.9, 2.9, 2.7, 2.3, \
            2.2, 2.3, 2.5, 2.7, 2.9, 3.4, 3.9, 3.9, 3.9, 2.7, \
            0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0, \
            10.2, 15.0, 17.0, 15.5, 11.0, 22.0])

        MAF = np.array([ 73.4, 65.2, 57.9, 52.7, 48.0, 45.0, 41.9, 39.3, 36.8, 33.0, \
            29.7, 27.1, 25.0, 22.0, 18.2, 16.0, 14.0, 11.4, 9.2, 8.0, \
             6.9,  6.2,  5.7,  5.1,  5.0,  5.0,  4.4,  4.3, 3.9, 2.7, \
             0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0, \
            10.2, 15.0, 17.0, 15.5, 11.0, 22.0])

        f2  = np.array([  125,  250,  500, 1000, 1500, 2000, 3000,  \
            4000, 6000, 8000, 10000, 12000, 14000, 16000])

        MAP = np.array([ 30.0, 19.0, 12.0,  9.0, 11.0, 16.0, 16.0, \
            14.0, 14.0,  9.9, 24.7, 32.7, 44.1, 63.7])

        if correctionType == 'ELC':
            freqTable = f1
            CorrectionTable = ELC
        elif correctionType == 'MAF':
            freqTable = f1
            CorrectionTable = MAF
        elif correctionType == 'MAP':
            freqTable = f2
            CorrectionTable = MAP
        else :
            print('Unrecongised operation: ELC will be used instead...')
            freqTable = f1
            CorrectionTable = ELC

        freqN = np.arange(0, firOrd) * fs/2. / (firOrd-1)
        spline = uspline(freqTable, CorrectionTable)
        crc = spline(freqN)
        crclin = 10. ** (-crc/ 10.)
        return crclin, freqN, crc

    def MOEar(self, correctionType = 'ELC'):
        """ Method to approximate middle-outer ear transfer function for linearly scaled
            frequency representations, using an FIR approximation of order 600 taps.
            As appears in :
            - A. Härmä, and K. Palomäki, ''HUTear – a free Matlab toolbox for modeling of human hearing'',
            in Proceedings of the Matlab DSP Conference, pp 96-99, Espoo, Finland 1999.
        Arguments          :
            correctionType : (string)     String which specifies the type of correction :
                                          'ELC' - Equal Loudness Curves at 60 dB (default)
                                          'MAP' - Minimum Audible Pressure at ear canal
                                          'MAF' - Minimum Audible Field
        Returns            :
            LTq            : (ndarray)    1D Array containing the transfer function, without the DC sub-band.
        """
        # Parameters
        firOrd = self.nfft
        Cr, fr, Crdb = self.OutMidCorrection(correctionType, firOrd, self.fs)
        Cr[self.nfft - 1] = 0.

        # FIR Design
        A = firwin2(firOrd, fr, Cr, nyq = self.fs/2)
        B = 1
        _, LTq = freqz(A, B, firOrd, self.fs)

        LTq = 20. * np.log10(np.abs(LTq))
        LTq -= max(LTq)
        return LTq[:self.nfft/2 + 1]

    def maskingThreshold(self, mX):
        """ Method to compute the masking threshold by non-linear superposition.
        As used in :
        - G. Schuller, B. Yu, D. Huang and B. Edler, "Perceptual Audio Coding Using Adaptive
        Pre and Post-filters and Lossless Compression", in IEEE Transactions on Speech and Audio Processing,
        vol. 10, n. 6, pp. 379-390, September, 2002.
        As appears in :
        - F. Baumgarte, C. Ferekidis, H Fuchs,  "A Nonlinear Psychoacoustic Model Applied to ISO/MPEG Layer 3 Coder",
        in Proceedings of the 99th Audio Engineering Society Convention, October, 1995.

        Args         :
            mX       : (ndarray)    2D Array containing the magnitude spectra (1 time frame x frequency subbands)
        Returns      :
            mT       : (ndarray)    2D Array containing the masking threshold.

        Authors      : Gerald Schuller('shl'), S.I. Mimilakis ('mis')
        """
        # Bark Scaling with the initialized, from the class, matrix W.
        if mX.shape[1] % 2 == 0:
            nyq = False
            mX = np.dot(np.abs(mX), self.W[:, :self.nfreqs].T)
        else :
            nyq = True
            mX = np.dot(np.abs(mX), self.W[:, :self.nfreqs + 1].T)

        # Parameters
        Numsubbands = mX.shape[1]
        timeFrames = mX.shape[0]
        fc = self._maxb*Numsubbands

        # Initialization of the matrix containing the masking threshold
        maskingThreshold = np.zeros((timeFrames, Numsubbands))

        for frameindx in xrange(mX.shape[0]) :
            mT = np.zeros((Numsubbands))
            for n in xrange(Numsubbands):
                for m in xrange(0, n):
                    mT[n] += (mX[frameindx, m]*self._fa * (self._fb ** ((n - m) * fc))) ** self._alpha

                for m in xrange(n+1, Numsubbands):
                    mT[n] += (mX[frameindx, m]*self._fa * (self._fbb ** ((m - n) * fc))) ** self._alpha

                mT[n] = mT[n] ** (self._fd)

            maskingThreshold[frameindx, :] = mT

        # Inverse the bark scaling with the initialized, from the class, matrix W_inv.
        if nyq == False:
            maskingThreshold = np.dot(maskingThreshold, self.W_inv[:-1, :self.nfreqs].T)
        else :
            maskingThreshold = np.dot(maskingThreshold, self.W_inv[:, :self.nfreqs].T)

        return maskingThreshold

    def NMREval(self, xn, xnhat):

        """ Method to perform NMR perceptual evaluation of audio quality between two signals.
        Args        :
            xn      :   (ndarray) 1D Array containing the true time domain signal.
            xnhat   :   (ndarray) 1D Array containing the estimated time domain signal.
        Returns     :
            NMR     :   (float)   A float measurement in dB providing a perceptually weighted
                        evaluation. Below -9 dB can be considered as in-audible difference/error.
        As appears in :
        - K. Brandenburg and T. Sporer,  “NMR and Masking Flag: Evaluation of Quality Using Perceptual Criteria,” in
        Proceedings of the AES 11th International Conference on Test and Measurement, Portland, USA, May 1992, pp. 169–179
        - J. Nikunen and T. Virtanen, "Noise-to-mask ratio minimization by weighted non-negative matrix factorization," in
         Acoustics Speech and Signal Processing (ICASSP), 2010 IEEE International Conference on, Dallas, TX, 2010, pp. 25-28.
        """
        mX, _ = TimeFrequencyDecomposition.STFT(xn, hanning(self.nfft/2 + 1), self.nfft, self.nfft/4)
        mXhat, _ = TimeFrequencyDecomposition.STFT(xnhat, hanning(self.nfft/2 + 1), self.nfft, self.nfft/4)

        # Compute Error
        Err = np.abs(mX - mXhat) ** 2.

        # Acquire Masking Threshold
        mT = self.maskingThreshold(mX)

        # Inverse the filter of masking threshold
        imT = 1./(mT + eps)

        # Outer/Middle Ear transfer function on the diagonal
        LTq = 10 ** (self.MOEar()/20.)
        LTq = LTq * np.identity(len(LTq))

        # NMR computation
        NMR = 10. * np.log10((1./mX.shape[0])*np.sum((imT * np.dot(Err, LTq))))

        return NMR

class WDODisjointness:
    """ A Class that measures the disjointness of a Time-frequency decomposition
    given the true and estimated signals. As appears in :
    - O. Yılmaz and S. Rickard, “Blind separation of speech mixtures
    via time-frequency masking,” IEEE Trans. on Signal Processing, vol. 52, no. 7, pp. 1830–1847, Jul. 2004.
    - J.J. Burred, "From Sparse Models to Timbre Learning: New Methods for Musical Source Separation", PhD Thesis,
    TU Berlin, 2009.
    - D. Gianoulis, D. Barchiesi, A. Klapuri, and M. D. Plumbley. "On the disjointness of sources
    in music using different time-frequency representations", in Proceedings of the IEEE Workshop on Applications of
    Signal Processing to Audio and Acoustics (WASPAA), 2011.

    In addition to this, measures of sparsity (l1/l2 norms) are provided.
    """

    @staticmethod
    def PSR(Mask, TrueTarget):
        """ Method to compute the Preserved-Signal Ratio (PSR) measure.
            Args:
                Mask         : 	 (2D array)  Computed Upper Bound Binary Mask to
                                             estimate the target source.
                TrueTarget   :   (2D array)  Computed Time-Frequency decomposition
                                             of target source to be separated.
        Returns:
                (float) Ratio of the squared Frobenious norms of each quantity.
        """
        num = ((np.sqrt(np.sum((Mask * TrueTarget) ** 2.))) ** 2.) + eps
        denom = ((np.sqrt(np.sum(TrueTarget ** 2.))) ** 2.) + eps
        return num/denom

    @staticmethod
    def SIR(Mask, TrueTarget, InterferingSources):
        """ Method to compute the Signal to Interference Ratio(SIR) measure.
            Args:
                Mask                 : 	 (2D array)  Computed Upper Bound Binary Mask to
                                                     estimate the target source.
                TrueTarget           :   (2D array)  Computed Time-Frequency decomposition
                                                     of target source to be separated.
                InterferingSources   :   (2D array)  Computed Time-Frequency decomposition
                                                     of interfering sources.
        Returns:
                (float) Ratio of the squared Frobenious norms of each quantity.
        """
        num = ((np.sqrt(np.sum( (Mask * TrueTarget) ** 2.))) ** 2.) + eps
        denom = ((np.sqrt(np.sum( (Mask * InterferingSources) ** 2.))) ** 2.) + eps
        return num/denom

    @staticmethod
    def WDO(PSR, SIR):
        """ Method to compute the objective W-Disjoint Orthogonality(WDO) measure.
            Args:
                PSR   : (float)  Computed Preserved-Signal Ratio (PSR) measure.
                SIR   : (float)  Signal to Interference Ratio(SIR) measure.

            Returns:
                (float) WDO Measure
        """
        return PSR - ((PSR + eps)/(SIR + eps))

    @staticmethod
    def l1l2_sparsity_measure(mX):
        """ Method to compute the sparsity of a representation. As appears in :
            -A. Repetti, M.Q. Pham,L. Duval, E. Chouzenoux and J.C. Pesquet, "Euclid in a Taxicab:
            Sparse Blind Deconvolution with Smoothed l1/l2 Regularization", 2015, IEEE Signal Processing Letters.
            -N. Hurley and S. Rickard, "Comparing Measures of Sparsity,"
            in IEEE Transactions on Information Theory, vol. 55, no. 10, pp. 4723-4741, Oct. 2009.

            Args:
                mX      : (2D Array)  Time-frequency decomposition of a mixture signal.

            Returns:
                (float) : Sparsity Measure
        """

        return np.sum(np.abs((mX + eps) / ((np.sqrt(np.sum(mX ** 2.))) + eps)))

    @staticmethod
    def gini_index(mX):
        """ Method to compute the sparsity of a given representation. As appears in :
            -N. Hurley and S. Rickard, "Comparing Measures of Sparsity,"
            in IEEE Transactions on Information Theory, vol. 55, no. 10, pp. 4723-4741, Oct. 2009.

            Args:
                mX      : (2D Array)  Time-frequency decomposition of a mixture signal.

            Returns:
                GI      : (1D Array)  Sparsity Measure
        """

        GI = np.zeros(mX.shape[0], dtype = np.float32)
        for frame in xrange(0, mX.shape[0]):
            vector = mX[frame, :]
            sortedVector = np.sort(np.abs(vector), axis = 0)
            l1norm = np.sum(sortedVector) + eps
            if l1norm == 0 or l1norm <= eps:
                GI[frame] = 0
            else :
                cgi = 0.
                k = np.arange(len(sortedVector)) + 1
                cgi = (sortedVector[k - 1]/l1norm) * ((len(sortedVector) - k + 0.5)/len(sortedVector))
                GI[frame] = (1./len(sortedVector) + 1. - 2.* np.sum(cgi))

        return GI

if __name__ == "__main__":
    import IOMethods as IO
    import matplotlib.pyplot as plt
    np.random.seed(218)
    # Test
    #kSin = np.cos(np.arange(88200) * (1000.0 * (3.1415926 * 2.0) / 44100)) * 0.5
    mix, fs = IO.AudioIO.audioRead('testFiles/cmente.mp3', mono = True)
    mix = mix[44100*25:44100*25 + 882000] * 0.25
    noise = np.random.uniform(-30., 30., len(mix))

    # STFT/iSTFT Test
    w = np.hanning(1025)
    magX, phsX =  TimeFrequencyDecomposition.STFT(mix, w, 2048, 512)
    magN, phsN =  TimeFrequencyDecomposition.STFT(noise, w, 2048, 512)
    # MDC(S)T Test
    #yA = TimeFrequencyDecomposition.complex_analysis(mix[:, 0], 1024)
    #yrek = TimeFrequencyDecomposition.complex_synthesis(yA, 1024)

    #y = np.real(y)

    # Usage of psychoacoustic model
    # Initialize the model
    pm = PsychoacousticModel(N = 2048, fs = 44100, nfilts = 64)
    # Acquire the response
    LTeq = 10 ** (pm.MOEar()/20.)
    # Compute masking threshold
    mt = pm.maskingThreshold(magX)
    #mt_y = pm.maskingThreshold(np.abs(y))

    sound = TimeFrequencyDecomposition.iSTFT(magX, phsX, 2048, 512)
    #sound2 = TimeFrequencyDecomposition.iSTFT(magN * (mt), phsN, 2048, 512)

    # Test the NMR Function
    #NMR = pm.NMREval(sound, sound+sound2)
    #print(NMR)

    import QMF.qmf_realtime_class as qrf
