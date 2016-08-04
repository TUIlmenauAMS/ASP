# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import numpy as np
from scipy.fftpack import fft, ifft
from TFMethods import TimeFrequencyDecomposition as TF
realmin = np.finfo(np.double).tiny

class FrequencyMasking:
	"""Class containing various time-frequency masking methods, for processing Time-Frequency representations.
	"""

	def __init__(self, mX, sTarget, nResidual, psTarget = [], pnResidual = [], alpha = 1.2, method = 'Wiener'):
		self._mX = mX
		self._eps = np.finfo(np.float).eps
		self._sTarget = sTarget
		self._nResidual = nResidual
		self._pTarget = psTarget
		self._pY = pnResidual
		self._mask = []
		self._Out = []
		self._alpha = alpha
		self._method = method

	def __call__(self, reverse = False):

		if (self._method == 'Phase'):
			if not self._pTarget.size or not self._pTarget.size:
				raise ValueError('Phase-sensitive masking cannot be performed without phase information.')
			else:
				FrequencyMasking.phaseSensitive(self)
				if not(reverse) :
					FrequencyMasking.applyMask(self)
				else :
					FrequencyMasking.applyReverseMask(self)

		elif (self._method == 'IRM'):
			FrequencyMasking.IRM(self)
			if not(reverse) :
				FrequencyMasking.applyMask(self)
			else :
				FrequencyMasking.applyReverseMask(self)

		elif (self._method == 'IBM'):
			FrequencyMasking.IBM(self)
			if not(reverse) :
				FrequencyMasking.applyMask(self)
			else :
				FrequencyMasking.applyReverseMask(self)

		elif (self._method == 'UBBM'):
			FrequencyMasking.UBBM(self)
			if not(reverse) :
				FrequencyMasking.applyMask(self)
			else :
				FrequencyMasking.applyReverseMask(self)


		elif (self._method == 'Wiener'):
			FrequencyMasking.Wiener(self)
			if not(reverse) :
				FrequencyMasking.applyMask(self)
			else :
				FrequencyMasking.applyReverseMask(self)

		elif (self._method == 'alphaWiener'):
			FrequencyMasking.alphaHarmonizableProcess(self)
			if not(reverse) :
				FrequencyMasking.applyMask(self)
			else :
				FrequencyMasking.applyReverseMask(self)

		return self._Out

	def IRM(self):
		"""
			Computation of Ideal Ratio Mask. As appears in :
			H Erdogan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux,
	   		"Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks,"
	   		in ICASSP 2015, Brisbane, April, 2015.
		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""

		self._mask = np.divide(self._sTarget, (self._eps + self._sTarget + self._nResidual))

	def IBM(self):
		"""
			Computation of Ideal Binary Mask.
		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		theta = 0.5
		mask = np.divide(self._sTarget ** self._alpha, (self._eps + self._nResidual ** self._alpha))
		bg = np.where(mask >= theta)
		sm = np.where(mask < theta)
		mask[bg[0],bg[1]] = 1.
		mask[sm[0], sm[1]] = 0.
		self._mask = mask

	def UBBM(self):
		"""
			Computation of Upper Bound Binary Mask. As appears in :
			- J.J. Burred, "From Sparse Models to Timbre Learning: New Methods for Musical Source Separation", PhD Thesis,
			TU Berlin, 2009.

		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component (Should not contain target source!)
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values
		"""

		mask = 20. * np.log(self._eps + np.divide((self._eps + (self._sTarget ** self._alpha)),
									  ((self._eps + (self._nResidual ** self._alpha)))))
		bg = np.where(mask >= 0)
		sm = np.where(mask < 0)
		mask[bg[0],bg[1]] = 1.
		mask[sm[0], sm[1]] = 0.
		self._mask = mask

	def Wiener(self):
		"""
			Computation of Wiener-like Mask. As appears in :
			H Erdogan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux,
	   		"Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks,"
	   		in ICASSP 2015, Brisbane, April, 2015.
		Args:
				sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
				nResidual: (2D ndarray) Magnitude Spectrogram of the residual component
		Returns:
				mask:      (2D ndarray) Array that contains time frequency gain values
		"""
		print('Wiener-like Mask')
		localsTarget = np.multiply(self._sTarget, self._sTarget)
		localnResidual = np.multiply(self._nResidual, self._nResidual)

		self._mask = np.divide(localsTarget, (self._eps + localsTarget + localnResidual))

	def alphaHarmonizableProcess(self):
		"""
			Computation of alpha harmonizable Wiener like mask, as appears in :
			A. Liutkus, R. Badeau, "Generalized Wiener filtering with fractional power spectrograms",
    		40th International Conference on Acoustics, Speech and Signal Processing (ICASSP),
    		Apr 2015, Brisbane, Australia. IEEE, 2015
		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		print('Harmonizable Process with alpha:', str(self._alpha))
		localsTarget = self._sTarget ** self._alpha
		numElements = len(self._nResidual)
		if numElements > 1:
			localnResidual = self._nResidual[0] ** self._alpha
			for indx in range(1, numElements):
				localnResidual += self._nResidual[indx] ** self._alpha
		else :
			localnResidual = self._nResidual ** self._alpha
			 
		self._mask = np.divide((localsTarget + self._eps), (self._eps + localnResidual))

	def phaseSensitive(self):
		"""
			Computation of Phase Sensitive Mask. As appears in :
			H Erdogan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux,
	   		"Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks,"
	   		in ICASSP 2015, Brisbane, April, 2015.

		Args:
			mTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			pTarget:   (2D ndarray) Phase Spectrogram of the output component
			mY:        (2D ndarray) Magnitude Spectrogram of the output component
			pY:        (2D ndarray) Phase Spectrogram of the output component
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		print('Phase Sensitive Masking.')
		# Compute Phase Difference
		Theta = (self._pTarget - self._pY)

		self._mask = 2./ (1. + np.exp(-np.multiply(np.divide(self._sTarget, self._eps + self._nResidual), np.cos(Theta)))) - 1.

	def applyMask(self):
		""" Compute the filtered output spectrogram.
		Args:
			mask:   (2D ndarray) Array that contains time frequency gain values
			mX:     (2D ndarray) Input Magnitude Spectrogram
		Returns:
			Y:      (2D ndarray) Filtered version of the Magnitude Spectrogram
		"""

		self._Out = np.multiply(self._mask, self._mX)

	def applyReverseMask(self):
		""" Compute the filtered output spectrogram, reversing the gain values.
		Args:
			mask:   (2D ndarray) Array that contains time frequency gain values
			mX:     (2D ndarray) Input Magnitude Spectrogram
		Returns:
			Y:      (2D ndarray) Filtered version of the Magnitude Spectrogram
		"""

		self._Out = np.multiply( (1. - self._mask), self._mX)

if __name__ == "__main__":

	# Small test
	kSin = (0.5 * np.cos(np.arange(4096) * (1000.0 * (3.1415926 * 2.0) / 44100)))
	noise = (np.random.uniform(-0.25,0.25,4096))
	# Noisy observation
	obs = (kSin + noise)

	kSinX = fft(kSin, 4096)
	noisX = fft(noise, 4096)
	obsX  = fft(obs, 4096)

	sigrms = np.sqrt((sum((obs-noise)**2))/len(kSin))
	nsrms = np.sqrt((sum((obs-kSin)**2))/len(noise))
	dummySNR = 20*np.log10((sigrms) / (nsrms))

	# Wiener Case (same with IRM case)
	mask = FrequencyMasking(np.abs(obsX), np.abs(kSinX), np.abs(noisX), [], [], 'Wiener')
	out1 = mask()
	sigR = np.real(ifft(out1))
	nosR = np.real(ifft(np.abs(np.abs(obsX) - out1)))

	sigrms = np.sqrt(sum(sigR**2)/len(sigR))
	nsrms = np.sqrt(sum(nosR**2)/len(nosR))
	M1dummySNR = 20*np.log10((sigrms) / (nsrms))

	# Phase Case
	mask = FrequencyMasking(np.abs(obsX), np.abs(kSinX), np.abs(noisX), np.angle(kSinX), np.angle(noisX), 'Phase')
	out2 = mask()
	sigR2 = np.real(ifft(out2))
	nosR2 = np.real(ifft(np.abs(np.abs(obsX) - out2)))

	sigrms2 = np.sqrt(sum(sigR2**2)/len(sigR2))
	nsrms2 = np.sqrt(sum(nosR2**2)/len(nosR2))
	M2dummySNR = 20*np.log10((sigrms2) / (nsrms2))