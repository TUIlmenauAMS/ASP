# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import IOMethods as IO
import TFMethods as TF
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import wave
import time

# Class for key catch
class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()

class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

getch = _Getch()

# Parameters
wsz = 1024
N = 2048
hop = 512
gain = 0.

# Reading
x = np.cos(np.arange(88200) * (1000.0 * (3.1415926 * 2.0) / 44100)) * 0.1
fs = 44100
# Generate noise. Scaling was found experimentally to perfectly match the masking threshold magnitude.
noise = np.random.uniform(-30., 30., len(x))

# STFT/iSTFT Test
w = np.bartlett(wsz)
# Normalise windowing function
w = w / sum(w)

# Initialize psychoacoustic mode
pm = TF.PsychoacousticModel(N = N, fs = fs, nfilts = 64)

# Visual stuff
plt.ion()
ax = plt.axes(xlim=(0, wsz), ylim=(-120, 0))
line, = plt.plot(np.bartlett(wsz), label = 'Signal')
line2, = plt.plot(np.bartlett(wsz), label = 'Masking Threshold')
plt.xlabel('Frequency sub-bands')
plt.ylabel('dB FS')
plt.legend(handles=[line, line2])
plt.show()

# Main Loop
# Iterations to set gain to masking threshold
# Adjust iterations to find gain
iterations = 1
for iter in xrange(iterations):
    # Initialize sound pointers and buffers
    pin = 0
    pend = x.size - wsz - hop
    b_mX = np.zeros((1, wsz + 1), dtype = np.float32)
    mt = np.zeros((1, wsz + 1), dtype = np.float32)
    output_buffer = np.zeros((1, wsz), dtype = np.float32)
    ola_buffer = np.zeros((1, wsz+hop), dtype = np.float32)
    prv_seg = np.zeros((1, 1024), dtype = np.float32)
    noiseMagnitude = np.ones((1, 1025), dtype = np.float32)
    # For less frequent plotting to avoide buffer underruns
    indx = 0

    # Streaming
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs,output=True, frames_per_buffer = wsz)

    print('More gain? Hit "+" for a 3dB increase')
    key = getch()
    if key == '+':
        gain += 3.

    while pin <= pend:
    	if indx % 20 == 0:
    	    # Visual stuff
            line.set_ydata(20. * np.log10(b_mX[0, :-1] + 1e-16))
            # Check for scaling the noise!
            #line.set_ydata(20. * np.log10(mt[0, :-1]*nX[:-1] + 1e-16))
            line2.set_ydata(20. * np.log10(mt[0, :-1] + 1e-16))
            plt.draw()
            plt.pause(0.00001)

        # Acquire Segment
        xSeg = x[pin:pin+wsz]
        nSeg = noise[pin:pin+wsz]

        # Perform DFT on segment
        mX, pX = TF.TimeFrequencyDecomposition.DFT(xSeg, w, N)
        nX, npX = TF.TimeFrequencyDecomposition.DFT(nSeg, w, N)

        # Set it to buffer
        b_mX[0, :] = mX

        # Masking threshold
        mt = pm.maskingThreshold(b_mX) * (10**(gain/20.))

        # Resynthesize
        nSeg = TF.TimeFrequencyDecomposition.iDFT(nX * mt[0, :], npX, wsz)
        xSeg = TF.TimeFrequencyDecomposition.iDFT(mX, pX, wsz)
        mix = (xSeg + nSeg) * hop

        ola_buffer[0, 0:wsz] = prv_seg
        ola_buffer[0, hop:hop+wsz] += mix

        # Place it to output buffer
        output_buffer = ola_buffer[:, :wsz]

        # Playback
        writedata = output_buffer[0, :].astype(np.float32).tostring()
        stream.write(writedata, num_frames = wsz/2, exception_on_underflow = False)
        
        # Store previous frame samples
        prv_seg = ola_buffer[:, hop:wsz+hop]

        # Clear the overlap
        ola_buffer = np.zeros((1, wsz+hop), dtype = np.float32)

        # Update pointer and index
        pin += hop
        indx += 1

plt.close()
stream.stop_stream()
stream.close()
p.terminate()

