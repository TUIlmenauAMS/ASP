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
import pygame

# Parameters
wsz = 1024
N = 2048
hop = 512
gain = 0.

# Reading
# File Reading
#x, fs = IO.AudioIO.audioRead('testFiles/dissection.mp3', mono = True)
#x = x[0:882000] * 0.1
# Cosine test
x = np.cos(np.arange(88200) * (1000.0 * (3.1415926 * 2.0) / 44100)) * 0.01
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
option = 'pygame'
# Using matplotlib

plt.ion()
ax = plt.axes(xlim=(0, wsz), ylim=(-120, 0))
line, = plt.plot(np.bartlett(wsz), label = 'Signal')
line2, = plt.plot(np.bartlett(wsz), label = 'Masking Threshold')
plt.xlabel('Frequency sub-bands')
plt.ylabel('dB FS')
plt.legend(handles=[line, line2])
plt.show()

# Pygame visual handles
pygame.init()
pygame.display.set_caption("Masking Threshold Visualization")
color = pygame.Color(255, 0, 0, 0)
color2 = pygame.Color(0, 255, 0, 0)
background_color = pygame.Color(0, 0, 0, 0)
screen = pygame.display.set_mode((wsz, 480))
screen.fill(background_color)
surface = pygame.display.set_mode((wsz, 480))
surface.fill(background_color)
surface2 = pygame.display.set_mode((wsz, 480))
surface2.fill(background_color)
pygame.display.flip()



# Main Loop
iterations = 1
run = True
while run == True:
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

    while pin <= pend:

        # Key events for the arrows. Up Arrow : + 1 dB Down Arrow : -1 dB
        cevent = pygame.event.get()
        if cevent != [] :
            if cevent[0].type == pygame.KEYDOWN:
                if cevent[0].key == pygame.K_q:
                    run = False
                    pygame.display.quit()
                    pygame.quit()
                    break
                elif cevent[0].key == 273 :
                    gain += 1.
                elif cevent[0].key == 274 :
                    gain -= 1.

        # Visual stuff
        if indx % 10 == 0:
            if option == 'matplotlib' :
                # Matplotlib
                line.set_ydata(20. * np.log10(b_mX[0, :-1] + 1e-16))
                # Check for scaling the noise!
                #line.set_ydata(20. * np.log10(mt[0, :-1]*nX[:-1] + 1e-16))
                line2.set_ydata(20. * np.log10(mt[0, :-1] + 1e-16))
                plt.draw()
                plt.pause(0.00001)

            else :
                # Pygame
                surface.fill(background_color)
                for n in range(0, wsz):
                    val = 20. * np.log10(b_mX[0, n] + 1e-16)
                    val2 = 20. * np.log10(mt[0, n] + 1e-16)
                    val /= -120
                    val2/= -120
                    val *= 480
                    val2 *= 480
                    surface.set_at((n, np.int(val2)), color)
                    surface2.set_at((n, np.int(val)), color2)

                # Print the surface
                screen.blit(surface, (0, 0))
                screen.blit(surface2, (0, 0))
                # Display
                pygame.display.flip()

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
pygame.display.quit()
pygame.quit()