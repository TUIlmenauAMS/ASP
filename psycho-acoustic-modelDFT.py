# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis, G. Schuller'
__copyright__ = 'MacSeNet, TU Ilmenau'
#Using original DFT for signal and noise, not magnitude and phase

import sys, os
#current_dir = os.path.dirname(os.path.realpath(__file__))
#print(current_dir + '/..')
#sys.path.insert(0, current_dir + '/..')
import IOMethods as IO
import TFMethods as TF
import numpy as np
#import numpy.fft
#from scipy.fftpack import fft, ifft
from numpy.fft import fft, ifft
#import matplotlib.pyplot as plt
import pyaudio
import wave
import time
import pygame

# Parameters
wsz = 1024
N = 2048
hop = 512
#hop = 1024
gain = 0.

# Reading
# File Reading
#x, fs = IO.AudioIO.wavRead('mixed.wav', mono = True)
x, fs = IO.AudioIO.wavRead('sc03_16m.wav', mono = True)
x *= 1.0
# Cosine testq
#x = np.cos(np.arange(88200) * (1000.0 * (3.1415926 * 2.0) / 44100)) * 0.1
#fs = 44100
# Generate noise. Scaling was found experimentally to perfectly match the masking threshold magnitude.
#should be according to equal energy, with quantization interval Delta, DFT and mask give "voltages", 
#squared voltages are power.
#Delta is the quantization step size in a coder,
# Delta= max noise - min noise,, Delta=1 means uniform dist from -0.5 to +0.5,
#noise Energy E=Delta^2/12=mask^2, Delta/sqrt(12)=mask, 
# hence Delta=mask*sqrt(12)
# We only have real valued noise, not imaginary: We need factor 2 in power to make up for 
# missing imag. part
#We only use half of the DFT of the spectrum, hence we need another factor of 2 in power to 
#compensate for it, hence a factor of sqrt(4)=2 in "voltage": 
#noise = np.random.uniform(-30., 30., (len(x), 1))
noise = np.random.uniform(-.5, .5, (len(x), 1))*np.sqrt(12)*2

# STFT/iSTFT Test
w = np.bartlett(wsz)
# Normalise windowing function
w = w / sum(w)

#Sine window: choose this for sine windowing for energy conservation (Parseval Theorem):
w=np.sin(np.pi/wsz*np.arange(0.5,wsz))

print("w.shape = ", w.shape)

# Initialize psychoacoustic mode
pm = TF.PsychoacousticModel(N = N, fs = fs, nfilts = 64)

# Visual stuff
option = 'pygame'
#option = 'matplotlib'

# Pygame visual handles
pygame.init()
pygame.display.set_caption("Masking Threshold Visualization")
color = pygame.Color(255, 0, 0, 0)
color2 = pygame.Color(0, 255, 0, 0)
color3 = pygame.Color(0, 110, 0, 0)
background_color = pygame.Color(0, 0, 0, 0)
screen = pygame.display.set_mode((wsz + 40, 480))
#screenbg = pygame.display.set_mode((wsz + 40, 480))
screenbg = pygame.Surface((wsz + 40, 480))
#screenbg.set_alpha(100)
screenbg.fill(background_color)

dBScales = np.linspace(-120, 0, 25)
dBpos = (dBScales/-120 * 480)

# Draw labels and help text on background surface "screenbg":
# Done here to get it out of the runtime loop.
font = pygame.font.Font(None, 24)
xlabel = font.render("Frequency sub-bands", 1, (100, 100, 250))
ylabel = font.render("Magnitude in dB FS", 1, (100, 100, 250))
legendA = font.render("Magnitude Spectrum", 1, (250, 0, 0))
legendB = font.render("Estimated masking threshold", 1, (0, 250, 0))
ylabel = pygame.transform.rotate(ylabel, 90)

screenbg.blit(xlabel, (895, 460))

screenbg.blit(ylabel, (0, 5))
screenbg.blit(legendA, (800, 0))
screenbg.blit(legendB, (800, 15))
#offset = font.render("Masking Threshold Offset (NMR) in dB: " + str(gain), 1, (0, 250, 0))
#bpc = font.render("Est. average bits per subband: " + str(bc), 1, (190, 160, 110))
helptext = font.render("(Adjust the threshold by pressing 'Up' & 'Down' Arrow keys)", 1, (0, 250, 0))
#screenbg.blit(offset, (300, 0))
screenbg.blit(helptext, (300, 15))
#screenbg.blit(bpc, (300, 30))

for n2 in xrange(len(dBpos)):
    dB = font.render(str(np.int(dBScales[n2])), 1, (0,120,120))
    screenbg.blit(dB, (20, int(dBpos[n2])))

pygame.display.flip()

if option == 'matplotlib':
# Using matplotlib
    plt.ion()
    ax = plt.axes(xlim=(0, wsz), ylim=(-120, 0))
    line, = plt.plot(np.bartlett(wsz), label = 'Signal')
    line2, = plt.plot(np.bartlett(wsz), label = 'Masking Threshold')
    plt.xlabel('Frequency sub-bands')
    plt.ylabel('dB FS')
    plt.legend(handles=[line, line2])
    plt.show()

# Main Loop
run = True
while run == True:
    # Initialize sound pointers and buffers
    pin = 0
    pend = x.size - wsz - hop
    #b_mX = np.zeros((1, wsz + 1), dtype = np.float32)
    #b_nX = np.zeros((1, wsz + 1), dtype = np.float32)
    b_mX = np.zeros((1, wsz ), dtype = np.float32)
    b_nX = np.zeros((1, wsz), dtype = np.float32)
    #mt = np.zeros((1, wsz + 1), dtype = np.float32)
    mt = np.zeros((1, wsz), dtype = np.float32)
    output_buffer = np.zeros((1, wsz), dtype = np.float32)
    ola_buffer = np.zeros((1, wsz+hop), dtype = np.float32)
    prv_seg = np.zeros((1, 1024), dtype = np.float32)
    noiseMagnitude = np.ones((1, 1025), dtype = np.float32)

    # For less frequent plotting to avoide buffer underruns
    indx = 0

    # Reshaping the signal
    x = x.reshape(len(x), 1)

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
        if indx % 30 == 0:
            # Maximum quantization rate computation
            bc = (np.log2( (b_mX[0, :] + 1e-16)/(mt[0, :] + 1e-16)))
            bc = (np.mean(bc[bc >= 0 ]))

            if option == 'matplotlib' :
                # Matplotlib
                line.set_ydata(20. * np.log10(b_mX[0, :-1] + 1e-16))
                # Check for scaling the noise!
                line2.set_ydata(20. * np.log10(mt[0, :-1] + 1e-16))
                plt.draw()
                plt.pause(0.00001)

            else :
                # Pygame
                #screen.fill(background_color)
		screen.blit(screenbg,(0,0))
                prv_pos = (60, 480)
                prv_pos2 = (60, 480)
                prv_pos3 = (60, 480)
                for n in xrange(0, wsz):
                    val = 20. * np.log10(b_mX[0, n] + 1e-16)
                    val2 = 20. * np.log10(mt[0, n] + 1e-16)
                    val3 = 20. * np.log10(b_nX[0, n] * mt[0, n] + 1e-16)
                    val3 /= -120
                    val /= -120
                    val2/= -120
                    val *= 480
                    val2 *= 480
                    val3 *= 480
                    position = (n + 60, int(val))
                    position2 = (n + 60, int(val2))
                    position3 = (n + 60, int(val3))
                    pygame.draw.line(screen, color, prv_pos, position)
                    pygame.draw.line(screen, color2, prv_pos2, position2)
                    pygame.draw.line(screen, color3, prv_pos3, position3)
                    prv_pos = position
                    prv_pos2 = position2
                    prv_pos3 = position3

                # Print the surface
		"""
                screen.blit(xlabel, (895, 460))
                screen.blit(ylabel, (0, 5))
                screen.blit(legendA, (800, 0))
                screen.blit(legendB, (800, 15))
		"""
                offset = font.render("Masking Threshold Offset (NMR) in dB: " + str(gain), 1, (0, 250, 0))
		bpc = font.render("Est. average bits per subband: " + str(bc), 1, (190, 160, 110))
		screen.blit(bpc, (300, 30))
		screen.blit(offset, (300, 0))
		"""
                helptext = font.render("(Adjust the threshold by pressing 'Up' & 'Down' Arrow keys)", 1, (0, 250, 0))
                
                screen.blit(helptext, (300, 15))
                for n2 in xrange(len(dBpos)):
                    dB = font.render(str(np.int(dBScales[n2])), 1, (0,120,120))
                    screen.blit(dB, (20, int(dBpos[n2])))
		"""
                # Display
		
                pygame.display.flip()

        # Acquire Segment
        xSeg = x[pin:pin+wsz, 0]
        nSeg = noise[pin:pin+wsz, 0]

        # Perform DFT on segment
        #mX, pX = TF.TimeFrequencyDecomposition.DFT(xSeg, w, N)
	X=fft(xSeg*w,n=N,norm='ortho')[:N/2]
        #print "size xSeg: ", xSeg.shape
	mX=np.abs(X)
        #nX, npX = TF.TimeFrequencyDecomposition.DFT(nSeg, w, N)

        # Set it to buffer
        b_mX[0, :] = mX
        #b_nX[0, :] = nX
	b_nX[0, :] = np.abs(nSeg)
        # Masking threshold
        mt = pm.maskingThreshold(b_mX) * (10**(gain/20.))
	#remove last entry (Nyquist freq)
	#mt=mt[:,:]
	#print "mt.shape= ", mt.shape
        # Resynthesize
        #nSeg = TF.TimeFrequencyDecomposition.iDFT(nX * mt[0, :], npX, wsz)
	#Add noise according to masking threshold:
	mixf=X+mt[0,:]*nSeg
        #xSeg = TF.TimeFrequencyDecomposition.iDFT(mX, pX, wsz)
	#choose this for Bartlett window:
	#mix=np.real(ifft(mixf, n=N,norm='ortho')[:N/2])*hop
	#Choose this for sine windowing:
        mix=np.real(ifft(mixf, n=N,norm='ortho')[:N/2])*w
        #mix = (xSeg + nSeg) * hop

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
