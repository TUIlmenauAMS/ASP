#This is a library that implements a QMF filter bank, including a realt time implelentation.
#The "main" is an example, which reads in an audio .wav file, block-wise, computes the QMF subbands, 
#then reconstructs the audio signal from the subbands, and writes them, block wise, into another audio .wav file. 
#As such, it can read and process arbitrary long audio files, without memory problems!
#example call from a terminal window: python qmf_realtime.py
#It loads the qmf coefficient file 'qmf1024_8x.mat'
# 
#Implements a real time Pseudo Quandrature Mirror Filter Bank
#"real time" means: it reads a block of N samples in and generates a block of N spectral samples from the QMF.
#The QMF implementaion uses internal memory to accomodate the window overlap
#Gerald Schuller, gerald.schuller@tu-ilmenau.de, January 2016

import numpy as np
import scipy.fftpack as spfft

N=1024  #number of subbands of the QMF filter bank
#N=64
#internal memory for the input blocks, for 8 times overlap:
overlap=8;
#overlap=10;
blockmemory=np.zeros((overlap,N))
blockmemorysyn=np.zeros((overlap,N))

def ha2Pa3d(ha,N):
	#usage: Pa=ha2Pa3d(ha,N);
	#produces the analysis polyphase matrix Pa
	#in 3D matrix representation
	#from a basband filter ha with
	#a cosine modulation
	#N: Blocklength
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Dec-2-15

	import numpy as np

	L=len(ha);

	blocks=int(np.ceil(L/N));
        #print(blocks)

	Pa=np.zeros((N,N,blocks));

	for k in range(N): #subband
	  for m in range(blocks):  #m: block number 
	    for nphase in range(N): #nphase: Phase 
	      n=m*N+nphase;
	      #indexing like impulse response, phase index is reversed (N-np):
	      Pa[N-1-nphase,k,m]=ha[n]*np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(blocks*N-1-n-N/2.0+0.5)); 

	return Pa

def hs2Ps3d(hs,N):
	#usage: Ps=hs2Ps3d(hs,N);
	#produces the synthesis polyphase matrix Ps
	#in 3D matrix representation
	#from a basband filter hs with
	#a cosine modulation
	#N: Blocklength
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Dec-2-15

	import numpy as np

	L=len(hs);

	blocks=int(np.ceil(L/N));
        #print(blocks)

	Ps=np.zeros((N,N,blocks));

	for k in range(N): #subband
	  for m in range(blocks):  #m: block number 
	    for nphase in range(N): #nphase: Phase 
	      n=m*N+nphase;
	      #synthesis:
	      Ps[k,nphase,m]=hs[n]*np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(n-N/2.0+0.5)); 

	return Ps


def ha2Fa3d(ha,N):
	#usage: Fa=ha2Fa3d(ha,N);
	#produces the analysis polyphase folding matrix Fa with all polyphase components
	#in 3D matrix representation
	#from a basband filter ha with
	#a cosine modulation
	#N: Blocklength
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Dec-2-15
	print "ha2Pa3d:"
	Pa=ha2Pa3d(ha,N);
	print "polmatmult DCT:"
	Fa=polmatmult(Pa,DCToMatrix(N))
        #round zeroth polyphase component to 7 decimals after point:
        Fa=np.around(Fa,8)

	return Fa

def ha2Fa3d_fast(qmfwin,N):
	#usage: Fa=ha2Fa3d_fast(ha,N);
	#produces the analysis polyphase folding matrix Fa with all polyphase components
	#in 3D matrix representation
	#from a basband filter ha with
	#a cosine modulation
	#N: Blocklength
	#using a fast implementation (important for large N)
	#See my book chapter about "Filter Banks", cosine modulated filter banks.
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Jan-23-16

	Fa=np.zeros((N,N,overlap))
 	for m in range(overlap/2):
	   Fa[:,:,2*m]+=np.fliplr(np.diag(np.flipud(-qmfwin[m*2*N:(m*2*N+N/2)]*((-1)**m)),k=-N/2))
	   Fa[:,:,2*m]+=(np.diag(np.flipud(qmfwin[m*2*N+N/2:(m*2*N+N)]*((-1)**m)),k=N/2))
	   Fa[:,:,2*m+1]+=(np.diag(np.flipud(qmfwin[m*2*N+N:(m*2*N+1.5*N)]*((-1)**m)),k=-N/2))
	   Fa[:,:,2*m+1]+=np.fliplr(np.diag(np.flipud(qmfwin[m*2*N+1.5*N:(m*2*N+2*N)]*((-1)**m)),k=N/2))
	   #print -qmfwin[m*2*N:(m*2*N+N/2)]*((-1)**m)
	return Fa

def hs2Fs3d_fast(qmfwin,N):
	#usage: Fs=hs2Fs3d_fast(hs,N);
	#produces the synthesis polyphase folding matrix Fs with all polyphase components
	#in 3D matrix representation
	#from a basband filter ha with
	#a cosine modulation
	#N: Blocklength
	#Fast implementation
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Jan-23-15

	#Fa=ha2Fa3d_fast(hs,N)
	#print "Fa.shape in hs2Fs : ", Fa.shape
	#Transpose first two dimensions to obtain synthesis folding matrix:
	#Fs=np.transpose(Fa, (1, 0, 2))
	global overlap
	Fs=np.zeros((N,N,overlap))
 	for m in range(overlap/2):
	   Fs[:,:,2*m]+=np.fliplr(np.diag(np.flipud(qmfwin[m*2*N:(m*2*N+N/2)]*((-1)**m)),k=N/2))
	   Fs[:,:,2*m]+=(np.diag((qmfwin[m*2*N+N/2:(m*2*N+N)]*((-1)**m)),k=N/2))
	   Fs[:,:,2*m+1]+=(np.diag((qmfwin[m*2*N+N:(m*2*N+1.5*N)]*((-1)**m)),k=-N/2))
	   Fs[:,:,2*m+1]+=np.fliplr(np.diag(np.flipud(-qmfwin[m*2*N+1.5*N:(m*2*N+2*N)]*((-1)**m)),k=-N/2))
	#print "Fs.shape in hs2Fs : ", Fs.shape
	#avoid sign change after reconstruction:
	return -Fs

def hs2Fs3d(hs,N):
	#usage: Fs=hs2Fs3d(hs,N);
	#produces the synthesis polyphase folding matrix Fs with all polyphase components
	#in 3D matrix representation
	#from a basband filter ha with
	#a cosine modulation
	#N: Blocklength
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Dec-2-15

	Ps=hs2Ps3d(hs,N);
	Fs=polmatmult(DCToMatrix(N),Ps)
        #round zeroth polyphase component to 7 decimals after point:
        Fs=np.around(Fs,8)

	return Fs

def DCToMatrix(N):
	#produces an odd DCT matrix with size NxN
	#Gerald Schuller, Dec. 2015

	import numpy as np

	y=np.zeros((N,N,1));

	for n in range(N):
	   for k in range(N):
	      y[n,k,0]=np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(n+0.5));
	      #y(n,k)=cos(pi/N*(k-0.5)*(n-1));
	return y   

def polmatmult(A,B):
	#function C=polmatmult(A,B)
	#multiplies 2 polynomial matrices A and B, where each matrix entry is a polynomial, e.g. in z^-1.
	#Those polynomial entries are in the 3rd dimension 
	#The third dimension can also be interpreted as containing the (2D) coefficient matrices for each 
	#exponent of z^-1. 
	#Result is C=A*B;

	import numpy as np
	from scipy import sparse

	[NAx,NAy,NAz]=A.shape;
	[NBx,NBy,NBz]=B.shape;

	#Degree +1 of resulting polynomial, with NAz-1 and NBz-1 beeing the degree of the input  polynomials:
	Deg=NAz+NBz-1;

	C=np.zeros((NAx,NBy,Deg));

	for n in range(Deg):
	  for m in range(n+1):
	    if ((n-m)<NAz and m<NBz):
	      C[:,:,n]=C[:,:,n]+ A[:,:,(n-m)].dot(B[:,:,m])
	      #sparse version:
	      #C[:,:,n]=C[:,:,n]+ (sparse.csr_matrix(A[:,:,(n-m)]).dot(sparse.csr_matrix(B[:,:,m]))).todense()
	return C

#The DCT4 transform:
def DCT4(samples):
   #use a DCT3 to implement a DCT4:
   samplesup=np.zeros(2*N)
   #upsample signal:
   samplesup[1::2]=samples
   y=spfft.dct(samplesup,type=3,norm='ortho')*np.sqrt(2)#/2
   return y[0:N]

def x2polyphase(x,N):
	#xp=x2polyphase(x);
	#Converts input signal x (a row vector) into a polphase row vector
	#For blocks of length N
	#For 3D polyphase representation (exponents of z in the third matrix/tensor dimension)
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Dec-2-2015

	#Number of blocks in the signal:
	L=int(np.floor(len(x)/N))

	xp=np.zeros((1,N,L));

	for m in range(L):
	  xp[0,:,m]=x[m*N: (m*N+N)];
	
	return xp

def polyphase2x(xp):
	#xp=polyphase2x(xp,N);
	#Converts polyphase input signal xp (a row vector) into a contiguous row vector
	#For blocks of length N
	#For 3D polyphase representation (exponents of z in the third matrix/tensor dimension)
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Aug-24-11

	#Number of blocks in the signal:
	[r,N,b]=xp.shape;
	L=b;

	x=np.zeros(N*L);

	for m in range(L):
	  #print x[(m*N):((m+1)*N)].shape, xp[0,:,m].shape
	  x[(m*N):(m+1)*N]=xp[0,:,m];

	return x

def analysisqmf(x,qmfwin,N):
	#QMF Analysis filter bank
	#usage: y=analysisqmf(x,qmfwin,N);
	#with x: input signal (row vector)
	#qmfwin: qmf window
	#N: number of subbands
	#y: output, subband signals as a polphase vector,
	#each column is one subband signal,
	#the third dimension corresponds to the blocks of the signal
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Dec 2015

	#generation of input polyphase vector:
	xp=x2polyphase(x,N);

	#Analysis polphase matrix:
	Fa=ha2Fa3d(qmfwin,N);

	#Analysis filter bank:
	#Sparse mult. with folding mat:
	y=polmatmult(xp,Fa);
        #Transform:
        y=polmatmult(y,DCToMatrix(N))
	return y

def analysisqmf_realtime(xrt,Fa,N):
	#computes the QMF subband samples for each real time input block xrt. 
	#Conducts an implicit polynomial multiplication with folding matrix Fa of 
	#the polyphase matrix of the QMF filter bank, using
	#internal memory of the past input blocks.

	#from scipy import sparse
        global blockmemory
	global overlap
	#push down old blocks:
	blockmemory[0:(overlap-1),:]=blockmemory[1:(overlap),:]
	#write new block into top of memory stack:
	blockmemory[overlap-1,:]=xrt;
    	y=np.zeros((1,N));
        #print "Fa.shape =", Fa.shape
	for m in range(overlap):
   	   y+=np.dot(np.array([blockmemory[overlap-1-m,:]]), Fa[:,:,m])
	   #y+= (sparse.csr_matrix(blockmemory[overlap-1-m,:]).dot(sparse.csr_matrix(Fa[:,:,m]))).todense()
	#fast DCT4:
	y=DCT4(y)
	return y

def synthesisqmf_realtime(y,Fs,N):
	#computes the inverse QMF for each subband block y, 
	#conducts an implicit polynomial multiplication with synthesis folding matrix Fs 
	#of the synthesis polyphase matrix of the QMF filter bank, using
	#internal memory for future output blocks.

	#from scipy import sparse
        global blockmemorysyn
	global overlap
	#print "overlap= ", overlap
	#push down old blocks:
	blockmemorysyn[0:(overlap-1),:]=blockmemorysyn[1:(overlap),:]
	blockmemorysyn[overlap-1,:]=np.zeros((1,N)) #avoid leaving previous values in top of memory.
	#print "memory content synthesis: ", np.sum(np.abs(blockmemorysyn))
        #print "Fs.shape =", Fs.shape
	#print "y.shape= ", y.shape
	#Overlap-add after fast (inverse) DCT4::
	for m in range(overlap):
   	   blockmemorysyn[m,:]+=np.dot(DCT4(y), Fs[:,:,m])
	   #y+= (sparse.csr_matrix(blockmemory[overlap-1-m,:]).dot(sparse.csr_matrix(Fa[:,:,m]))).todense()
   	xrek=blockmemorysyn[0,:]
	
	
	return xrek

def synthesisqmf(y,qmfwin,N):
	#QMF Synthesis filter bank
	#usage: xrek=synthesisqmf(y,qmfwin,N);
	#with y: subband signals (polyphase row vector, 3d tensor in Matlab/Octave)
	#each column is one subband signal,
	#the third dimension corresponds to the blocks of the signal
	#qmfwin: qmf window
	#N: number of subbands
	#xrek: output, rekonstructed signal 
	#Gerald Schuller
	#shl@idmt.fhg.de
	#Dec. 2015

	#generation of input polyphase vector:

	#Analysis polphase matrix:
	Fs=hs2Fs3d(qmfwin,N);

	#Analysis filter bank:
	#Inverse transform:
	xrekp=polmatmult(y,DCToMatrix(N))
        #print xrekp.shape
	#Sparse mult with folding mat.:
	xrekp=polmatmult(xrekp,Fs);
	#print xrekp.shape

	#polyphase to contiguous vector:
	xrek=polyphase2x(xrekp);
	return xrek


def qmfrt_example():
	from scipy import sparse
	import matplotlib.pyplot as plt
        import time

	#QMF Test:
	global N  #number of subbands
   	global overlap 
	qmfwin=np.loadtxt('qmf1024_8x.mat');
	qmfwin=np.hstack((qmfwin,np.flipud(qmfwin)))
	#qmfwin=np.loadtxt('qmf.dat');
	print "qmfwin.shape= ", qmfwin.shape
        
	plt.plot(qmfwin)
	plt.show(block=False)
	#Testing Analysis:
	#Input signal (can be live from the sound card):
	#x=np.random.random((10*N))-0.5;
	x=np.sin(2*np.pi/N*10*np.arange(10*N))
	x=np.zeros((10*N))
	x[10]=1
	#initialize subband matrix:
	y=np.zeros((10,N));
       
	start=time.time()
	#Analysis polphase matrix:
	print "Compute analysis Folding matrix:"
	#Fao=ha2Fa3d(qmfwin,N);
	Fa=ha2Fa3d_fast(qmfwin,N) 
	time1=time.time()
        print "Time for Fa: ", time1-start
	#mult vector with full matrix:
	xtest=np.random.rand(N);
	startmultfull=time.time()
	testout=np.dot(xtest,Fa[:,:,4])
	endtime=time.time()
	print "time for full mult: ", endtime-startmultfull

	sparsemat=sparse.csr_matrix(Fa[:,:,4])
	sparsex=sparse.csr_matrix(xtest)
	startmultsparse=time.time()
	testout=sparsex.dot(sparsemat)
	endtime=time.time()
	print "time for sparse mult: ", endtime-startmultsparse	
	
	#plt.figure()
	#plt.imshow(np.sum(np.abs(Fa), axis=2))
	#plt.show(block=False)
  	print "Fa.shape= ", Fa.shape	   
	#print "Fa=", Fa[N-1,N/2-1,:]
     	#print "sum of diff:", np.sum(np.abs(Fa-Fao))
        #print "Fa.shape =", Fa.shape
        
	print "Start QMF:"
	for m in range(10):
	   y[m,:]=analysisqmf_realtime(x[m*N:(m+1)*N],Fa,N)
	time2=time.time()
	print "Time for filtering: ", time2-time1
	
        print "y.shape= \n", y.shape

  	#xrek=synthesisqmf(y,qmfwin,N)
  	plt.figure()
	#plt.plot(y.T)
	plt.imshow(np.abs(y[:,0:300]))
	plt.show(block=False)

	#Testing synthesis
	#Synthesis folding matrix:
	Fs=hs2Fs3d_fast(qmfwin,N)
	#plt.figure()
	#plt.imshow(np.sum(np.abs(Fa), axis=2))
	#plt.show(block=False)
	#Fso=hs2Fs3d(qmfwin,N)
	#print "sum of diff of Fs:", np.sum(np.abs(Fs-Fso))
	#print "Fs.shape after hs2Fs: ", Fs.shape
	#Test synthesis impulse response of lowest filter:
	#y[:,30:]=np.zeros((10,1024-30))
	"""
	y=np.zeros((10,N))
	y[2:,19]=6.0*np.ones((8));
	y[2:,20]=-6.0*np.ones((8));
	"""
	plt.figure()
	plt.plot(y.T)
	#plt.imshow(y[:,0:40])
	plt.show(block=False)

	print "y.shape= \n", y.shape
	xrek=np.zeros((10*N));
	for m in range(10):
	   xrek[m*N:(m+1)*N]=synthesisqmf_realtime(y[m,:],Fs,N)
	#clear plot figure:
	#plt.clf()
	plt.figure()
	plt.plot(xrek)
	plt.show()
	#Observe: Since QMF is non PR, a reconstructed pulse has small pre- and post-pulses!!! 

if __name__ == '__main__':

        import IOMethods as io
	import matplotlib.pyplot as plt
	import wave
	import struct

	#qmfrt_example()
	
	#global N  #number of subbands
   	#global overlap 
	qmfwin=np.loadtxt('qmf1024_8x.mat');
	qmfwin=np.hstack((qmfwin,np.flipud(qmfwin)))
	plt.plot(qmfwin)
	plt.show(block=False)
	plt.figure()
	#Analysis Folding matrix:
	Fa=ha2Fa3d_fast(qmfwin,N)
	#Synthesis Folding matrix: 
	Fs=hs2Fs3d_fast(qmfwin,N)

	#Open sound file to read:
   	wf=wave.open('mixed.wav','rb');
   	nchan=wf.getnchannels();
   	bytes=wf.getsampwidth();
   	rate=wf.getframerate();
   	length=wf.getnframes();
   	print("Number of channels: ", nchan);
   	print("Number of bytes per sample:", bytes);
   	print("Sampling rate: ", rate);
   	print("Number of samples:", length);

	#open sound file to write the reconstruced sound:
	wfw = wave.open('mixedrek.wav', 'wb')
   	wfw.setnchannels(1)
   	wfw.setsampwidth(2)
   	wfw.setframerate(rate)

	print "Start QMF:"
	#Process the audio signal block-wise (N samples) from file and into file:
	for m in range(length/N):
	   print "Block number: ", m
	   #Analysis:
	   data=wf.readframes(N);
	   x = (struct.unpack( 'h' * N, data ));
	   y=analysisqmf_realtime(x,Fa,N)
	   plt.plot(y)
	   plt.show(block=False)

	   #Synthesis:
	   xrek=synthesisqmf_realtime(y,Fs,N)
	   #write with 2 Bytes per sample:
	   data=struct.pack( 'h' * N, *xrek )
	   wfw.writeframes(data)
	wf.close()   	
	wfw.close()
	

