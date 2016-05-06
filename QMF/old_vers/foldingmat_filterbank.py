import numpy as np
import scipy.fftpack as spfft

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

	Pa=ha2Pa3d(ha,N);
	Fa=polmatmult(Pa,DCToMatrix(N))
        #round zeroth polyphase component to 7 decimals after point:
        Fa=np.around(Fa,8)

	return Fa

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
   y=spfft.dct(samplesup,type=3)/2
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

def test():
	from scipy import sparse
	import matplotlib.pyplot as plt
	import IOMethods as io
        #ha=np.arange(1,9)
	ha=np.sin(np.pi/8*(np.arange(8)+0.5))
        print "ha= ", ha
	Fa=ha2Fa3d(ha,4)
	Fs=hs2Fs3d(ha,4)
	# convert to compressed sparse row matrix: 
        sFa0 = sparse.csr_matrix(Fa[:,:,0])
        sFa1 = sparse.csr_matrix(Fa[:,:,1]) 
	#umkehrung: sFa0.todense()
        #List of sparse 2d matrices, to obtain sparse 3d tensor:
        #sFa=[sFa0, sFa1];
        sFa=[sFa0]
	sFa.append(sFa1)

        [NAx,NAy,NAz]=Fa.shape;
	np.set_printoptions(suppress=True)
        print "Fa= \n"
 	for n in range(NAz):
	   print "Fa ",n,':\n', Fa[:,:,n] 
	print "Fs= \n"
 	for n in range(NAz):
	   print "Fs ",n,':\n', Fs[:,:,n] 

	print "test perfect reconstruction: Fa*Fs=Ident: \n"
        pr=polmatmult(Fa,Fs)
	[NAx,NAy,NAz]=pr.shape;
	for n in range(NAz):
	   print "Prod ",n,':\n', pr[:,:,n] 
        
        #print sFa0
	#print sparse.issparse(sFa0)
        #print sparse.issparse(Fa)
        #pr=Fa[:,:,0].dot(Fa[:,:,0])
	#print '\n', np.dot(Fa[:,:,0],Fa[:,:,0])
        #print '\n', pr
	#print sFa[0], '\n'
        #print sFa[1]
        #print sparse.issparse(sFa[0])
        #print(sFa[0].shape)
        #print len(sFa)

	#QMF Test:
	N = 64
	qmfwin=np.loadtxt('qmf.dat');
	mdctwin = np.sin(np.pi/(2*N) * (np.arange(2*N) + 0.5))

	win = qmfwin
	plt.plot(win)
	
	x=np.arange(1024);
	print(x.shape)
	N=64 #number of subbands

	y=analysisqmf(x,win,N)
	#Test synthesis impulse response of lowest filter:
	#y=np.zeros((1,64,10))
	#y[0,0,0]=1;
        print "y.shape= \n", y.shape

  	xrek=synthesisqmf(y,win,N)
  	plt.figure()
	plt.plot(xrek)
	plt.show()

if __name__ == '__main__':

    import IOMethods as io
    import matplotlib.pyplot as plt
	#test()

    #QMF Example:
    N=1024 #Number of subbands
	#qmfwin=np.loadtxt('qmf.dat');
    #mdctwin = np.sin(np.pi/(2*N) * (np.arange(2*N) + 0.5))
    qmfwin = np.loadtxt('qmf1024.mat')

    win = qmfwin

    plt.plot(win)
    plt.title('Filter Bank Window')
    plt.xlabel('Sample at original sampling rate')
    x,fs=io.AudioIO.wavRead('Dido44.wav', True)

    x1 = x[0 : (15 * fs)]
    x2 = x[(15 * fs): (30 * fs)]
    x1 = x1[:, 0]
    x2 = x2[:, 0]

    #make x 1-dimensional:
    x=x[:,0];
    print "x.shape", x.shape


    #y=analysisqmf(x,win,N) #in y are the subbands over time
    y1=analysisqmf(x1,win,N) #in y are the subbands over time
    y2=analysisqmf(x2,win,N) #in y are the subbands over time

    print "y.shape= ", y.shape
    #show the subbands as spectrogram:
    plt.figure()
    plt.imshow(np.abs(y[0,:,200:250])/N, aspect='auto', interpolation='nearest', origin='lower')
    plt.title('Part of filter bank subbands')
    plt.xlabel('time (sample) after downsampling (block)')
    plt.ylabel('Frequency')

    #xrek=synthesisqmf(y,win,N)
    xrek1=synthesisqmf(y1,win,N)
    xrek2=synthesisqmf(y2,win,N)
    np.hstack((xrek1, xrek2))
    plt.figure()
    plt.plot(xrek)
    plt.title('Reconstructed signal')
    plt.xlabel('Sample at original sampling rate')
    plt.show()

