#Program to display the qmf or MDCT, MDST spectrum from the pickle .bin file.
#example: python dispbinspectrum.py mixed_mdct_y.bin
#Gerald Schuller, gerald.schuller@tu-ilmenau.de, May 2016

import numpy as np



if __name__ == '__main__':

        
	import cPickle as pickle
	import sys
        import matplotlib.pyplot as plt

	
	#qmfrt_example()
	
	#global N  #number of subbands
   	#global overlap 
	if len(sys.argv) < 2:
	    print("Need 1 argument.\n\nUsage: %s infile.bin" % sys.argv[0])
 	    sys.exit(-1)
	#Open pickle file to read:	
	wf=open(sys.argv[1], 'r')

	#Process the audio signal block-wise (N samples) from file and into file:
	spec=[]
	m=0
	while True:
	   print "Block number: ", m
	   m+=1
	   #Synthesis:
	   try:
	      y=pickle.load(wf)
	      
	   except (EOFError):
              break
	   if (spec==[]):
	      spec=y
	   else:
	      spec=np.vstack((spec,y))
	      #spec=np.append(spec,y,axis=1)
	print("Dim of spec: ", spec.shape)
	plt.imshow(np.abs(spec))	   
	wf.close()   	
	plt.show()
	

