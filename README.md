# ASP

### You have just found Audio Signal Processing (ASP).
It covers a range of utilities for I/O handling and Time-Frequency Decompositions and soon enough audio source separation methods.
Currently supported functionallity :
- WAV/MP3/AAC Reading and Writing
- Time Frequency Methods : MDCT/MDST/PQMF/STFT
- Cepstral Analysis : Uniform Discrete Cepstrum
- Misc Operations : Bark Scaling, W-Disjoint Orthogonality Measure

For code usage, please refer to each class. Examples are given inside method or in the "main()" call.

### Requirements :
- NumPy version   '1.10.4' or later
- SPAMS		   For Matching Pursuit based decompositions (Used in the WDO Experiment)
- SciPy version   '0.17.0' or later (Crucial for avoiding poor reconstruction for the complex PQMF)
- cPickle version '1.71' or later
- pyglet           For audio playback routines

