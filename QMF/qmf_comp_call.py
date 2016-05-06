# -*- coding: utf-8 -*-
"""
    Main method to analyse and synthesize audio files using PQMF, without much python interaction.
    Usage :
        To analyse a wave file :
            1) cd QMF
            2) python qmf_comp_call.py analysis mixed.wav test_analysis.p

        To re-synthesize a a pickled file :
            1) cd QMF (if you are already in, just skip to 2)
            2) python qmf_comp_call.py synthesis test_analysis.p resynth.wav

    Arguments :
            Method    :   (string)  Can be either "analysis" or "synthesis"
            Path#1    :   (string)  String containing the path to the file. If "analysis" method is selected
                                    the path should prompt to an audio file. Else, it should refer the analysed
                                    pickle file.
            Path#2    :   (string)  String containing an arbitrary path and filename. If "analysis" method is selected
                                    the string should contain the path and the filename for the analysis results to be
                                    stored. Else, path and the filename for the resynthesized audio should be denoted.
"""
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import sys
import qmf_realtime_class as qmfc
import cPickle as pickle

def main(argv):
    mode = argv[0]

    if mode == 'analysis' :
        filePathWave = argv[1]
        savePathFile = argv[2]

        # Complex analysis, with external storing using pickle
        qmfc.PQMFAnalysis.analyseNStore(filePathWave, N = 1024, saveStr = savePathFile)

    elif mode == 'synthesis' :

        filePathPickle = argv[1]
        saveWavePath = argv[2]
        # Load complex matrix from the analysis
        qmfc.PQMFSynthesis.loadNRes(filePathPickle, fs = 44100, saveStr = saveWavePath)

    else :
        print('Unrecognised operation! "analysis" & "synthesis" are currently supported...')

if __name__ == "__main__":
    main(sys.argv[1:])