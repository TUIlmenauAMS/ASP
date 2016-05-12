# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'
"""
    Program to test the MDCT and MDST methods.
    It generates binary files for all the audio files
    located in the test folder.
"""
import os
import subprocess

def main():
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    mdct_analysis_script = os.path.abspath(os.path.join(current_dir, '..', 'MDCT', 'MDCT_realtime_analysis.py'))
    mdct_synthesis_script = os.path.abspath(os.path.join(current_dir, '..', 'MDCT', 'MDCT_realtime_synthesis.py'))
    mdst_analysis_script = os.path.abspath(os.path.join(current_dir, '..', 'MDCT', 'MDST_realtime_analysis.py'))
    mdst_synthesis_script = os.path.abspath(os.path.join(current_dir, '..', 'MDCT', 'MDST_realtime_synthesis.py'))
    
    # For all audio files
    filelist = os.listdir(current_dir)
    for filename in filelist:
        if filename.endswith(".wav"):
            print(filename)
            print('MDCT Analysis')
            call_args = ['python', mdct_analysis_script, os.path.join(current_dir,filename), os.path.join(current_dir,'MDCT_'+(filename[:-3]+'bin'))]
            pr1 = subprocess.Popen(call_args)
            # For waiting until the end of operation
            pr1.communicate()
            print('MDCT Synthesis')
            call_args = ['python', mdct_synthesis_script, os.path.join(current_dir,'MDCT_'+(filename[:-3]+'bin')), os.path.join(current_dir,'rec_mdct_'+filename)]
            pr2 = subprocess.Popen(call_args)
            pr2.communicate()

            print('MDST Analysis')
            call_args = ['python', mdst_analysis_script, os.path.join(current_dir,filename), os.path.join(current_dir,'MDST_'+(filename[:-3]+'bin'))]
            pr3 = subprocess.Popen(call_args)
            # For waiting until the end of operation
            pr3.communicate()
            print('MDST Synthesis')
            call_args = ['python', mdst_synthesis_script, os.path.join(current_dir,'MDST_'+(filename[:-3]+'bin')), os.path.join(current_dir,'rec_mdst_'+filename)]
            pr4 = subprocess.Popen(call_args)
            pr4.communicate()

            
            

if __name__ == "__main__":
    main()
