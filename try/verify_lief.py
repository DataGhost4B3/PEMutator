"""
import lief
import os

sample_dir = "./samples"

files = os.listdir(sample_dir)

for f in files:
    path = os.path.join(sample_dir, f)
    binary = lief.parse(path)
    print(f, "OK" if binary else "=============!!!!!!!!!!!!!!!!!!!!FAIL!!!!!!!!!!!!===============")
"""

import sys
sys.path.append("/media/radon/Data1/gym-malware-master/gym_malware/envs/utils")
import pefeatures
import numpy as np
np.int = int
import os

sample_dir = "./samples"

files = os.listdir(sample_dir)

extractor = pefeatures.PEFeatureExtractor()

for f in files:
    with open(os.path.join(sample_dir, f), "rb") as fp:
        feat = extractor.extract(fp.read())
        print(f, len(feat))
    
    

