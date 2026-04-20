import sys
sys.path.append("/media/radon/Data1/gym-malware-master/gym_malware/envs/utils")
import pefeatures
import numpy as np
np.int = int
import os
import lief
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

sample_dir = "./samples"

files = os.listdir(sample_dir)

extractor = pefeatures.PEFeatureExtractor()

X = []
for f in files[:20]:
    with open(os.path.join(sample_dir, f), "rb") as fp:
        X.append(extractor.extract(fp.read()))

X = np.array(X)

# fake labels (balanced)
y = [0]*(len(X)//2) + [1]*(len(X) - len(X)//2)

modelA = GradientBoostingClassifier(n_estimators=50)
modelA.fit(X, y)

modelB = RandomForestClassifier(n_estimators=50)
modelB.fit(X, y)

#pX = modelA.predict(X[:2])
#pY = modelB.predict(X[:2])

#print(pX,'\n',pY)




def append_bytes(path, n=100):
    binary = lief.parse(path)
    builder = lief.PE.Builder(binary)
    builder.build()
    data=bytearray(builder.get_build())
    data += b'\x00' * n
    return bytes(data)
    
    
def add_import(path):
    binary = lief.parse(path)
    binary.add_library("kernel32.dll")

    builder = lief.PE.Builder(binary)
    builder.build()
    return bytes(builder.get_build())
    

    
for f in files[:10]:
    path = os.path.join(sample_dir, f)

    with open(path, "rb") as fp:
        orig = fp.read()

    feat_orig = extractor.extract(orig)
    X_orig = np.array(feat_orig).reshape(1, -1)

    mut = append_bytes(path, 50000)   # keep this one fixed for now
    feat_mut = extractor.extract(mut)
    X_mut = np.array(feat_mut).reshape(1, -1)

    a_before = modelA.predict_proba(X_orig)[0][1]
    a_after  = modelA.predict_proba(X_mut)[0][1]

    b_before = modelB.predict_proba(X_orig)[0][1]
    b_after  = modelB.predict_proba(X_mut)[0][1]

    print(f, "| A:", a_before, "→", a_after,
              "| B:", b_before, "→", b_after)
