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

modelA = GradientBoostingClassifier(n_estimators=50, random_state=42)
modelA.fit(X, y)

modelB = RandomForestClassifier(n_estimators=50, random_state=42)
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
    
    
def pad_header(path):
    binary = lief.parse(path)
    binary.optional_header.sizeof_headers += 512

    builder = lief.PE.Builder(binary)
    builder.build()
    return bytes(builder.get_build())
    
def rename_section(path):
    binary = lief.parse(path)
    if len(binary.sections) > 0:
        binary.sections[0].name = ".abcd"

    builder = lief.PE.Builder(binary)
    builder.build()
    return bytes(builder.get_build())


def feature_groups(feat):
    return {
        "header": feat[:100],
        "section": feat[100:500],
        "imports": feat[500:1000],
        "histogram": feat[1000:2000],
        "other": feat[2000:]
    }
    




count = 0
change_A = 0
change_B = 0
    

THRESH = 0.02

# -------- Step 12: Size Sweep --------

sizes = list(range(0, 20000, 500))

path = os.path.join(sample_dir, files[25])

with open(path, "rb") as f:
    orig = f.read()

feat_orig = extractor.extract(orig)
X_orig = np.array(feat_orig).reshape(1, -1)

b_before = modelB.predict_proba(X_orig)[0][1]

print("Base B:", b_before)

important_idx = [0, 1, 2, 73, 140, 205, 256, 257]

print("\n--- feature[0] vs prediction ---")

results = []

for n in range(0, 20000, 200):
    mut = append_bytes(path, n)

    feat_mut = extractor.extract(mut)
    X_mut = np.array(feat_mut).reshape(1, -1)

    b_after = modelB.predict_proba(X_mut)[0][1]

    f0 = feat_mut[0]
    f257 = feat_mut[257]

    results.append((n, f0, f257, b_after))

    print(f"n={n:5d} | f0={f0:.2f} | f257={f257:.2f} | B={b_after:.4f}")
