import sys
import numpy as np
np.int = int

# fix old sklearn pickle refs (harmless to keep)
import sklearn.tree
import sklearn.tree._classes
sys.modules['sklearn.tree.tree'] = sklearn.tree._classes

import lief
import random
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# import pefeatures directly (bypass broken package init)
sys.path.append("/media/radon/Data1/gym-malware-master/gym_malware/envs/utils")
import pefeatures

# ---- init extractor ----
extractor = pefeatures.PEFeatureExtractor()

# ---- load binary ----
binary = lief.parse("test.exe")
if binary is None:
    print("Failed to parse PE file")
    exit()

builder = lief.PE.Builder(binary)
builder.build()
original_data = bytearray(builder.get_build())

# ---- build training data ----
X_train = []
y_train = []

for i in range(50):
    temp = bytearray(original_data)

    # random mutation strength
    noise = random.randint(0, 500)
    temp += b'\x00' * noise

    try:
        f = extractor.extract(bytes(temp))
        X_train.append(f)

        # random labels (avoid trivial model)
        y_train.append(random.randint(0, 1))

    except:
        continue

X_train = np.array(X_train)
y_train = np.array(y_train)

print("Training samples:", X_train.shape)

# ---- train models (YOU FORGOT THIS) ----
modelA = GradientBoostingClassifier(n_estimators=50)
modelA.fit(X_train, y_train)

modelB = RandomForestClassifier(n_estimators=50)
modelB.fit(X_train, y_train)

# ---- loop ----
"""
cost = 0

for i in range(5):
    data = bytearray(original_data)

    # random mutation
    if random.random() < 0.5:
        data += b'\x00' * random.randint(10, 200)
    else:
        for _ in range(10):
            idx = random.randint(0, len(data)-1)
            data[idx] = random.randint(0, 255)

    cost += 1

    # ---- extract features ----
    try:
        features = extractor.extract(bytes(data))
        X = np.array(features).reshape(1, -1)
    except Exception as e:
        print(f"Iter {i} | Feature extraction failed: {e}")
        continue

    # ---- evaluate ----
    scoreA = modelA.predict(X)
    scoreB = modelB.predict(X)

    evadeA = int(scoreA[0] == 0)
    evadeB = int(scoreB[0] == 0)

    print(f"Iter {i} | EvadeA={evadeA} | EvadeB={evadeB} | Cost={cost} | Size={len(data)}")

print("Done")
"""
successA = 0
successB = 0

agree = 0

for i in range(20):
    data = bytearray(original_data)

    if random.random() < 0.5:
        mutation = "append"
        data += b'\x00' * random.randint(10, 200)
    else:
        mutation = "flip"
        for _ in range(10):
            idx = random.randint(0, len(data)-1)
            data[idx] = random.randint(0, 255)

    cost = i + 1

    try:
        features = extractor.extract(bytes(data))
        X = np.array(features).reshape(1, -1)
    except:
        continue

    scoreA = modelA.predict(X)
    scoreB = modelB.predict(X)

    evadeA = int(scoreA[0] == 0)
    evadeB = int(scoreB[0] == 0)
    
    if evadeA == evadeB:
        agree+=1
    
    successA += evadeA
    successB += evadeB

    print(f"Iter {i} | {mutation} | A={evadeA} B={evadeB}")

# ---- final metrics ----
print("\n--- RESULTS ---")
print("Evasion Rate A:", successA / 20)
print("Evasion Rate B:", successB / 20)
print("Efficiency A:", successA / (20))   # simple version
print("Agreement Rate:", agree / 20)
