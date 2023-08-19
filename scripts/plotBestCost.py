import numpy as np
import json
import matplotlib.pyplot as plt

fig = plt.figure()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

try:
    with open("../data/DepthBasedi7" + ".json") as f:
        data = json.load(f)
        compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
        bestCost = np.array(data["BestCost"], dtype=np.double)
        plt.plot(compTimeMs, bestCost, '.b-.', label='i7 Uniform sampling (RAPPIDS)')
        bestCostDb = np.array(data["DbBestCost"], dtype=np.double)
        plt.plot(compTimeMs, bestCostDb, '.b-', label='i7 Depth-based sampling (our)')

except Exception:
    print(' file not found.')

try:
    with open("../data/DepthBasedNano" + ".json") as f:
        data = json.load(f)
        compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
        bestCost = np.array(data["BestCost"], dtype=np.double)
        plt.plot(compTimeMs, bestCost, '.r-.', label='Jetson Nano Uniform sampling (RAPPIDS)')
        bestCostDb = np.array(data["DbBestCost"], dtype=np.double)
        plt.plot(compTimeMs, bestCostDb, '.r-', label='Jetson Nano Depth-based sampling (our)')
except Exception:
    print(' file not found.')

try:
    with open("../data/DepthBasedYour" + ".json") as f:
        data = json.load(f)
        compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
        bestCost = np.array(data["BestCost"], dtype=np.double)
        plt.plot(compTimeMs, bestCost, '.k-.', label='Your Uniform sampling (RAPPIDS)')
        bestCostDb = np.array(data["DbBestCost"], dtype=np.double)
        plt.plot(compTimeMs, bestCostDb, '.k-', label='Your Depth-based sampling (our)')
except Exception:
    print('You can run your experiments and then get back here.')

plt.yscale('linear')
plt.grid()
plt.legend(markerscale=0)
plt.xlabel('Allocated Computation Time [ms]')
plt.ylabel('Best Cost')

plt.show()
