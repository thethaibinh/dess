import numpy as np
import json
import matplotlib.pyplot as plt

fig = plt.figure()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

try:
    with open("../data/DepthBasedi7" + ".json") as f:
        data = json.load(f)
        compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
        avgTrajGen = np.array(data["AvgTrajGen"], dtype=np.double)
        plt.plot(compTimeMs, avgTrajGen, '.b-.', label='i7 Uniform sampling (RAPPIDS)')
        DbAvgTrajGen = np.array(data["DbAvgTrajGen"], dtype=np.double)
        plt.plot(compTimeMs, DbAvgTrajGen, '.b-', label='i7 Depth-based sampling (our)')
except Exception:
    print(' file not found.')

try:
    with open("../data/DepthBasedNano" + ".json") as f:
        data = json.load(f)
        compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
        avgTrajGen = np.array(data["AvgTrajGen"], dtype=np.double)
        plt.plot(compTimeMs, avgTrajGen, '.r-.', label='Jetson Nano Uniform sampling (RAPPIDS)')
        DbAvgTrajGen = np.array(data["DbAvgTrajGen"], dtype=np.double)
        plt.plot(compTimeMs, DbAvgTrajGen, '.r-', label='Jetson Nano Depth-based sampling (our)')
except Exception:
    print(' file not found.')

try:
    with open("../data/DepthBasedYour" + ".json") as f:
        data = json.load(f)
        compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
        avgTrajGen = np.array(data["AvgTrajGen"], dtype=np.double)
        plt.plot(compTimeMs, avgTrajGen, '.k-.', label='Your Uniform sampling (RAPPIDS)')
        DbAvgTrajGen = np.array(data["DbAvgTrajGen"], dtype=np.double)
        plt.plot(compTimeMs, DbAvgTrajGen, '.k-', label='Your Depth-based sampling (our)')
except Exception:
    print('You can run your experiments and then get back here.')

plt.yscale("log")
plt.grid()
plt.legend(markerscale=1)
plt.xlabel('Allocated Computation Time [ms]')
plt.ylabel('Number of Trajectories Evaluated')

plt.show()