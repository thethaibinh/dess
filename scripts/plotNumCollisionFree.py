import numpy as np
import json
import matplotlib.pyplot as plt

fig = plt.figure()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

try:
    with open("../data/DepthBasedNano" + ".json") as f:
        data = json.load(f)
        compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
        DbAvgCollisionFreeTraj = np.array(data["DbAvgCollisionFreeTraj"], dtype=np.double)
        plt.plot(compTimeMs, DbAvgCollisionFreeTraj, '.r-', label='Jetson Nano Depth-based sampling (our)')
        avgCollisionFreeTraj = np.array(data["AvgCollisionFreeTraj"], dtype=np.double)
        plt.plot(compTimeMs, avgCollisionFreeTraj, '.r-.', label='Jetson Nano Uniform sampling (RAPPIDS)')
except Exception:
    print(' file not found.')

try:
    with open("../data/DepthBasedi7" + ".json") as f:
        data = json.load(f)
        compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
        DbAvgCollisionFreeTraj = np.array(data["DbAvgCollisionFreeTraj"], dtype=np.double)
        plt.plot(compTimeMs, DbAvgCollisionFreeTraj, '.b-', label='i7 Depth-based sampling (our)')
        avgCollisionFreeTraj = np.array(data["AvgCollisionFreeTraj"], dtype=np.double)
        plt.plot(compTimeMs, avgCollisionFreeTraj, '.b-.', label='i7 Uniform sampling (RAPPIDS)')
except Exception:
    print(' file not found.')

try:
    with open("../data/DepthBasedYour" + ".json") as f:
        data = json.load(f)
        compTimeMs = np.array(data["CompTime"], dtype=np.double) * 1e3
        DbAvgCollisionFreeTraj = np.array(data["DbAvgCollisionFreeTraj"], dtype=np.double)
        plt.plot(compTimeMs, DbAvgCollisionFreeTraj, '.k-', label='Your Depth-based sampling (our)')
        avgCollisionFreeTraj = np.array(data["AvgCollisionFreeTraj"], dtype=np.double)
        plt.plot(compTimeMs, avgCollisionFreeTraj, '.k-.', label='Your Uniform sampling (RAPPIDS)')
except Exception:
    print('You can run your experiments and then get back here.')

plt.yscale('linear')
plt.grid()
plt.legend(bbox_to_anchor=(0,1.2,1,0.2), loc="upper center", ncol=1, markerscale=1, fontsize=12, handlelength=3)
plt.xlabel('Allocated Computation Time [ms]', fontsize=14)
plt.ylabel('Number of Collision-free Trajectories Found', fontsize=14)

plt.show()
