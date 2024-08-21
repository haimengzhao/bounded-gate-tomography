import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import scienceplots
plt.style.use(['science','no-latex'])

# n = 10000
# focus_n = 4
# gate_set_size = 2
# sample_size_max = 80
# n_run = 100000
# G_max = 10
# G_list = range(1, G_max+1)
# sample_size_list = np.arange(1, sample_size_max+1)

# if focus_n is not None:
#     fid_list = np.load(f"n={n}_G={G_max}_nrun={n_run}_focus.npy")
# else:
#     fid_list = np.load(f"n={n}_G={G_max}_nrun={n_run}.npy")

levels = np.linspace(0, 1, 100)

G_max = 10
sample_size_max = 80
fid_list = np.load(f"n=10000_G=10_nrun=100000_focus.npy")
G_list = range(1, G_max+1)
sample_size_list = np.arange(1, sample_size_max+1)

fid_mean = np.mean(fid_list, axis=0)
fid_med = np.quantile(fid_list, 0.5, axis=0)
fid_std = fid_list.std(axis=0)

def median_of_mean(x, k):
    shape = x.shape
    return np.median(x.reshape(k, -1, *shape[1:]).mean(axis=0), axis=0)

fid_med_mean = median_of_mean(fid_list, 10)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3))
ax1, ax2 = axs

thres_list = [0.95, 0.9, 0.8]

for thres in thres_list:
    ind = np.array(np.argmax(fid_mean>thres, axis=-1))
    ax1.plot(G_list, sample_size_list[ind], linestyle='-', marker='o', label=f"$F={thres}$")

cf1 = ax1.contourf(G_list, sample_size_list, fid_mean.T, levels=levels, vmin=0, vmax=1)

ind = np.array(np.argmax(fid_med>0.999, axis=-1))
ax1.plot(G_list, sample_size_list[ind], linestyle='--', marker='o', label="$F_{med}=0.999$")

ax1.set_ylim(None, 50)
ax1.set_xlabel("Gate Size $G$")
ax1.set_ylabel("Sample Size $N$")
ax1.legend()

G_max = 6
sample_size_max = 60
fid_list = np.load(f"n=10000_G=6_nrun=25000_shallow.npy")
G_list = range(1, G_max+1)
sample_size_list = np.arange(1, sample_size_max+1)

fid_mean = np.mean(fid_list, axis=0)
fid_med = np.quantile(fid_list, 0.5, axis=0)
fid_std = fid_list.std(axis=0)

for thres in thres_list:
    ind = np.array(np.argmax(fid_mean>thres, axis=-1))
    ax2.plot(G_list, sample_size_list[ind], linestyle='-', marker='o', label=f"$F={thres}$")

ind = np.array(np.argmax(fid_med>0.999, axis=-1))
ax2.plot(G_list, sample_size_list[ind], linestyle='--', marker='o', label="$F_{med}=0.999$")

cf2 = ax2.contourf(G_list, sample_size_list, fid_mean.T, levels=levels, vmin=0, vmax=1)

ax2.set_ylim(None, 50)
ax2.set_xlabel("Gate Size $G$")
ax2.set_ylabel("Sample Size $N$")
ax2.legend()

plt.tight_layout()
plt.colorbar(cf1, ax=axs, ticks=np.arange(0, 1.1, 0.2))

plt.savefig(f"numerics.pdf")

