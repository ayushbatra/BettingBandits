import numpy as np
import seaborn as sns
import pandas as pd
from bandits import *
import itertools
from utils import *

# plot Figure 1

bandit = DssatBandit(normalize=False)
data = pd.DataFrame()
data["Optimal distribution"] = bandit.samples[bandit.best_arm][:]
X = []
for arm in bandit.samples:
    m = np.floor(np.mean(arm))
    if np.abs(m - 3013) < 2:
        data["Distribution 1"] = arm[:]

for arm in bandit.samples:
    m = np.floor(np.mean(arm))
    if np.abs(m - 3317) < 2:
        data["Distribution 2"] = arm[:]

for arm in bandit.samples:
    m = np.floor(np.mean(arm))
    if np.abs(m - 3396) < 2:
        data["Distribution 3"] = arm[:]

g = sns.displot(data, kind="kde", height=10, aspect=1.8)
ax_sns = g.axes[0, 0]
plt.close('all')

dpi = 96
scale_x = 1080
scale_y = 566
fig = plt.figure(figsize=(scale_x / dpi, scale_y / dpi), dpi=dpi)
ax = fig.add_subplot()
for pos in ["top", "right"]:
    ax.spines[pos].set_visible(False)


y_title = "density"
x_title = "value"
plt.ylabel(y_title, fontsize=18, fontweight='medium', fontname="Noto Serif")
plt.xlabel(x_title, fontsize=18, fontweight='medium', fontname="Noto Serif")
marker = itertools.cycle((',', '+', 'o', '*', 'x', 's', 'v', 'P'))

labels = [f"F$_4$", f"F$_3$", f"F$_2$", "F$_1$ (Opt.)"]
mean_tick = [3630, 3013, 3317, 3396]
b = True
for line, lab, mt in zip(ax_sns.lines[::-1], labels[::-1], mean_tick):
    x, y = line.get_data()
    if b:
        lab = "\n" + lab + f"\n$\mu$ = {mt}"
    else:
        b = True
        lab = lab + f"\n$\mu$ = {mt}"
    p = plt.plot(x, y, label=lab, marker=next(marker), markevery=0.2, linewidth=2.5, markersize=7)
    m = np.argmin(np.abs(mt - np.abs(x)))
    plt.vlines(mt, ymin=0, ymax=y[m], color=p[0].get_color(), linestyle='--')

plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.legend(prop={'size': 14}, loc='upper right')
plt.savefig("figure_1.pdf", dpi=dpi, bbox_inches='tight')
# plt.show()

# Print table 1

dssat = DssatBandit(normalize=False)
means = dssat.means.tolist()
best_arm = dssat.best_arm
best_mean = dssat.best_mean
p = [m / dssat.ub for m in means]
p_star = p.pop(best_arm)
print(f"best mean = {best_mean}")
arms = dssat.arms
arms.pop(best_arm)

Kinf_dssat = [np.abs(-kinf(arm.sample_array, best_mean, upper_bound=dssat.ub).fun) for arm in arms]
Delta_ratio = [k * (dssat.ub**2) / (2 * (best_mean - arm.mean)**2) for k, arm in zip(Kinf_dssat, arms)]
Bernoulli_kl_ratio = [k / (kl_bernoulli(p_arm, p_star)) for k, p_arm in zip(Kinf_dssat, p)]

means = np.delete(means, best_arm)
print(f"means = {means}")
print(f"Bernoulli_kl_ratio = {Bernoulli_kl_ratio}")
print(f"Delta_ratio = {Delta_ratio}")
