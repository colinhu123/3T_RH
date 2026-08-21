import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


# 找到所有时间步文件
files = sorted(
    glob.glob("data/solution_*.dat")
)


step = 0

def read_file(filename):

    data = pd.read_csv(filename)

    x = data["x"].values
    rho = data["ee"].values

    return (x,rho)

step_l = []
rho_tot = []

for i in range(len(files)):
    step_l.append(i)

    x, rho = read_file(files[i])

    rho_tot.append(rho.sum())

step_l = np.array(step_l)
rho_tot = np.array(rho_tot)
print(rho_tot)

plt.plot(step_l,rho_tot)


plt.show()