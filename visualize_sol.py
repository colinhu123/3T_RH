import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


# 找到所有时间步文件
files = sorted(
    glob.glob("data/solution_*.dat")
)

if len(files) == 0:
    raise RuntimeError("No solution files found")


# 当前时间步
step = 0


fig, ax = plt.subplots(figsize=(8,5))


def read_file(filename):

    data = pd.read_csv(filename)

    x = data["x"].values
    rho = data["rho"].values

    return x, rho



def update():

    global step

    x, rho = read_file(files[step])

    ax.clear()

    ax.plot(
        x,
        rho
    )

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$")

    ax.set_title(
        f"Step {step}/{len(files)-1}\n"
        f"{files[step]}"
    )

    ax.grid(True)

    fig.canvas.draw_idle()



def keyboard(event):

    global step


    # 右方向键：下一步
    if event.key == "right":

        if step < len(files)-1:
            step += 1

        update()


    # 左方向键：上一步
    elif event.key == "left":

        if step > 0:
            step -= 1

        update()


    # q退出
    elif event.key == "q":

        plt.close()



# 注册键盘事件
fig.canvas.mpl_connect(
    "key_press_event",
    keyboard
)


# 初始显示
update()


plt.show()