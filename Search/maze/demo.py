'''
Author: fengsc
Date: 2022-03-16 12:20:07
LastEditTime: 2022-03-16 18:43:03
'''
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def data_gen():#返回控制参数
    for cnt in itertools.count():
        t = cnt / 10
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)# *每十帧返回一组值


def init():# 第一帧前调用，此处用来调整尺度
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    line.set_data(xdata, ydata)
    return line,
# !函数可访问外部的变量，以下变量，句柄为各个函数共享，函数内可以修改但不能赋值，除非声明global
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)#定义为元组因为作为返回值需要可迭代
ax.grid()
xdata, ydata = [], []


def run(data):# !接受从gen_function()传来的数据，其默认是itertools.count()
    # update the data
    t, y = data#获取一组数据加入data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)# *修改x轴范围
        ax.figure.canvas.draw()# *刷新画布
    line.set_data(xdata, ydata)

    return line,# 返回画布句柄(元组)

ani = animation.FuncAnimation(fig, run, data_gen, interval=10, init_func=init)
ani.save("demo1.gif")
plt.show()