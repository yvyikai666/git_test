# %%
# %%time
from cProfile import label
from dataclasses import replace
from locale import normalize
from math import *
from time import time
from sympy import *
x = symbols('x')
y = symbols('y')
t = (pi-x) / 2
a = cos(y) * (sin(pi/4) * sin(1/2)*atan(tan(y) / sin(pi/4)) + cos(pi/4)-cos(pi/4)*cos(1/2*atan(tan(y)/ sin(pi/4))))
integrate(a, (y, 0, t))

# %%
import matplotlib.pyplot as plt
label = ['a', 'b', 'c', 'd']
people = [0.83, 0.08, 0.02, 0.07]
explode = [0.1] * 4
color = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
plt.pie(people, explode=explode, colors=color, labels=label, normalize=True)
plt.show()
# %%
fig, axes = plt.subplots(2, 3)
print(axes)
# %%
fig = plt.figure(figsize=(5, 5), dpi=100)
fig.patch.set_color('g')
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax1.patch.set_color('r')
ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
ax2.grid(True)
ax3 = plt.subplot2grid((3, 3), (1, 0), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 1), colspan=2)
ax5 = plt.subplot2grid((3, 3), (1, 1))
plt.tight_layout()

# %%
fig2 = plt.figure(2)
ax1 = fig2.add_subplot(121)
ax2 = fig2.add_subplot(122)
# %%
from matplotlib.lines import Line2D
fig3 = plt.figure(3)
line1 = Line2D([0, 1], [0, 1], transform=fig3.transFigure, figure=fig3, color='r')
line2 = Line2D([0, 1], [1, 0], transform=fig3.transFigure, figure=fig3, color='g')
fig3.lines.extend([line1, line2])
plt.show()
# %%
import numpy as np
fig, ax = plt.subplots()
n, bins, rects = ax.hist(np.random.randn(1000), 50, facecolor='b')
rects[0] is ax.patches[0]
# %%
import numpy as np
import matplotlib.pyplot as plt

def fuc1(x):
    return 0.6 * x + 0.3

def fuc2(x):
    return 0.4 * x * x + 0.1 * x + 0.2

def find_curve_intersects(x, y1, y2):
    d = y1 - y2
    # 找出前一个和后一个
    idx = np.where(d[:-1] * d[1:] <= 0)[0]
    x1, x2 = x[idx], x[idx+1]
    d1, d2 = d[idx], d[idx+1]
    return -d1 * (x2 - x1) / (d2 - d1) + x1

x = np.linspace(-3, 3, 100)
f1 = fuc1(x)
f2 = fuc2(x)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, f1, 'o')
ax.plot(x, f2, 'o')
ax.fill_between(x, f1, f2, where=f1>f2, facecolor='g', alpha=0.5)
plt.show()
# %%三体
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 1.98e-10
M1 = 2e30
mu = 1.2
M2 = mu * M1
L = 1.49e8
om = np.sqrt(G*(M1+M2)/L**3)

dt = 2
t = np.arange(0, 250, 2)

x2 = L / (mu+1)*np.cos(om*t)
y2 = L / (mu+1)*np.sin(om*t)
x1, y1 = -mu * x2, -mu * y2

# 绘图过程
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, autoscale_on=False, 
    xlim=(0.8*L, 0.8*L), ylim=(-0.8*L, 0.8*L))
ax.grid()

line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(i):
    line1.set_data(x1[:i], y1[:i])
    line2.set_data(x2[:i], y2[:i])
    time_text.set_text(f'time = {i*dt:.1f}')
    return line1, line2, time_text

ani = animation.FuncAnimation(fig, animate, 
    range(len(t)), interval=10, blit=True)
ani.save('tri_1.gif', writer='imagemagick')
plt.show()

# %%
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 4.98e-10   #时间单位为天
M1 = 2e30  
mu = 1.2
M2 = mu*M1
L = 1.49e8      #km
om = np.sqrt(G*(M1+M2)/L**3)

dt = 2
t = np.arange(0, 250, dt)

x2 = L/(mu+1)*np.cos(om*t)
y2 = L/(mu+1)*np.sin(om*t)
x1,y1 = -mu*x2, -mu*y2

# 下面为绘图过程
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, autoscale_on=False, 
    xlim=(-0.8*L, 0.8*L), ylim=(-0.8*L, 0.8*L))
ax.grid()

line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
time_template = 'time = %.1f d'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# 这里是绘制动态图
def animate(i):
    line1.set_data(x1[:i],y1[:i])
    line2.set_data(x2[:i],y2[:i])
    time_text.set_text(time_template % (i*dt))
    return line1, line2, time_text

ani = animation.FuncAnimation(fig, animate, range(len(t)),   
        interval=10, blit=True)
ani.save("tri_1.gif",writer='imagemagick')
fig.show()
# %% 
# 其中，mu,G,M1,M2为全局变量
def derivs(state, t):
    dydx = np.zeros_like(state)
    x, vx, y, vy = state
    x2 = L/(mu+1)*np.cos(om*t)
    y2 = L/(mu+1)*np.sin(om*t)
    x1 = -mu*x2
    y1 = -mu*y2
    L1 = np.sqrt((x-x1)**2+(y-y1)**2)**3
    L2 = np.sqrt((x-x2)**2+(y-y2)**2)**3
    dydx[0] = state[1]
    dydx[1] = -G*(M1*(x-x1)/L1+M2*(x-x2)/L2)
    dydx[2] = state[3]
    dydx[3] = -G*(M1*(y-y1)/L1+M2*(y-y2)/L2)
    return dydx
# %%
# 星体等数据可按照上面的代码来写
# 生成时间
dt = 1
t = np.arange(0, 250, dt)

x, y = -L/3, L/3
vx0 = 0
vy0 = 0
state = np.array([x,vx0,y,vy0])

# 微分方程组数值解
x,vx,y,vy = integrate.odeint(derivs, state, t).T
plt.plot(x,y)
plt.show()


# %%
x2 = L/(mu+1)*np.cos(om*t)
y2 = L/(mu+1)*np.sin(om*t)
x1 = -mu*x2
y1 = -mu*y2

def animate(i):
    pt0.set_data(x[i],y[i])
    pt1.set_data(x1[i],y1[i])
    pt2.set_data(x2[i],y2[i])
    line0.set_data(x[:i],y[:i])
    line1.set_data(x1[:i],y1[:i])
    line2.set_data(x2[:i],y2[:i])
    time_text.set_text(time_template % (i*dt))
    return line0, line1, line2, pt0, pt1, pt2, time_text

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, autoscale_on=False, 
    xlim=(-0.8*L, 0.8*L), ylim=(-0.8*L, 0.8*L))
ax.grid()

line0, = ax.plot([], [], lw=2)
line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
pt0, = ax.plot([x[0]],[y[0]] ,marker='o')
pt1, = ax.plot([x1[0]],[y1[0]] ,marker='o')
pt2, = ax.plot([x2[0]],[y2[0]] ,marker='o')

time_template = 'time = %.1f d'
time_text = ax.text(0.05, 0.9, '', 
    transform=ax.transAxes)

ani = animation.FuncAnimation(fig, animate, t,   
        interval=0.1, blit=True)
plt.show()
ani.save("tri_3.gif")
# %%
# 方程组为dp/dt = a*x-b**2/2*x**3, dx/dt = c*p-c**2/2*p**3
import numpy as np
import matplotlib.pyplot as plt

p = [0]
x = [0]  # 给定p，x初始值
a = 2
b = 5
c = 3
t = np.arange(0, 5, 0.001)  # 时间和步长
for i, dt in enumerate(t):
    '''龙格库塔法'''
    k11 = a * x[i] - b ** 2 / 2 * x[i] ** 3
    k21 = c * p[i] - c ** 2 / 2 * p[i] ** 3

    k12 = a * (x[i]+0.01/2) - b ** 2 / 2 * (x[i]+0.01/2) ** 3
    k22 = c * (p[i]+0.01/2) - c ** 2 / 2 * (p[i]+0.01/2) ** 3

    k13 = k12
    k23 = k22

    k14 = a * (x[i] +0.01/2) - b ** 2 / 2 * (x[i]+0.01/2 ** 3)
    k24 = a * (p[i] +0.01/2) - b ** 2 / 2 * (p[i]+0.01/2 ** 3)

    pn = p[i] + 0.001/6 * (k11+2*k12+2*k13+k14)
    xn = x[i] + 0.001/6 * (k21+2*k22+2*k23+k24)
    
    p.append(pn)
    x.append(xn)
# print(r'$\frac{dp}{dx}=ax-b^2/2*x^3$')
plt.plot(t, p[: -1], label='p')
plt.plot(t, x[: -1], label='x')
plt.title(r'$\frac{dp}{dt}=ax-b^2/2x^3&&\frac{dx}{dt}=cp-c^2/p^3$')
plt.legend()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

x = [1]
y = [2]
z = [3]

s = 10
m = 28
r = 8 / 3
h = 0.01
t = np.arange(0, 1, h)

for i, _ in enumerate(t):
    k11 = s * (x[i]-y[i])
    k21 = (m-z[i])-y[i]
    k31 = x[i] * y[i] - r * z[i]

    k12 = s * ((x[i]+1/2*h*k11)-y[i])
    k22 = (m-z[i])-(y[i]+1/2*h*k21)
    k32 = x[i] * y[i] - r * (z[i]+1/2*h*k31)

    k13 = s * ((x[i]+1/2*h*k12)-y[i])
    k23 = (m-z[i])-(y[i]+1/2*h*k22)
    k33 = x[i] * y[i] - r * (z[i]+1/2*h*k32)

    k14 = s * ((x[i]+1/2*h*k13)-y[i])
    k24 = (m-z[i])-(y[i]+1/2*h*k23)
    k34 = x[i] * y[i] - r * (z[i]+1/2*h*k33)

    xn = x[i] + h / 6 *(k11+2*k12+2*k13+k14)
    yn = y[i] + h / 6 *(k21+2*k22+2*k23+k24)
    zn = z[i] + h / 6 *(k31+2*k32+2*k33+k34)
    x.append(xn)
    y.append(yn)
    z.append(zn)


plt.plot(t, z[: -1], label='x')
plt.legend()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1])
y = np.array([2])
z = np.array([3])
X = np.array([[2], [1]])  # 给定初始值

s = 10.0
m = 28.0
r = 8 / 3
h = 0.01  # 步长
t = np.arange(0, 10, h)  # 仿真时间

for i, _ in enumerate(t):
    k11 = s * (x[i] - y[i])
    k21 = (m - z[i]) - y[i]
    k31 = x[i] * y[i] - r * z[i]

    k12 = s * ((x[i] + 1 / 2 * h * k11) - (y[i] + 1 / 2 * h * k21))
    k22 = (m - z[i] + 1 / 2 * h * k31) - (y[i] + 1 / 2 * h * k21)
    k32 = (x[i] + 1 / 2 * h * k11) * (y[i] + 1 / 2 * h * k21) - r * (
        z[i] + 1 / 2 * h * k31)

    k13 = s * ((x[i] + 1 / 2 * h * k12) - (y[i] + h / 2 * k22))
    k23 = (m - (z[i] + h / 2 * k32)) - (y[i] + 1 / 2 * h * k22)
    k33 = (x[i] + h / 2 * k12) * (y[i] + h / 2 * k22) - r * (z[i] +
                                                             1 / 2 * h * k32)

    k14 = s * ((x[i] + 1 / 2 * h * k13) - (y[i] + h / 2 * k23))
    k24 = (m - (z[i] + h / 2 * k33)) - (y[i] + 1 / 2 * h * k23)
    k34 = (x[i] + h / 2 * k13) * (y[i] + h / 2 * k23) - r * (z[i] +
                                                             1 / 2 * h * k33)

    xn = x[i] + h / 6 * (k11 + 2 * k12 + 2 * k13 + k14)
    yn = y[i] + h / 6 * (k21 + 2 * k22 + 2 * k23 + k24)
    zn = z[i] + h / 6 * (k31 + 2 * k32 + 2 * k33 + k34)
    #     print(xn)
    x = np.append(x, xn)
    y = np.append(y, yn)
    z = np.append(z, zn)
#     print(x)

# x1 = y.copy()
# x1 = np.array(x1)
# x = np.array(x)
# x2 = x1 - 1.1 * x1 + 3.0 * x
plt.plot(t, x[:-1], label='x')
plt.legend()
plt.show()
# %%
%%time
import numpy as np
import matplotlib.pyplot as plt

h = 1e-5  # 步长
t = np.arange(0, 1, h)
N = len(t)

x = np.ones(N)
y = np.ones(N)
z = np.ones(N)


for i, _ in enumerate(t[:-1]):
    x_n = x[i]
    y_n = y[i]
    z_n = z[i]
    t_n = t[i]

    kx1 = y_n + 3 * z_n + np.sin(5*t_n)
    ky1 = x_n + np.cos(t_n)
    kz1 = x_n + z_n - 3 * np.cos(3*t_n) * np.sin(4*t_n)

    kx2=(y_n+ky1*h/2)+3*(z_n+kz1*h/2)+np.sin(5*(t_n+h/2))
    ky2=(x_n+kx1*h/2)+np.cos(t_n+h/2)
    kz2=(x_n+kx1*h/2)+(z_n+kz1*h/2)-3*np.cos(3*(t_n+h/2))*np.sin(4*(t_n+h/2))
    
    kx3=(y_n+ky2*h/2)+3*(z_n+kz2*h/2)+np.sin(5*(t_n+h/2))
    ky3=(x_n+kx2*h/2)+np.cos(t_n+h/2)
    kz3=(x_n+kx2*h/2)+(z_n+kz2*h/2)-3*np.cos(3*(t_n+h/2))*np.sin(4*(t_n+h/2))
    
    kx4=(y_n+ky3*h)+3*(z_n+kz3*h)+np.sin(5*(t_n+h))
    ky4=(x_n+kx3*h)+np.cos(t_n+h)
    kz4=(x_n+kx3*h)+(z_n+kz3*h)-3*np.cos(3*(t_n+h))*np.sin(4*(t_n+h))
    
    x[i+1]=x_n+h/6*(kx1+2*kx2+2*kx3+kx4)
    y[i+1]=y_n+h/6*(ky1+2*ky2+2*ky3+ky4)
    z[i+1]=z_n+h/6*(kz1+2*kz2+2*kz3+kz4)

plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.plot(t, z, label='z')
plt.legend()
plt.show()
print(x)
# %%
import numpy as np
import matplotlib.pyplot as plt

h = 1e-5
t = np.arange(0, 1, h)
# print(t)
x = np.ones(len(t))
y = np.ones(len(t))
z = np.ones(len(t))

for i, _ in enumerate(t[: -1]):
    x_n = x[i]
    y_n = y[i]
    z_n = z[i]
    t_n = t[i]
    
    kx1 = y_n + 3 * z_n + np.sin(5*t_n)
    ky1 = x_n + np.cos(t_n)
    kz1 = x_n + z_n - 3 * np.cos(3*t_n) * np.sin(4*t_n)
    
    kx2 = (y_n+h/2*ky1) + 3 * (z_n+h/2*kz1) + np.sin(5*(t_n+h/2))
    ky2 = (x_n+h/2*kx1) + np.cos(t_n+h/2)
    kz2 = (x_n+h/2*kx1) + (z_n+h/2*kz1) - 3 * np.cos(3*(t_n+h/2)) * np.sin(4*(t_n+h/2))
    
    kx3 = (y_n+h/2*ky2) + 3 * (z_n+h/2*kz2) + np.sin(5*(t_n+h/2))
    ky3 = (x_n+h/2*kx2) + np.cos(t_n+h/2)
    kz3 = (x_n+h/2*kx2) + (z_n+h/2*kz2) - 3 * np.cos(3*(t_n+h/2)) * np.sin(4*(t_n+h/2))
    
    kx4 = (y_n+h*ky3) + 3 * (z_n+h*kz3) + np.sin(5*(t_n+h))
    ky4 = (x_n+h*kx3) + np.cos(t_n+h)
    kz4 = (x_n+h*kx3) + (z_n+h*kz3) - 3 * np.cos(3*(t_n+h)) * np.sin(4*(t_n+h))
    
    x[i+1] = x_n + h / 6 * (kx1+2*kx2+2*kx3+kx4)
    y[i+1] = y_n + h / 6 * (ky1+2*ky2+2*ky3+ky4)
    z[i+1] = z_n + h / 6 * (kz1+2*kz2+2*kz3+kz4)
    
plt.plot(t, x)
plt.plot(t, y)
plt.plot(t, z)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

h = 1e-5  # 步长
t = np.arange(0, 1, h)
N = len(t)

x = np.ones(N)
y = np.ones(N)
z = np.ones(N)


for i, t_n in enumerate(t[:-1]):
    x_n = x[i]
    y_n = y[i]
    z_n = z[i]

    kx1 = y_n + 3 * z_n + np.sin(5*t_n)
    ky1 = x_n + np.cos(t_n)
    kz1 = x_n + z_n - 3 * np.cos(3*t_n) * np.sin(4*t_n)

    kx2=(y_n+ky1*h/2)+3*(z_n+kz1*h/2)+np.sin(5*(t_n+h/2))
    ky2=(x_n+kx1*h/2)+np.cos(t_n+h/2)
    kz2=(x_n+kx1*h/2)+(z_n+kz1*h/2)-3*np.cos(3*(t_n+h/2))*np.sin(4*(t_n+h/2))
    
    kx3=(y_n+ky2*h/2)+3*(z_n+kz2*h/2)+np.sin(5*(t_n+h/2))
    ky3=(x_n+kx2*h/2)+np.cos(t_n+h/2)
    kz3=(x_n+kx2*h/2)+(z_n+kz2*h/2)-3*np.cos(3*(t_n+h/2))*np.sin(4*(t_n+h/2))
    
    kx4=(y_n+ky3*h)+3*(z_n+kz3*h)+np.sin(5*(t_n+h))
    ky4=(x_n+kx3*h)+np.cos(t_n+h)
    kz4=(x_n+kx3*h)+(z_n+kz3*h)-3*np.cos(3*(t_n+h))*np.sin(4*(t_n+h))
    
    x[i+1]=x_n+h/6*(kx1+2*kx2+2*kx3+kx4)
    y[i+1]=y_n+h/6*(ky1+2*ky2+2*ky3+ky4)
    z[i+1]=z_n+h/6*(kz1+2*kz2+2*kz3+kz4)

plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.plot(t, z, label='z')
plt.legend()
plt.show()
# print(x)
# %%

# %%
from operator import itemgetter

a = [[1, 2], [2, 3], [4, 5], [3, 7]]
a = sorted(a, key=itemgetter(1), reverse=True)
print(a)

# %%
# %%time
import numpy as np
# %%
%%timeit
a = np.array([])
for i in np.arange(10000):
    a = np.append(a, i)

# %% 
%%timeit
b = []
for i in range(10000):
    b.append(i)
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


# 定义init_func给定初始信息
def init():
    line.set_ydata([np.nan]*len(x))
    return line,


# 定义func, 用来反复调用的函数
def animate(i):
    line.set_ydata(np.sin(x + i / 100))
    return line,


ani = FuncAnimation(fig, animate, init_func=init, interval=2, blit=True, save_count=50)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
w = np.ones((3, 3))
y = w * 2
g = w * 3
b = w * 4
r = w * 5
o = w * 6
z = np.zeros((3, 3))
ma = [[z, w, z],
      [g, r, b], 
      [z, y, z],
      [z, o, z]]
ma = np.concatenate((np.concatenate((z, w, z), axis=1), 
                     np.concatenate((g, r, b), axis=1), 
                     np.concatenate((z, y, z), axis=1), 
                     np.concatenate((z, o, z), axis=1)))

ma = np.random.permutation(ma)
plt.imshow(ma)
plt.show()
# %%
import numpy as np
rand = np.random.RandomState(42)
mean = [0, 0]
cov = [[1, 2], [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape
# %%
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
plt.scatter(X[:, 0], X[:, 1])
# %%
indices = np.random.choice(X.shape[0], 20, replace=False)
indices
# %%
selection = X[indices]
selection.shape
# %%
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1], facecolor='none', edgecolor='b', s=200)
# %%
import numpy as np
import pandas as pd
