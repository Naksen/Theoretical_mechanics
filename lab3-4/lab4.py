import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import math
from matplotlib.animation import FuncAnimation
import sympy as sp
from scipy.integrate import odeint

def odesys(y, t, M, m, c, k, l, F0, p):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = M + m
    a12 = m * l * np.cos(y[1])
    a21 = np.cos(y[1])
    a22 = 3 * l / 2

    b1 = F0 * np.sin(p*t) - c * y[0] - k * y[2] + m * l * y[3] ** 2 * np.sin(y[1])
    b2 = -np.sin(y[1]) * m * 9.81 * l

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy

if __name__ == '__main__':

    t = sp.Symbol('t')

    # Начальные параметры

    M = 10
    m = 3
    R = 1
    r = 0.1
    F0 = 0
    p = 0.4
    c = 0 # 2
    k = 0 # 3
    l = R-r
    x0 = 0
    alpha0 = 0.1 # 1
    dx0 = 0
    dalpha0 = 0

    d = R - r
    O1_x = 1.1
    O1_y = 1.3

    T = np.linspace(0, 50, 4001)

    y0 = [x0, alpha0, dx0, dalpha0]
    # Решение дифура
    Y = odeint(odesys, y0, T, (M, m, c, k, l, F0, p))


    alpha = Y[:, 1]
    Rec_x = Y[:, 0]
    V_x = Y[:, 2]

    # o_y = O1_y - d * sp.cos(alpha)
    # o_x = O1_x + d * sp.sin(alpha) + Rec_x

    # VxO = sp.diff(o_x, t)
    # VyO = sp.diff(o_y, t)

    # VxRec = sp.diff(Rec_x, t)

    # Массивы
    OY = np.zeros_like(T)
    OX = np.zeros_like(T)
    VXREC = np.zeros_like(T)
    VXO = np.zeros_like(T)
    VYO = np.zeros_like(T)

    for i in np.arange(len(T)):
        # ALPHA[i] = sp.Subs(alpha, t, T[i])
        # REC_X[i] = sp.Subs(Rec_x, t, T[i])
        OY[i] = O1_y - d * sp.cos(alpha[i])
        OX[i] = O1_x + d * sp.sin(alpha[i]) + Rec_x[i]
        # VXREC[i] = sp.Subs(VxRec, t, T[i])
        # VXO[i] = sp.Subs(VxO, t, T[i])
        # VYO[i] = sp.Subs(VyO, t, T[i])

    # Границы рисунка
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(4, 10))
    ax = fig.add_subplot(1, 2, 1)
    plt.xlim(-3, 4)
    plt.ylim(-1, 4)
    ax.set_aspect(1)

    # Прямоугольник
    body1 = plt.Rectangle((x0, 0.0), width=2.2, height=1.3, color='b')

    # Линия нижней поверхности
    bottom_line_x = [-2, 2.5]
    bottom_line_y = [0, 0]
    plt.plot(bottom_line_x, bottom_line_y, 'k')

    # Линия боковой поверхности
    side_line_x = [-1.5, -1.5]
    side_line_y = [0, 2]
    plt.plot(side_line_x, side_line_y, 'k')

    # Выколотая окружность
    white_circle = plt.Circle((1.1, 1.3), radius=1, color='w')

    # Функция создает набор координат x,y для построения зигзагообразной линии, которая изображает пружину
    def get_spring_line(length, coils, diameter):
        x = np.linspace(0, length, coils * 2)
        y = [diameter * 0.5 * (-1) ** i for i in range(len(x))]
        return np.array([x, y])

    # Пружина
    spring_xy = get_spring_line(1.5, 10, 0.1)
    spring = mlines.Line2D(spring_xy[0] - 1.5, spring_xy[1] + 0.25, lw=0.5, color='r')

    # Цилиндр
    cylinder = plt.Circle((OX[0], OY[0]), radius=0.1, color='r')

    # Графики
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(T, alpha)
    plt.title('Alpha of the cylinder')
    plt.xlabel('t values')
    plt.ylabel('alpha values')

    ax3 = fig.add_subplot(4, 2, 4)
    ax3.plot(T, Rec_x)
    plt.title('x of the Ramp')
    plt.xlabel('t values')
    plt.ylabel('x values')

    ax4 = fig.add_subplot(4, 2, 6)
    ax4.plot(T, V_x)
    plt.title('V of the Ramp')
    plt.xlabel('t values')
    plt.ylabel('V values')

    plt.subplots_adjust(wspace=0.3, hspace=0.7)

    def init():
        # Прямоугольник
        ax.add_patch(body1)
        # Выколотая окружность
        ax.add_patch(white_circle)
        # Пружина
        ax.add_line(spring)
        # Цилиндр
        ax.add_patch(cylinder)
        return body1, white_circle, spring, cylinder

    def anima(j):  # Анимация движения
        cylinder.center = OX[j], OY[j]
        white_circle.center = Rec_x[j] + 1.1, 1.3
        body1.xy = Rec_x[j], 0
        sp_xy = get_spring_line(1.5 + Rec_x[j], 10, 0.1)
        spring.set_data(sp_xy[0] - 1.5, sp_xy[1] + 0.25)
        return body1, white_circle, spring, cylinder


    # Анимация
    anim = FuncAnimation(fig, anima, init_func=init, frames=len(T), interval=10, blit=True)
    plt.show()