from scipy.fft import fft, ifft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 100
dt = 0.01
t = np.linspace(0, 1, 100)
x = np.arange(-1, 1, 0.02)
k = []



def f0(x):
    return 3 * np.exp(-8 * (x ** 2))


U = np.zeros((n, n), dtype=np.complex64)

for i in range(n):
    U[0][i] = f0(x[i])


k = fftfreq(100)*np.pi*100



for i in range(1,n):
    U[i] = ifft(np.exp(-complex(0,1)*(k**2)*dt)*fft(np.exp(-1*complex(0,1)*(x**2)*dt)*U[i-1]))

for i in range(1,n):
    U[i][0]=U[i-1][0]
    U[i][99]=U[i-1][99]

for i in range(0, n-1):
    if (abs(sum(abs(U[i]) ** 2) - sum(abs(U[i + 1]) ** 2))<0.1):
        print("Интеграл сохраняется")
    else:
        print("Интеграл не сохраняется")
        print(sum(abs(U[i]) ** 2))

for i in range(n):
    for j in range(n):
        U[i][j] = abs(U[i][j])

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(x, t)
Z = U

ax.plot_surface(X, Y, U, rcount=100, ccount=100,
                cmap='viridis')
ax.set_xlabel('x, м')
ax.set_ylabel('t, c')
ax.set_title('U(x,t)')
plt.savefig('Surface.png')
plt.show()
