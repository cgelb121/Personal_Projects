import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

def laneemden(y,xi,n): # Lane Emden 
    theta, phi = y[0], y[1]  
    dydxi = [-phi/(xi**2), (xi**2)*theta**n]  
    return dydxi

fig, ax = plt.subplots()

y0 = [1.,0.]  
xi = np.linspace(10e-4, 16., 201)

for n in range(6):
    sol = odeint(laneemden, y0, xi, args=(n,))
    ax.plot(xi, sol[:, 0])

ax.axhline(y = 0.0, linestyle = '--', color = 'k')
ax.set_xlim([0, 15])
ax.set_ylim([-0.5, 1.0])
ax.set_xlabel(r"$\xi$ - radial param.")
ax.set_ylabel(r"$\theta$ - pressure & density param.")
#plt.legend('012345')
plt.legend(['n = 0', 'n = 1', 'n = 2', 'n = 3', 'n = 4', 'n = 5'])
plt.title('Lane-Emden Equation of State')
plt.grid()