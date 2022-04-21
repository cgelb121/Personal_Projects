# add necessary imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


# constants

G = 4*np.pi**2

mass = {
    'sun': 1.0,
    'earth': 3.0034e-6,
    'moon': 3.6923e-7}

r0 = {
    'sun': np.array([0,0,0]),
    'earth': np.array([9.978977040419635E-01, 6.586825681892025E-02, -6.320430920521123E-06]),
    'moon': np.array([9.956768547953816E-01, 6.676030485840675E-02, 1.641093070596718E-04])
     }
v0 = {
    'sun': np.array([0,0,0]),
    'earth': np.array([-4.70015711e-01, 6.25165839e+00, -3.40817831e-04]),
    'moon': np.array([-0.55065949, 6.03534661, 0.01111456])
}

# functions

def F_gravity(ri,rj,mi,mj):
    rij = ri - rj
    rij_mag = np.sqrt(rij.dot(rij))
    rhat = (rij / rij_mag)
    force = ((-G * mi * mj) / (rij_mag)**2) * np.array(rhat)
    return np.array(force)

def F_ES(rE):
    return np.array(F_gravity(rE,r0['sun'],mass['earth'],mass['sun']))
def F_EM(rE,rM):
    return np.array(F_gravity(rE,rM,mass['earth'],mass['moon']))
    
def F_ME(rM,rE):
    return np.array(F_EM(rE,rM))
def F_MS(rM):
    return np.array(F_gravity(rM,r0['sun'],mass['moon'],mass['sun']))
    
def F_E(rE, rM): #total force from moon and sun
    return np.array(F_EM(rE,rM)) + np.array(F_ES(rE))
    
def F_M(rE, rM): #total force from earth and sun
    return -1*np.array(F_ME(rM,rE)) + np.array(F_MS(rM))

def integrate_EM(tmax, dt=1e-3):
    re_values = [r0['earth']]
    ve_values = [v0['earth']]
    rm_values = [r0['moon']]
    vm_values = [v0['moon']]
    t = 0
    n = 0
    
    while 0 <= t < tmax:
        # Euler
        rE = re_values[n] + ve_values[n] * dt
        rM = rm_values[n] + vm_values[n] * dt 
        re_values.append(rE.copy())
        rm_values.append(rM.copy())
        
        vE = ve_values[n] + mass['earth']**-1*(F_E(rE,rM))*dt
        vM = vm_values[n] + mass['moon']**-1*(F_M(rE,rM))*dt
        ve_values.append(vE.copy())
        vm_values.append(vM.copy())
        
        t += dt
        n +=1
    r_E = np.array(re_values)
    r_M = np.array(rm_values)
    return r_E, r_M


if __name__ == "__main__":
    # create the trajectory
    trajectory = integrate_EM(1)
    radial = np.array(trajectory)
    x_earth = radial[0,:,0]
    y_earth = radial[0,:,1]
    z_earth = radial[0,:,2]
    x_moon = radial[1,:,0]
    y_moon = radial[1,:,1]
    z_moon = radial[1,:,2]
    
    # plot
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    
    ax.plot(x_earth,y_earth,z_earth, 'r-', label = 'Earth')
    ax.plot(x_moon,y_moon,z_moon, 'b--', label = 'Moon')
    x = np.arange(-1,1.1,0.1)
    y = np.arange(-1,1.1,0.1)
    ax.tick_params(axis='x', rotation = 30)
    ax.tick_params(axis='y', rotation = -30)
    
    #plt.xlabel('x (AU)')
    #plt.ylabel('y (AU)')
    ax.legend()
    #plt.gca().set_aspect("equal")
    #ax.plot(r0["earth"][0],r0["earth"][1])
    plt.tight_layout()
    plt.savefig("orbit_earth_moon.png")