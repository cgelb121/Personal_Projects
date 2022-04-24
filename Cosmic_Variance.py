from astropy import units, coordinates
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# take date from a file, np.loadtext = _____.open() file? neat

data = np.loadtxt('lss_data_highz.txt') #1st column = RA, 2nd = Dec, 3rd = z-shift
print(data[0,:]) #prints first row(0), all columns (:)

%matplotlib inline

plt.plot(data[:,0],data[:,1],'o',alpha = .1) #plots RA and DEC, all rows, RA and DEC columns for X and Y, 'o' plots dot, alpha=transparacy of points 
plt.show()
sdsspts = coordinates.SkyCoord(ra=data[:,0]*units.deg,dec=data[:,1]*units.deg) # Sloan Digital Sky Survey Data 

print('Length of sdsspts data list:',len(sdsspts)) #RA and DEC in degrees
print('shape of data set:', sdsspts.shape)
print(sdsspts.ra) #RA units

woksdss=np.where((sdsspts.ra>140*units.deg) & (sdsspts.ra<230*units.deg) & (sdsspts.dec>3*units.deg) & (sdsspts.dec<60*units.deg))[0]
# need to use [0] dimension at the end of np.where because it returns multidimensional values typically
# can use '|' to represent 'or' in np.where

print(woksdss)



plt.plot(sdsspts.ra[woksdss], sdsspts.dec[woksdss], 'o', alpha=0.1)
plt.xlim(160,168)
plt.ylim(10,20)
plt.show()

np.random.randn(10,2) # 10 rows 2 columns normally distributed around 0 with std 1
ras = np.random.rand(1000)*(230-140+140)
decs = np.random.rand(1000)*(60-3)+3
plt.plot(ras,decs,'o',alpha=.1)
plt.show()

sdsspts=sdsspts[woksdss]

def getrandompoints(num,rarange,decrange):
    uv=np.random.rand(num*10,2)
    ra=2*np.pi*uv[:,0]*180/np.pi
    dec=np.arccos(2*uv[:,1]-1)*180/np.pi-90
    w=np.where((ra>=rarange[0]) & (ra<=rarange[1]) & (dec>=decrange[0]) & (dec<=decrange[1]))[0][0:num] #last 2 lists is dimensionality and all random points included for where 
    return ra[w], dec[w]

rra,rdec=getrandompoints(1000,[150,220],[13,50]) #Num of points, ra coord, dec coord
#print(rra,rdec)
plt.plot(rra,rdec,'o')
plt.show()

#plt.plot(sdsspts.ra[woksdss], sdsspts.dec[woksdss], 'o', alpha=0.1) #data
#plt.plot(rra,rdec,'o') #random generation
#plt.show()

rpts=coordinates.SkyCoord(ra=rra*units.deg,dec=rdec*units.deg)
print(rpts[0].separation(sdsspts[0])) #random point coordinate of 0th element

nin1=[]

for i in range(len(rpts)):
    seps=rpts[i].separation(sdsspts)
    w=np.where(seps<1*units.deg)[0] #1 degree away from 0th element in rpts list
    nin1.append(len(w))
    
#print(nin1)

plt.hist(nin1)
plt.xlabel('Num of Galaxies around a random point')
plt.ylabel('Num of of points that have many Galaxies around them', fontsize=10)
plt.show()


randompointings = getrandompoints(1000,[150,220],[13,50])
sim_gals=getrandompoints(len(sdsspts),[140,230],[3,60]) #sim galaxy points

plt.plot(sim_gals[0],sim_gals[1],'o',alpha=.1)
plt.plot(randpointings[0],randpointings[1],'o',alpha=.1)
#plt.plot(rra,rdec,'o',alpha=.1)
plt.show()

sim_coords=coordinates.SkyCoord(sim_gals[0]*units.deg,sim_gals[1]*units.deg)
rand_point_coords=coordinates.SkyCoord(randompointings[0]*units.deg,randompointings[1]*units.deg)

nin1_sim=[]

for i in range(len(rand_point_coords)):
    pointsin1=np.where(rand_point_coords[i].separation(sim_coords)<1*units.deg)[0] #1 degree away from 0th element in rpts list
    nin1_sim.append(len(pointsin1))

print(len(sim_coords),len(sdsspts), 'length of sim_coords and sdsspts')
print(np.mean(nin1),np.mean(nin1_sim))

plt.hist(nin1,histtype='step')
plt.hist(nin1_sim,histtype='step')
plt.title('Sim vs. Real galaxy data')
plt.show()


plt.hist(nin1,histtype='step',density=True)
plt.hist(nin1_sim,histtype='step',density=True)
plt.plot(np.arange(1,60),np.mean(nin1_sim)**np.arange(1,60)*np.exp(-np.mean(nin1_sim))/gamma(np.arange(1,60)+1)) #Poisson distribution
plt.show()

print('The standard distribution of simulated galxies is:',np.std(nin1_sim))