from astropy import units, coordinates        
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# take date from a file, np.loadtext = _____.open() file

data = np.loadtxt('lss_data_highz.txt') #1st column = RA, 2nd = Dec, 3rd = z-shift (distance away)
print(data[0,:]) #starts from first row(0), all columns (:) ----- data set of 76k+ rows 3 columns - galaxies?

%matplotlib inline 
'''IPython has a set of predefined ‘magic functions’ that you can call with a command line style syntax. 
There are two kinds of magics, line-oriented and cell-oriented. 
Line magics are prefixed with the % character and work much like OS command-line calls: 
they get as an argument the rest of the line, where arguments are passed without parentheses or quotes. 
Lines magics can return results and can be used in the right hand side of an assignment. 
Cell magics are prefixed with a double %%, and they are functions that get as an argument not only the rest of the line, 
but also the lines below it in a separate argument.
'''

#plots RA and DEC, all rows, RA and DEC columns for X and Y, 'o' plots dot, alpha=transparacy of points
plt.subplot(projection="aitoff") # special oval format of plot
plt.plot(data[:,0],data[:,1],'.', alpha = 0.1)
plt.xlabel('Right Ascension (Deg.)')
plt.ylabel('Declination (Deg.)')
plt.title('Location of Objects on Sky SDSS')
plt.grid()
plt.show()
 

# Sloan Digital Sky Survey Data Points - galaxy survey count with position and distance
sdsspts = coordinates.SkyCoord(ra=data[:,0]*units.deg,dec=data[:,1]*units.deg) # .Skycoord reads data format in 

print('Length of sdsspts data list:',len(sdsspts)) #RA and DEC in degrees
print('shape of data set:', sdsspts.shape)
print(f'Value Range in Right Ascention: {sdsspts.ra}') # prints whole column of RA

# looking for points in a particular field (140-230)RA, (3-60)DEC and stating its from the first input in the list
woksdss=np.where((sdsspts.ra>140*units.deg) & (sdsspts.ra<230*units.deg) & (sdsspts.dec>3*units.deg) & (sdsspts.dec<60*units.deg))[0]
# need to use [0] dimension at the end of np.where because it returns multidimensional values typically
# can use '|' to represent 'or' in np.where

print(f'Boxed out location of sky where: (140-230RA,3-60DEC) data set: {woksdss}')


# applying total list of SDSS data with Selected area of points
plt.plot(sdsspts.ra[woksdss], sdsspts.dec[woksdss], 'o', alpha= 0.1)
plt.title('Random area of points in sdsspts[woksdss]')
plt.xlabel('Right Ascension (Deg.)')
plt.ylabel('Declination (Deg.)')
plt.xlim(160,168)
plt.ylim(10,20)
plt.grid()
plt.show()

np.random.randn(10,2) # 10 rows 2 columns normally distributed around 0 with std 1
ras = np.random.rand(1000)*(230-140)+140 
decs = np.random.rand(1000)*(60-3)+3
plt.plot(ras,decs,'o',alpha=.1)
plt.grid
plt.title('Random Distribution of RA/DEC from sdss limit')
plt.grid()
plt.show()

sdsspts=sdsspts[woksdss] #limiting SDSS points to particular area of sky
print('The new sdsspts list length is: ', len(sdsspts))

def getrandompoints(num,rarange,decrange):
    uv=np.random.rand(num*10,2) # prints 2 columns with n * 10 rows
    ra=2*np.pi*uv[:,0]*180/np.pi # converts first column of RA units to degrees
    dec=np.arccos(2*uv[:,1]-1)*180/np.pi-90 # converts second column of DEC units to degrees
    
    w=np.where((ra>=rarange[0]) & (ra<=rarange[1]) & (dec>=decrange[0]) & (dec<=decrange[1]))[0][0:num] # for array: [rows][columns]
    
    print('w is an array: ', type(w))
    
    #last 2 lists is dimensionality and all random points included for where
    
    return ra[w], dec[w]


rra,rdec=getrandompoints(1000,[150,220],[13,50]) #Num of points, ra coord, dec coord
#print(rra,rdec)
plt.plot(rra,rdec,'o')
plt.title('Random points similar RA/DEC ranges')
plt.grid()
plt.show()

#plt.plot(sdsspts.ra[woksdss], sdsspts.dec[woksdss], 'o', alpha=0.1) #data
#plt.plot(rra,rdec,'o') #random generation
#plt.show()

rpts=coordinates.SkyCoord(ra=rra*units.deg,dec=rdec*units.deg) # making a new coordinate map with value/degree update
print('RA has been converted to hours/mins/secs:', rpts[0].separation(sdsspts[0])) #random point coordinate of 0th element



nin1=[]

for i in range(len(rpts)):
    seps=rpts[i].separation(sdsspts)
    w=np.where(seps<1*units.deg)[0] #1 degree away from 0th element in rpts list
    nin1.append(len(w))
    
print(f'nin populated list: {len(nin1)}')

plt.hist(nin1)
plt.xlabel('Num of Galaxies around a random point')
plt.ylabel('Num of of points that have many Galaxies around them', fontsize=10)
plt.title('Random Galaxies')
plt.grid()
plt.show()


randompointings = getrandompoints(1000,[150,220],[13,50])
sim_gals=getrandompoints(len(sdsspts),[140,230],[3,60]) #sim galaxy points

plt.plot(sim_gals[0],sim_gals[1],'k.',alpha=.1)
plt.plot(randompointings[0],randompointings[1],'ro',alpha=.1,)
plt.plot(rra,rdec,'go',alpha=.1)
plt.legend(['black = sim', 'red = rand', 'green = data'])
plt.title('sim gals, rand points, sdss data')
plt.grid()
plt.show()

sim_coords=coordinates.SkyCoord(sim_gals[0]*units.deg,sim_gals[1]*units.deg)
rand_point_coords=coordinates.SkyCoord(randompointings[0]*units.deg,randompointings[1]*units.deg)

nin1_sim=[]

for i in range(len(rand_point_coords)):
    pointsin1=np.where(rand_point_coords[i].separation(sim_coords)<1*units.deg)[0] #1 degree away from 0th element in rpts list
    nin1_sim.append(len(pointsin1))

print( 'length of sim_coords and sdsspts', len(sim_coords),len(sdsspts))
print('the mean of nin1: ',np.mean(nin1),'the mean for nin1_sim: ',np.mean(nin1_sim))

plt.hist(nin1,histtype='step')
plt.hist(nin1_sim,histtype='step')
plt.title('Sim vs. Real galaxy data')
plt.grid()
plt.show()


plt.hist(nin1,histtype='step',density=True)
plt.hist(nin1_sim,histtype='step',density=True)
plt.plot(np.arange(1,60),np.mean(nin1_sim)**np.arange(1,60)*np.exp(-np.mean(nin1_sim))/gamma(np.arange(1,60)+1)) #Poisson distribution
plt.title('Means per data set')
plt.legend(['nin1', 'nin1_sim', 'poisson nin1'])
plt.grid()
plt.show()

print('The standard distribution of simulated galxies is:',np.std(nin1_sim))