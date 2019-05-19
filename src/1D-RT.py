#Importing
import sys
import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt
#Mathematical constants
pi = np.pi
tpi = 2.0*pi
fpi = 4.0*pi
zI = 1.0j
#Physical constants
sol = 137.0 #speed of light
aB = 0.05292 #nanometer
Hartree = 27.21 #eV
Atomtime = 0.02419 #fs
Atomfield = 514.2 #V/nm
ch = 1240.0 #eV * nm
chbar = 197.3 # eV * nm
halfepsc = 3.509e16 # W/cm^2 \frac{1}{2}*\epsilon_0 * c 
Atomfluence = halfepsc*Atomtime*1.0e-15 # J/cm^2 ,W/cm^2 * fs = femto J/cm^2
#Default values
sys_name = '_'
cluster_mode = False
write_ASCII = False #Writting Data to not only *.npz but also ASCII files, only available for Jt
a1 = 8.0 #Lattice constant
flat = -1.0 #Length of flat potential
NG1 = 12 #Number of grid point in both real/reciprocal space
NK1 = 20 #Number of grid point along 1-axis in Brillouin zone
Nave = 4.0  #Number of particle in a cell
NT = 4000 #Number of time steps
dt = 5.0e-1 #Size of a time-step
# Giving parameters for the field
omegac = 0.3875 #eV: Center frequency
E = 0.9*1.843 #V/nm: Field strength
Tpulse = 40.00
phiCEP = 0.25 #tpi
Delay = 0.0
nenvelope = 4

argv = sys.argv
argc = len(argv)

#Reading part from standard input
if (argc == 1):
    print('Defalut parameters are chosen.')
elif (argc == 2):
    print('Name of input file is "'+argv[1]+'".')
    f = open(argv[1],'r')
    lines = f.readlines()
    f.close
    Nlen = len(lines)
    text = [0]*Nlen
    for i in range(Nlen):
        text[i] = lines[i].strip()
    for i in range(Nlen):
        if (str(text[i]) == 'sys_name') :
            sys_name = str(text[i+1])
        if (str(text[i]) == 'cluster_mode') :
            cluster_mode = str(text[i+1])
            if (cluster_mode=='True'):
                cluster_mode = True
            else :
                cluster_mode = False
        if (str(text[i]) == 'write_ASCII') :
            write_ASCII = str(text[i+1])
            if (write_ASCII=='True'):
                write_ASCII = True
            else :
                write_ASCII = False
        if (str(text[i]) == 'a1') :
            a1 = float(str(text[i+1]))
        if (str(text[i]) == 'flat') :
            flat = float(str(text[i+1]))
        if (str(text[i]) == 'NG1') :
            NG1 = int(str(text[i+1]))
        if (str(text[i]) == 'NK1') :
            NK1 = int(str(text[i+1]))
        if (str(text[i]) == 'Nave') :
            Nave = int(str(text[i+1]))
        if (str(text[i]) == 'NT') :
            NT = int(str(text[i+1]))
        if (str(text[i]) == 'dt') :
            dt = float(str(text[i+1]))
        if (str(text[i]) == 'omegac') :
            omegac = float(str(text[i+1]))
        if (str(text[i]) == 'E') :
            E = float(str(text[i+1]))
        if (str(text[i]) == 'Tpulse') :
            Tpulse = float(str(text[i+1]))
        if (str(text[i]) == 'phiCEP') :
            phiCEP = float(str(text[i+1]))
        if (str(text[i]) == 'Delay') :
            Delay = float(str(text[i+1]))
        if (str(text[i]) == 'nenvelope') :
            nenvelope = int(str(text[i+1]))
else :
    sys.exit('Error: Number of argument is wrong.')

#Primitive cell construction
b1 = tpi/a1
H1 = a1/np.float(NG1)
x1 = np.linspace(0.0,a1,num=NG1,endpoint=False,dtype='float64')
G1 = np.fft.fftfreq(NG1)*(b1*np.float(NG1))
#Brillouin zone construction
k1 = np.linspace(-0.5*b1,0.5*b1,num=NK1,endpoint=False,dtype='float64')
k1 = k1 + (0.5*b1)/np.float(NK1)

ubkG = np.zeros([NG1,NG1,NK1],dtype='complex128') #Wave function in reciprocal space
hk = np.zeros([NG1,NG1,NK1],dtype='complex128') #Hamiltonian in terms of reciprocal space
epsbk = np.zeros([NG1,NK1],dtype='float64') #Hamiltonian in terms of reciprocal space
occbk = np.zeros([NG1,NK1],dtype='float64') #Occupation number

Nocc = int(Nave/2.0)
occbk[0:Nocc,:] = 2.0/float(NK1)

def Make_vextr():
    alpha = 5.0e-2
    beta = 5.0e-2
    gamma = 1.0e-1
    v0 =  0.37
    vextrloc = -v0*(1.0+np.cos(tpi*x1/a1))
    if (flat > 0.0):
        if (flat > a1):
            print('Error: flat is larger than a1.')
            sys.exit()
        for ig in range(NG1):
            if (x1[ig] < (a1-flat)):
                vextrloc[ig] = -v0*(1.0 - np.cos(tpi*x1[ig]/(a1-flat)))
            else : 
                vextrloc[ig] = 0.0
        if(not cluster_mode):
            plt.figure()
            plt.plot(x1,vextrloc,label='The local potential')
            plt.grid()
            plt.legend()
            plt.show()
    return vextrloc
vextr = Make_vextr()
vsG = np.fft.fft(vextr)/np.float(NG1)

def Make_hk(Aloc):
    vsGG = np.zeros([NG1,NG1],dtype='complex128') 
    for ig1 in range(NG1):
        gind = np.remainder(np.arange(NG1)+NG1-ig1,NG1)
        for ig2 in range(NG1):
            igloc = gind[ig2]
            vsGG[ig1,ig2] = vsG[igloc] #This definition should be carefully checked 
    for ik in range(NK1):
        hk[:,:,ik] = vsGG[:,:]
        kloc = k1[ik] + Aloc
        for ig1 in range(NG1):
            hk[ig1,ig1,ik] = hk[ig1,ig1,ik] + 0.5*(G1[ig1]+kloc)**2
    return hk
hk = Make_hk(0.0)

#Band calculation 
for ik in range(NK1):
    epsbk[:,ik], ubkG[:,:,ik] = np.linalg.eigh(hk[:,:,ik])
ubkG = ubkG/np.sqrt(a1)*float(NG1) #Normalization
Eg = np.amin(epsbk[Nocc,:])-np.amax(epsbk[Nocc-1,:])
print('Eg = '+str(Eg)+' a.u. = '+str(Hartree*Eg)+' eV')

#Plot the band
def Band_plot():
    for ib in range(4):
        plt.plot(k1,epsbk[ib,:]*Hartree)
    plt.show()
if(not cluster_mode):
    Band_plot()

T = float(NT)*dt
print('========Temporal grid information======')
print ('Time step is '+str(dt)+' in atomic unit and '+str(dt*Atomtime)+' in femtosecond.')
print ('Corresponding energy is {:f} in a.u. and {:f} in eV.'.format(tpi/dt,tpi/dt*Hartree))
print ('')
print ('Number of time step is '+str(NT))
print ('Simulation time is '+str(T)+' in atomic unit and '+str(T*Atomtime)+' in femtosecond.')
print ('Corresponding energy is {:f} in a.u. and {:f} in eV.'.format(tpi/T,tpi/T*Hartree))
print ('')

# Array for real-time construction
tt = np.zeros(NT,dtype='float64')
for it in range(NT):
    tt[it] = it*dt
Jt = np.zeros(NT,dtype='float64')
# Conversion to atomic unit from SI input
omegac = omegac/Hartree #Conversion to atomic unit
E = E/Atomfield #Conversion to atomic unit
Tpulse = Tpulse/Atomtime #Conversion to atomic unit
phiCEP = phiCEP*tpi
Delay = Delay/Atomtime

print('========Laser profile======')
print ('Frequency is '+str(omegac)+' in atomic unit and '+str(omegac*Hartree)+' in eV.')
print ('Corresponding period is {:f} in a.u. and {:f} in femtosecond.'.format(tpi/omegac,tpi/omegac*Atomtime))
print ('Corresponding wave length of light is '+str(ch/(omegac*Hartree))+' in nanometer.')
print ('')
print ('Carrier envelope phase (CEP) is defined as f(t)*cos(w(t-Tpeak) + phi), f is an envelope function peaked at t=Tpeak.')
print ('The CEP value is {:f} in a unit of tpi.'.format(phiCEP/tpi))
print ('')
print ('The field strength E1 is '+str(E)+' in atomic unit and '+str(E*Atomfield)+' in V/nm.')
print ('Corresponding intensity is {:0.6e} in W/cm^2 '.format(E**2*halfepsc))
print ('(E x a) is '+str(E*a1)+' in atomic unit and ' + str(E*a1*Hartree)+ ' in eV.')
print ('')
print ('A parameter for the pulse duraion Tpulse is '+str(Tpulse)+' in atomic unit and '+str(Tpulse*Atomtime)+' in femtosecond.')
print ('Corresponding energy is '+str(tpi/Tpulse)+' in a.u. and '+str(tpi/Tpulse*Hartree)+' in eV.')
print ('')
print ('Number of cycle in a pulse, Tpulse*freqc, is {:f}.'.format(Tpulse/(tpi/omegac)))
print ('')

#Field construction
def Make_AtEt():
    At = np.zeros(NT,dtype='float64')
    Et = np.zeros(NT,dtype='float64')
    A = E/omegac
    envelope = np.zeros(NT,dtype='float64')
    for it in range(NT):
        if ((Delay < tt[it] )and(tt[it] < Tpulse + Delay)):
            envelope[it] = (np.cos(pi/(Tpulse)*(tt[it]-Tpulse/2.0 - Delay)))**nenvelope
    At = A*envelope*np.cos(omegac*(tt - Tpulse/2.0 - Delay)+phiCEP) 
    for it in range(1,NT-1):
        Et[it] = -(At[it+1] - At[it-1])/2.0/dt
        Et[0] = 2.0*Et[1] - Et[2]
        Et[NT-1] = 2.0*Et[NT-2] - Et[NT-3]
    return At,Et
At, Et = Make_AtEt()
def AtEt_plot():
    plt.xlabel('fs')
    plt.plot(tt*Atomtime,At)
    plt.plot(tt*Atomtime,b1*np.ones(NT)*b1/2.0)
    plt.plot(tt*Atomtime,-b1*np.ones(NT)*b1/2.0)
    plt.show()
    plt.xlabel('fs')
    plt.ylabel('V/nm')
    plt.plot(tt*Atomtime,Et*Atomfield)
    plt.show()
if (not cluster_mode):
    AtEt_plot()
print('Etmax= '+str(np.amax(np.abs(Et)))+' a.u.= '+str(np.amax(np.abs(Et))*Atomfield)+' V/nm')
print('Atmax= '+str(np.amax(np.abs(At)))+' a.u.= '+str(np.amax(np.abs(At))/b1)+' b1')
print('Approximated Apek - Avalley = '+str(np.amax(At)-np.amin(At))+' a.u.= '+str((np.amax(At)-np.amin(At))/b1)+' b1')
print('a1*Etmax= '+str(a1*np.amax(np.abs(Et)))+' a.u.')
print('Eg= '+str(Eg)+' a.u.')


#sys.exit()
#Relevant functions
def occbkubkG_dns(occbkloc,ubkGloc):
    dnsloc = np.zeros(NG1,dtype='float64')
    work = np.empty_like(ubkGloc[:,0,0])
    NBactloc = np.shape(ubkGloc)[1]
    for ik in range(NK1):
        for ib in range(NBactloc):
            work = np.fft.ifft(ubkGloc[:,ib,ik])
            dnsloc = dnsloc + occbk[ib,ik]*(np.abs(work))**2
    return dnsloc
dns = occbkubkG_dns(occbk,ubkG)
print('Check for dns, '+str(np.sum(dns)*H1))

def occbkubkG_J(occbkloc,ubkGloc,Aloc): #Exact formula should be checked=========
    Jloc = 0.0
    for ik in range(NK1):
        kloc = k1[ik] + Aloc
        for ib in range(NG1):
            Jloc = Jloc + occbk[ib,ik]*(np.sum(G1[:]*(np.abs(ubkGloc[:,ib,ik]))**2)*a1/float(NG1**2)+kloc)
    return Jloc/a1
J = occbkubkG_J(occbk,ubkG,0.0) #Matter current
print('Check for current, '+str(J))

def occbkubkG_Etot(occbkloc,ubkGloc,Aloc): #Exact formula should be checked=========
    Etotloc = 0.0
    hkloc = Make_hk(0.0)
    for ik in range(NK1):
        hubGloc = np.dot(hkloc[:,:,ik],ubkGloc[:,:,ik])
        for ib in range(NG1):
            Etotloc = Etotloc + occbk[ib,ik]*np.real(np.vdot(ubkGloc[:,ib,ik],hubGloc[:,ib]))
    return Etotloc*a1/float(NG1**2) #orbital function is normalized to give correct number of particle in the cell.
Etot = occbkubkG_Etot(occbk,ubkG,0.0)
print('Check for Etot, '+str(Etot))

def h_U(h):
    eigs, coef = np.linalg.eigh(h*dt)
    U = np.exp(-zI*eigs[0])*np.outer(coef[:,0],np.conj(coef[:,0]))
    for ib in range(1,NG1):
        U = U + np.exp(-zI*eigs[ib])*np.outer(coef[:,ib],np.conj(coef[:,ib]))
    return U

#Time-evolution
U = np.zeros([NG1,NG1],dtype='complex128') 
print('################################')
print('Time-evolution starts.          ')
for it in range(NT):
    Jt[it] = occbkubkG_J(occbk,ubkG,At[it])
    hk = Make_hk(At[it])
    for ik in range(NK1):
        U = h_U(hk[:,:,ik])
        ubkG[:,:,ik] = np.dot(U,ubkG[:,:,ik])
    if(it%1000 == 0):
        dns = occbkubkG_dns(occbk,ubkG)
        Etot = occbkubkG_Etot(occbk,ubkG,At[it])
        print(it,np.sum(dns)*H1, Jt[it], Etot)
print('Time-evolution ends.            ')
print('################################')
np.savez(sys_name+'Jt.npz',tt=tt,Jt=Jt)
if(write_ASCII):
    f = open(sys_name+'Jt.out','w')
    f.write('# tt[it], Jt[it], Et[it], At[it]  (it=1,NT) in atomic unit \n')
    for it in range(NT):
        f.write(str(tt[it])+'  '+str(Jt[it])+'  '+str(Et[it])+'  '+str(At[it])+'\n')
    f.close()

if(not cluster_mode):
    plt.xlabel('fs')
    plt.plot(tt*Atomtime,Jt)
    plt.show()
        


#Taking filter in real-time
omega = np.fft.fftfreq(NT)*(tpi/dt)
envelope = 1.0 - 3.0*(tt/T)**2 + 2.0*(tt/T)**3
Jomega = np.fft.fft(envelope*Jt)
envelope = np.zeros(NT,dtype='float64')
for it in range(NT):
    if ((Delay < tt[it] )and(tt[it] < Tpulse + Delay)):
        envelope[it] = (np.cos(pi/(Tpulse)*(tt[it]-Tpulse/2.0 - Delay)))**nenvelope
Jt_filter = envelope*Jt
Jomega_filter = np.fft.fft(Jt_filter)


print('Fourier transformation ends.    ')
print('################################')

def Jomega_plot():
    plt.xlabel('Harmonic order')
    plt.xlim(0,500.0)
    plt.yscale('log')
    plt.plot(omega[:NT//2]/omegac,np.abs(Jomega[:NT//2]),label='J(w)')
    plt.plot(omega[:NT//2]/omegac,np.abs(Jomega_filter[:NT//2]),label='J_filter(w)')
    plt.legend()
    plt.show()
    plt.xlabel('eV')
    plt.xlim(0,80.0)
    plt.ylim(1.0e-15,10.0)
    plt.yscale('log')
    plt.plot(omega[:NT//2]*Hartree,np.abs(Jomega[:NT//2]),label='J(w)')
    plt.plot(omega[:NT//2]*Hartree,np.abs(Jomega_filter[:NT//2]),label='J_filter(w)')
    plt.legend()
    plt.show()
if(not cluster_mode):
    Jomega_plot()

def Gabor_transform(twidth,Nshift):
    JGaboromega = np.zeros([NT,Nshift],dtype='complex128')
    tshift = np.zeros(Nshift,dtype='float64')
    dtshift = (T+20.0*twidth)/float(Nshift)
    for itshift in range(Nshift):
        tshift[itshift] = dtshift*itshift - 10.0*twidth
        Jt_filter = np.exp(-(tt-tshift[itshift])**2/twidth**2)*Jt
        JGaboromega[:,itshift] = np.fft.fft(Jt_filter)
    return tshift, JGaboromega

def Gaboranalys_plot(twidth,Nshift,emax):
    print('twidth ='+str(twidth)+' a.u. = '+str(twidth*Atomtime)+' fs')
    print('2pi/twidth ='+str(tpi/twidth)+' a.u. = '+str(tpi/twidth*Hartree)+' eV')
    tshift, JGaboromega = Gabor_transform(twidth,Nshift)
    plt.xlabel('fs')
    plt.ylabel('eV')
    plt.ylim(0.0,emax)
    plt.contourf(tshift*Atomtime,omega[:NT//2]*Hartree,np.log10(np.abs(JGaboromega[:NT//2,:])+1.0e-18),50,cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
if(not cluster_mode):
    twidth = 40.0
    Nshift = 1000
    emax = 40.0
    Gaboranalys_plot(twidth,Nshift,emax)


print('Everything is done properly.    ')
print('################################')

#np.savez('test.npz',tshift=tshift,omega=omega,JGaboromega=JGaboromega)
