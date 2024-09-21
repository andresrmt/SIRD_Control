from cycler import cycler
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from random import choices
from matplotlib import gridspec as Gs
import math
import random


num=2
endpt=int(365.25*num) #length of r0l=620 (for sg); 612 for sweden
rateofrecovery=1/8 #as the infectious period is ~7days, thus rate of recovery from being infectious is 1/7
sigma = 1/5.6
rateofreinfection=1/400
#sg (21 jan to 19 nov)
translation = 0 #37 as the zeroth day is March 1st 2020
nonstr_index=[0.8056,0.8056,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6944,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.6389,0.5463,0.5463,0.5463,0.5463,0.3796,0.2315,0.2315,0.1759,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.1759,0.2315,0.2315,0.2315,0.2315,0.2315,0.1759,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2315,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.2685,0.287,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.3426,0.4352,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4352,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4352,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4352,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4352,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4352,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4352,0.4907,0.4907,0.5463,0.5463,0.4907,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.4907,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.4907,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.4907,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.5463,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4722,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4167,0.4537,0.4537,0.4537,0.4537,0.4537,0.4537,0.4537,0.4537,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.5278,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.4907,0.5278,0.5278,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556,0.5556][translation:]
#The data is taken from https://ourworldindata.org/grapher/covid-stringency-index?tab=chart&country=~SGP, and each element is 1-stringency index
d=0.0000093666 #natural death rate- birth rate per day 
mu=0.0005 #as death rate per day of covid cases is very low, 0.25%
#all these are observed by randomly sampling worldometer's new cases and new death and new recoveries
N=5850000 #UN, mid 2020
a=1/5.6 #latency period is on avg, 5.6 days (4-9 days) for non delta variant; delta do be arnd 4days
# According to the lancet, the viral shedding peaks in the first week of symptom onset so 7+(on avg)#b=1.2*((mu+gamma)*(mu+a))/a
#recovery per day of covid is ~0.86 - ~1.06..?
#Rnought=(a/(mu+a))*(b/(mu+gamma))#assume 90% prob of exposed->I #why the hell does e=0????!
dpertime=1/12 #rate for a person to die
sg_avg_hhsize=3.22#singstat, 2020; This leads to the assumption that all hhsizes are 3.22, rather than a distri
#first case entering sg (confirmed) is on 23rd jan (2020 covid year = 343 days)
gamma=rateofrecovery*(1-mu) #technically gamma is rateofrecovery, but since idw to create a new variable, for gamma*prob of not dying...
vaccsum=0; vl=[[0.0,0.0,0.0]];patient0s=1;imml=[];newC=[];newD=[];El=[];drl=[];gammal=[];bl=[[gamma,gamma]]
o=np.random.rand(endpt);time = [i for i in range(endpt+1)]; t = np.array(time)
y0=((N-patient0s)/N,  patient0s/N, 0.0,0.0,0.0) #NEEDS TO BE UPDATED!!!
y0_ld=y0;output=[y0];output_ld=[y0_ld]; Nl=[N,N,N]
deaths=[0,1,2,3];weight=[1-mu-mu**2-mu**3,mu,mu**2,mu**3]
r0log=[]
r0len=len(r0log);r0l=r0log.copy();r0l_ld=r0log.copy()
protl=[]
#btw r0nb
#=1 doesnt rly cause an increase in a non-random scenario, r0nbvl_eq<1 if its random
lockdownperiod=[]
psi_V0=0.93
psi_B0=psi_V0 
class protection:
    def __init__(self,b,v,t0):
        self.v=v
        self.b=b
        self.t0=t0
    def netpsi(self,t):
        psi_V=1-psi_V0*0.5**((t-self.t0)/300)
        psi_B=1-psi_B0*0.5**((t-self.t0)/500)
        return psi_V*(self.v-self.b)+psi_B*self.b #as v is inclusive of b

def r0nb(r0l,s,const=1):
    if k<r0len:
        return r0l, r0l[k]*(gamma)#+o[k]/20-1/40
    if abs(const)>1 or const==0:#root out large const
        const=1
    r0l.append((s/((N-patient0s)/N))*setr0(k)*const)
    return r0l, r0l[k]*(gamma)#+o[k]/20-1/40

def setr0(k,rstart=2.79,rend=5.08,f=0.15,k0=445-translation): #396 - 494 #f=0.0554? 
    k0+=math.exp(1)/f
    if k<=900:
        return (rstart-rend)/(1+math.exp(-f*(-k+k0)))+rend #396
    return 5.08

def newDeaths(deaths,weight,newC,m,ld=0):
    newC=np.array(newC)
    try:
        if sum(newC[len(newC)-5:len(newC),ld])>0:
            if round(mu*m[e][2]*Nl[e],0)==0:
                return choices(deaths,weight)[0]
            else:
                return round(mu*m[e][2]*Nl[e],0)
        else:
            return 0
    except IndexError:
        return 0

def setmu(e):
    if e>=433:
        return 1.54*0.999**e*0.0035#mu decreasing in a not so accurate manner
    else:
        return 0.0035

def lat(u=0,aend=1/4,astart=1/5.6,f=0.4,u0=445-translation):
    u0+=math.exp(1)/f
    if u<=800:
        return (astart-aend)/(1+math.exp(-f*(-u+u0)))+aend #transition from predelta to delta (Apr 1, day 396 is first day delta variant is detected)
    else:
        return aend

def vaccrate(k):#600-translation: day when booster vaccination campaign began
    #sth i assumed to be based off https://covidvax.live/location/sgp and desmos (21st june, sg passed 50% vacc rate, but since initial vaccing is slower than Jun - Sep, the graph is translated forward by 10days from 515)
    if k<800-translation:
        return 0.86/(math.cosh((k-514+translation)/80)**2*80*2) #525 og #+ #assume +230 based off desmos, cuz no real data yet (booster); 
    return 0
    #3rd one to observe whether it is ok to go endemic with regular shots

def boostrate(k):
    if k>=680-translation and k<730-translation:
        return 0.00363636364*(k-700+translation)
    elif k>730-translation:
        return 0.65/(math.cosh((k-800+translation)/80)**2*80*2)
    return 0
def ld_check(k):
    if k>10:#randomly set threshold
        return (output_ld[k][2]/(output_ld[k-1][2]+0.0000000000001)<1.2 and output_ld[k][2]*N<1000) #remove zeruh error
    return True

def seir(y,t,b,dpertime):
    s, e, i, r, v = y
    sigma = lat(u=k) # r0=1 gives u linear R graph, tallies with expectations! (assuming no randomness)
    mu=setmu(k)
   #95% according to https://www.yalemedicine.org/news/covid-19-vaccine-comparison, pfizer vaccine provide 95% protection
    vp = lat(u=k,aend=0.07,astart=0.05)#vaccine protection log curve
    return np.array([-b * i * s+rateofreinfection*r-d*s -vaccrate(k)*s +(1/500)*v, #assume 200 days is half life
                     -sigma * e + b * i * s-d*e + b*i*v*vp, 
                     -gamma * i + sigma * e-mu*i*dpertime-d*i,
                     gamma * i-rateofreinfection*r-d*r,
                     vaccrate(k)*s - (1/450)*(v-boostrate(k)*0.3) - b*i*(v-boostrate(k)*0.6)*vp*sum(effl) -d*v]) #0.3,0.6 r assumptions;
    
######
for k in range(endpt):
    effl=[]
    if k>=1:
        y0=tuple(output[len(output)-1].tolist())
        y0_ld=tuple(output_ld[len(output_ld)-1].tolist())
        if ld_check(k) and k<len(nonstr_index):
            r0l_ld,b_ld=r0nb(r0l_ld,output_ld[k][0],nonstr_index[k])
        elif ld_check(k) and k>=len(nonstr_index):
            r0l_ld,b_ld=r0nb(r0l_ld,output_ld[k][0],nonstr_index[len(nonstr_index)-1])#maintain current stringency
        else:
            r0l_ld,b_ld=r0nb(r0l_ld,output_ld[k][0],min(nonstr_index)) #strictest lockdown EVER!
            lockdownperiod.append(k)
        if k<len(nonstr_index):
            r0l,b=r0nb(r0l,output[k][0],nonstr_index[k])
        else:
            r0l,b=r0nb(r0l,output[k][0],0.65)
    else:
        r0l_ld,b_ld=r0nb(r0l_ld,output_ld[k][0],nonstr_index[k])
        r0l,b=r0nb(r0l,output[k][0],nonstr_index[k])
    if k>=360-translation:
        protl.append(protection(boostrate(k),vaccrate(k),k))
        for g in protl:
            effl.append(g.netpsi(k))
    bl.append([b,b_ld])
    u=odeint(seir, y0,np.array([0,1,2]),args=(b,dpertime), rtol=1e-6)[1] #for loop helps us to circumvent the annoying discontinuous solutions sometimes returned by odeint
    output.append(u)
    u_ld=odeint(seir, y0_ld,t,args=(b_ld,dpertime),rtol=1e-6)[1]
    output_ld.append(u_ld)
    if k>1:
        N+=(sum(output[k])-sum(output[k-1]))*N
        Nl.append(N)
    vaccsum+=vaccrate(k)
    vl.append([vaccsum,output[k][4],output_ld[k][4]])

fig=plt.figure()
gs=Gs.GridSpec(8,5,figure=fig)
axseir=fig.add_subplot(gs[6:8,0:5])
ax = fig.add_subplot(gs[0:3,0:2])
ax1 = fig.add_subplot(gs[0,3:5])#axis[x,y] if (n,m), where n,m≥2
axsr=fig.add_subplot(gs[2,3:5])
ax6=fig.add_subplot(gs[4,0:2])
ax7=fig.add_subplot(gs[4,3:5])
m=np.array(output.copy())
m_ld=np.array(output_ld.copy())
q=[];sr=[];
for e in range(len(m[:,1])):
    a=lat(u=e)
    mu=setmu(e)
    newC.append([round(m[e][1] * Nl[e],0),round(m_ld[e][1] * Nl[e],0)])
    weight=[1-mu-mu**2-mu**3,mu,mu**2,mu**3]
    newD.append([newDeaths(deaths,weight,newC,m),newDeaths(deaths,weight,newC,m_ld,1)])
    drl.append(mu)
    q.append([m[e][1]*Nl[e],m[e][2]*Nl[e],m_ld[e][1]*Nl[e],m_ld[e][2]*Nl[e]])
    sr.append([m[e][0]*Nl[e],m[e][3]*Nl[e],m_ld[e][0]*Nl[e],m_ld[e][3]*Nl[e]])
newC=np.array(newC)#newC is without any lockdown
newD=np.array(newD)
drl=np.array(drl)
cyclerC = cycler(linestyle=['-', '--'],
                    color=['red', 'orange'])
ax.set_prop_cycle(cyclerC)
case=ax.plot(t,newC)
ax.legend(case,['New cases','New cases (Lockdown)'])
ax.axvline(x=366-translation-21,linestyle='--')
ax.axvline(x=731-translation-21,linestyle='--')
cyclerD = cycler(linestyle=['-', '--'],color=['purple', 'magenta'])
ax1.set_prop_cycle(cyclerD)
death=ax1.plot(t,newD)
ax1.axvline(x=366-translation-21,linestyle='--')
ax1.axvline(x=731-translation-21,linestyle='--')
#ax1.legend(death,['New deaths','New deaths (Lockdown)'])
#cyclersr = cycler(linestyle=['-', '-', '--', '--'], color=['cyan', 'green','skyblue', 'lawngreen'])
efflsum=[0 for i in range(362)]
for i in range(1,len(effl)):
    efflsum.append(sum(effl[:i]))
curvesr=axsr.plot(t,efflsum)
axsr.axvline(x=366-translation-21,linestyle='--')
axsr.axvline(x=731-translation-21,linestyle='--')
#axsr.legend(curvesr, ['Susceptible', 'Recovered','Susceptible (Lockdown)', 'Recovered (Lockdown)'])
ax6.plot(t,np.array([gamma for i in range(endpt+1)]),"k--")#line of no increase
cyclerb=cycler(linestyle=['-','--'], color=['red','pink'])
ax6.set_prop_cycle(cyclerb)
ax6.axvline(x=366-translation-21,linestyle='--')
ax6.axvline(x=731-translation-21,linestyle='--')
bl=np.array(bl)
ax6.plot(t,bl)
vl=np.array(vl)
cyclerv=cycler(linestyle=['-.','-','--'], color=['black','green','lightgreen'])
ax7.set_prop_cycle(cyclerv)
ax7.plot(t,vl)
ax7.axvline(x=366-translation-21,linestyle='--')
ax7.axvline(x=731-translation-21,linestyle='--')
cyclerseir = cycler(linestyle=['-', '-', '--', '--'],
                    color=['blue', 'red','aqua', 'tomato'])
axseir.set_prop_cycle(cyclerseir)
curve = axseir.plot(t, np.array(q))
axseir.legend(curve, ['Exposed', 'Infected','Exposed (Lockdown)', 'Infected (Lockdown)'])
axseir.axvline(x=366-translation-21,linestyle='--')
axseir.axvline(x=731-translation-21,linestyle='--')
ax.set_ylabel('no. of ppl')
ax.set_title('new cases')
ax1.set_ylabel('no. of ppl')
ax1.set_title('new deaths')
axseir.set_ylabel('no. of ppl')
axseir.set_title('SEI(R/D)VS Model ('+str(num)+' years)')
axsr.set_ylabel('no. of ppl')
ax7.set_title("vaccinated individuals (efficacy)")
ax6.set_title("avg contacts/ day")
plt.show()
