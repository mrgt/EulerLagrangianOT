from EulerCommon import *
import pylab, os, argparse

N = 500; t = 2.; eps = 0.1; nt = 50 # small testcase
#N = 50000; t = 2.; eps = 0.002; nt = 2000
integrator = "euler"; symtype = "sym";

H = 3.; L=1. # hauteur/largeur
rhot = 3; rhob = 1; # density of the top/bottom liquid
ee = .2 # frontier between the light and heavy fluid: y = eps cos(kx)
square=np.array([[-L,-H],[L,-H],[L,H],[-L,H]]);
bbox = np.array([-L, -H, L, H])
dens = ma.Density_2(square);

#============== generation of initial points ==============
if symtype == "sym":
    X,N = gen_grid(bbox,N)
else:
    symtype = "asym"; X = ma.optimized_sampling_2(dens,N,niter=5)
    
#============== display ==============
It = np.nonzero(X[:,1] > ee*np.cos(np.pi*X[:,0]));
Ib = np.nonzero(X[:,1] <= ee*np.cos(np.pi*X[:,0]));
colors = np.zeros((N, 3)); colors[It,0] = 1.; colors[Ib,1] = 1.

def plot_timestep(X, w, colors, bbox, fname):
    img = ma.laguerre_diagram_to_image(dens,X,w, colors, bbox, int(200*L), int(200*H))
    img.save(fname)

#============== physics ==============
masses = np.zeros((N,2));
masses[It,1] = rhot; masses[Ib,1] = rhob
G0 = 10.; G = -G0 * masses

def force(X):
    P,w = project_on_incompressible(dens,X)
    return X,1./(eps*eps)*(P-X) + G, P, w 
def energy(X,P,V):
    kin  = .5*sqmom(V)/N
    potD = .5/(eps*eps) * sqmom(X-P)/N
    potG = np.sum(G0 * X[:,1]*masses[:,1])/N
    return kin + potD + potG

# ====================
bname="results/rayleigh-taylor-%s/RT-N=%d-tmax=%g-nt=%g-eps=%g-sym=%s-integ=%s" % (symtype,N,t,nt,eps,symtype,integrator)
plot_ts = lambda X,P,w,fname: plot_timestep(X,w,colors,bbox,fname)
perform_euler_simulation(X, np.zeros((N,2)), nt, dt=t/nt, bname=bname,
                         force=force, energy=energy, plot=plot_ts, integrator=integrator)
