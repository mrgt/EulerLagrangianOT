import sys
sys.path.append('..');
sys.path.append('../../MongeAmpere/PyMongeAmpere-build');
sys.path.append('../../MongeAmpere/PyMongeAmpere-build/lib');
import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.optimize as opt
import multiprocessing
from scipy.spatial import ConvexHull

def distance_point_line(m, n, pt):
    u = n - m
    Mpt = pt - m
    norm_u = np.linalg.norm(u)
    dist = np.linalg.norm(Mpt - (np.inner(Mpt,u)/(norm_u*norm_u))*u)
    return dist

# compute an admissible dual variable for the optimal transport
# problem between Y and dens, assuming that dens does not vanish on
# the convex hull of its vertices
def estimate_dual_variable(dens, Y):
    X = dens.vertices
    bX = np.mean(X, 0)
    bY = np.mean(Y, 0)

    # compute radius of Y wrt bY 
    rY = np.max(np.linalg.norm(Y - np.tile(bY,(Y.shape[0],1)), axis=1))

    # search the largest inscribed circle in X centered on bX
    chX = ConvexHull(X)
    points = chX.points
    simplices = chX.simplices
    rX = float('inf')
    for simplex in simplices:
        d = distance_point_line(points[simplex[0]], points[simplex[1]], bX)
        rX = min(d,rX)

    ratio = min(rX / rY,1.0)
    psi_tilde0 = (0.5 * ratio * (np.power(Y[:,0]-bY[0],2)+
                                np.power(Y[:,1]-bY[1],2)) +
                  bX[0]*Y[:,0] + bX[1]*Y[:,1])
    psi0 = np.power(Y[:,0],2) + np.power(Y[:,1],2) - 2*psi_tilde0
    return psi0
		

# project an ordered point cloud to the L^2-closest "incompressible"
# point cloud, i.e.  compute the solution of the optimal transport
# problem between the density dens and the point cloud Z, and move
# each point to the centroid of the corresponding Laguerre cell
def project_on_incompressible(dens,Z,verbose=False):
    N = Z.shape[0]
    nu = np.ones(N) * dens.mass()/N
    w0 = estimate_dual_variable(dens,Z)
    w = ma.optimal_transport_2(dens, Z, nu, w0=w0, verbose=verbose)
    return dens.lloyd(Z,w)[0],w

# compute projection on incompressible
# then compute the mass, centroids and second moment of Laguerre cells
def projection_on_incompressible_moments(dens, Z):
    N = Z.shape[0]
    nu = np.ones(N) * dens.mass()/N
    w0= estimate_dual_variable(dens, Z)
    w = ma.optimal_transport_2(dens, Z, nu, w0=w0, verbose=False)
    return dens.moments(Z,w) 

def squared_distance_to_incompressible(dens, s):
    mass,cent,mom = projection_on_incompressible_moments(dens,s)
    # energy = cxx + cyy + |s|^2 mass - 2 (cx sx + cy sy)
    E = sum(mass * (np.power(s[:,0],2) + np.power(s[:,1],2)) +
            mom[:,0] + mom[:,1] - 
            2 * (s[:,0] * cent[:,0] + s[:,1] * cent[:,1]));
    g = 2 * (np.vstack((mass,mass)).T * s - cent);
    return E,g

def sq_dist_to_incompressible(args):
    shape,s = args
    E,g = squared_distance_to_incompressible(ma.Density_2(shape),s)
    return E,g

def euler_partial_energy(shape, S, s0, s1, lbda, parallel_map=None):
    T = S.shape[0]
    N = S.shape[1]
    E = 0
    g = np.zeros((T,N,2))

    # we divide by N in the definition of alpha and gamma, because the
    # L^2 distance should be weighted by a probability measure nb:
    # this is not necessary for beta, because the rescaling is done
    # in squared_distance_to_incompressible
    alpha = float(T)/float(N)   
    beta = lbda
    gamma = lbda/float(N)

    # kinetic energy
    for i in xrange(0,T-1):
        dS = S[i+1,:,:] - S[i,:,:]
        E = E + alpha * np.sum(np.sum(np.power(dS,2)))
        g[i,:,:] = g[i,:,:] - 2 * alpha * dS
        g[i+1,:,:] = g[i+1,:,:] + 2 * alpha * dS

    # the computation of square distances to incompressibility
    # constraints can be optionally parellized
    if parallel_map is None:
        parallel_map = map;
    EG = parallel_map(sq_dist_to_incompressible, zip([shape for i in xrange(1,T-1)],
                                                     [S[i] for i in xrange(1,T-1)]))

    # penalization of incompressibility constraint
    for i in xrange(1,T-1):
        Ed,gd = EG[i-1]
        E = E + beta * Ed 
        g[i,:,:] = g[i,:,:] + beta * gd
    
    # penalization of boundary conditions
    bc0 = S[0,:,:] - s0
    bc1 = S[T-1,:,:] - s1
    E = E + gamma * (np.sum(np.sum(np.power(bc0,2))) +
                     np.sum(np.sum(np.power(bc1,2))))
    g[0,:,:] = g[0,:,:] + 2 * gamma * bc0
    g[T-1,:,:] = g[T-1,:,:] + 2 * gamma * bc1

    # to save intermediary solutions, uncomment next line
    # euler_save("/tmp/euler-save-temp.npz", shape=shape, S=S)

    return E,g


def convert_shape(F,S,shape):
    N = len(S)
    E,g = F(np.reshape(S,shape))
    return E, np.reshape(g, len(S))

from contextlib import closing

def euler_solve_lbfgs_step(shape, S0, s0, s1, lbda, pgtol=1e-10):
    with  closing(multiprocessing.Pool(processes=2)) as pool:
        T = S0.shape[0]
        N = S0.shape[1]
        parallel_map = lambda f,v: pool.map(f,v)
        F = lambda S: euler_partial_energy(shape, S, s0, s1, lbda,
                                           parallel_map)
        Fc = lambda S: convert_shape(F,S,(T,N,2))
        S,f,d = opt.fmin_l_bfgs_b(Fc,np.reshape(S0,T*N*2),
                                  iprint=1, pgtol=pgtol, factr=10, m=20);
        pool.terminate()
        return np.reshape(S,(T,N,2))

def euler_solve_lbfgs(shape, X, Y, p, k=2, nsmooth=4, sigma=0):
    N = X.shape[0]
    T = np.power(2,k)+1;
    h = 1.0/np.sqrt(N); # 1/h^2 = N

    S0 = np.zeros((2,N,2))
    S0[0] = X;
    S0[1] = Y;
    
    for j in xrange(k):
        T0 = S0.shape[0]
        T = 2*T0-1
        S = np.zeros((T,N,2))

        print "\nADDING TIMESTEPS (T=%d)" % T
        for i in xrange(T0-1):
            S[2*i] = S0[i]
            mean = (S0[i] + S0[i+1])/2

            # when adding the first intermediate timestep, it might be
            # necessary to perturb the mean of the two point clouds
            # (this happens for the disk inversion, because in this
            # case mean == 0)
            if T0 == 2:
                mean = mean + sigma*np.random.randn(N,2)
            # initial guess for the inserted timesteps
            S[2*i+1] = project_on_incompressible(ma.Density_2(shape), mean, verbose=True)
        S[T-1] = S0[T0-1]
        S0 = S

        print "\nLBFGS OPTIMIZATION (T=%d)" % T
        lbda = 1/np.power(h, p)
        S0 = euler_solve_lbfgs_step(shape,S0,X,Y,lbda)

    return S0

import os
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
def euler_save(fname, **kwargs):
    f = open(fname, "wb")
    np.savez_compressed(f,**kwargs)
    f.close()

def euler_load_experiment(fname):
    f = open(fname, "rb")
    v = np.load(f)
    shape = v['shape']
    X = v['X']
    Y = v['Y']
    f.close()
    return shape, X, Y

def euler_load_result(fname):
    f = open(fname, "rb")
    v = np.load(f)
    shape = v['shape']
    S = v['S']
    Sproj = v['Sproj']
    f.close()
    return shape, S, Sproj

# display code


def embed_solution_into_H1(S):
    T = S.shape[0]
    N = S.shape[1]
    Sflat = np.zeros((N, T*2 + (T-1)*2))
    for i in xrange(N):
        Sflat[i,0:T*2] = np.reshape(S[:,i,:], (T*2))
    for i in xrange(N):
        Sflat[i,T*2:T*2+(T-1)*2] = T*np.reshape(S[1:T,i,:]-S[0:(T-1),i,:], ((T-1)*2))
    return Sflat

def bounding_box(X):
    xm = np.amin(X[:,0])
    xM = np.amax(X[:,0])
    ym = np.amin(X[:,1])
    yM = np.amax(X[:,1])
    return np.array([xm,xM, ym, yM])

def cut_vertically(X):
    bb = bounding_box(X)
    Y = X.copy()
    Y[:,0] = (Y[:,0] - bb[0])/(bb[1]-bb[0])
    Y[:,1] = (Y[:,1] - bb[2])/(bb[3]-bb[2])
    ii = np.nonzero(Y[:,0] > .66)
    jj = np.nonzero((Y[:,0] <= .66) & (Y[:,0] > .33))
    kk = np.nonzero(Y[:,0] <= .33)
    return ii,jj,kk


# display
def draw_voronoi_edges(E):
    nan = float('nan')
    N = E.shape[0]
    x = np.zeros(3*N)
    y = np.zeros(3*N)
    for i in xrange(N):
        x[3*i] = E[i,0]
        x[3*i+1] = E[i,2]
        x[3*i+2] = nan
        y[3*i] = E[i,1]
        y[3*i+1] = E[i,3]
        y[3*i+2] = nan
    return x,y

def draw_bbox(bbox):
    nan = float('nan')
    x = np.zeros(12)
    y = np.zeros(12)
    x0 = bbox[0]; x1 = bbox[2];
    y0 = bbox[1]; y1 = bbox[3];
    x[0] = x0; y[0] = y0; 
    x[1] = x1; y[1] = y0;
    x[2] = nan; y[2] = nan;

    x[3] = x1; y[3] = y0; 
    x[4] = x1; y[4] = y1;
    x[5] = nan; y[5] = nan;

    x[6] = x1; y[6] = y1; 
    x[7] = x0; y[7] = y1;
    x[8] = nan; y[8] = nan;

    x[9] = x0; y[9] = y1; 
    x[10] = x0; y[10] = y0;
    x[11] = nan; y[11] = nan;

    return x,y

def sqmom(V):
    return np.sum(V[:,0] * V[:,0] + V[:,1] * V[:,1])

def gen_grid(bbox, N):
    L = (bbox[2] - bbox[0])/2.
    H = (bbox[3] - bbox[1])/2.
    # nn = 2*K*L, mm = 2*K*H, mm*nn = N => 4K^2*L*H = N
    K = np.sqrt(N/(4.*L*H))
    nn = int(np.floor(2*K*L))
    mm = int(np.floor(2*K*H))
    x, y = np.meshgrid(np.linspace(bbox[0],bbox[2],nn),
                       np.linspace(bbox[1],bbox[3],mm))
    N = int(np.floor(mm*nn))
    X = np.vstack((np.reshape(x,N,1),np.reshape(y,N,1))).T
    return X,N

def perform_euler_simulation(X, V, nt, dt, bname, force, energy, plot, integrator):
    ensure_dir(bname)
    X,A,P,w = force(X)
    energies = np.zeros((nt,1))
    for i in xrange(nt):
        print(i)
        plot(X, P, w, '%s/%03d.png' % (bname, i))
        
        if integrator == "vv":
            # Velocity-Verlet integrator
            W = V + 0.5*A*dt # V(t+dt/2)
            X = X + W*dt     # X(t+dt)
            X,A,P,w = force(X) # A(t+dt)
            V = W + 0.5*A*dt # V(t+dt)
        else:
            V = V + A*dt # V(t+dt) = V(t) + dt A(t)
            X = X + V*dt # X(t+dt) = X(t) + dt V(t+dt)
            X,A,P,w = force(X)

        energies[i,:] = energy(X,P,V)
        print energies[i,:]
    np.savetxt('%s/energies.txt' % (bname), energies, delimiter=",")

    
