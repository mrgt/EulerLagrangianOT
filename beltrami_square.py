import matplotlib
import matplotlib.pyplot as plt
from EulerCommon import *
import pylab, os

# N = nb de points
# eps = eps en espace
# t = temps maximum
# nt = nombre de pas de temps
def euler_square_article_experiment(n=50, eps=.1, t=1., nt=10, vistype='cells'):
    square=np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]]);
    bbox = [-0.5, -0.5, 0.5, 0.5]
    dens = ma.Density_2(square);
    x, y = np.meshgrid(np.linspace(-0.5,0.5,n),
                       np.linspace(-0.5,0.5,n))
    N = n*n
    #X = np.vstack((np.reshape(x,N,1),np.reshape(y,N,1))).T + 1e-5*np.random.rand(N,2)
    X = ma.optimized_sampling_2(dens,N,niter=5);
    a = -.5+.33
    b = -.5+.66
    ii = np.nonzero(X[:,0] > b);
    jj = np.nonzero((X[:,0] <= b) & (X[:,0] > a));
    kk = np.nonzero(X[:,0] <= a);
    colors = np.zeros((N, 3))
    colors[ii,0] = 1.
    colors[jj,1] = 1.
    colors[kk,2] = 1.

    def plot_timestep(P, X, w, colors, bbox, fname, vistype='cells'):
        if (vistype == 'cells'):
            plt.cla()
            plt.scatter(P[ii,0], P[ii,1], s=50, color='red');
            plt.scatter(P[jj,0], P[jj,1], s=50, color='yellow');
            plt.scatter(P[kk,0], P[kk,1], s=50, color='blue');
        
            E = dens.restricted_laguerre_edges(X,w)
            x,y = draw_voronoi_edges(E)
            plt.plot(x,y,color=[0.5,0.5,0.5],linewidth=0.5,aa=True)

            x,y = draw_bbox(bbox)
            plt.plot(x,y,color=[0,0,0],linewidth=2,aa=True)

            ee = 1e-2
            plt.axis([bbox[0]-ee, bbox[2]+ee, bbox[1]-ee, bbox[3]+ee])
            ax = pylab.gca()
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
            plt.pause(.1)
            pylab.savefig(fname, bbox_inches='tight', pad_inches = 0)
        else:
            img = ma.laguerre_diagram_to_image(dens,X,w, colors, bbox, 500, 500)
            img.save(fname)


    def force(X):
        P,w = project_on_incompressible(dens,X)
        return 1./(eps*eps)*(P-X), P, w

    def sqmom(V):
        return np.sum(V[:,0] * V[:,0] + V[:,1] * V[:,1])

    def energy(X,P,V):
        return .5 * sqmom(V)/N + .5/(eps*eps) * sqmom(X-P)/N

    # ====================
    # CALCUL DE LA SOLUTION
    pi = np.pi;
    dt = t/nt
    V = np.zeros((N,2))
    V[:,0] = -np.cos(pi*X[:,0]) * np.sin(pi*X[:,1])
    V[:,1] = np.sin(pi*X[:,0]) * np.cos(pi*X[:,1])
    bname="results/beltrami-square/RT-N=%d-tmax=%g-nt=%g-eps=%g" % (N,t,nt,eps)
    ensure_dir(bname)

    A,P,w = force(X)
    for i in xrange(nt):
        print i
        plot_timestep(P, X, w, colors, bbox, '%s/%03d.png' % (bname, i), vistype=vistype)
        #plt.pause(.1)
        if False:
            # Velocity-Verlet integrator
            W = V + 0.5*A*dt # V(t+dt/2)
            X = X + W*dt     # X(t+dt)
            A,P,w = force(X) # A(t+dt)
            V = W + 0.5*A*dt # V(t+dt)
        else:
            V = V + dt*A
            X = X + dt*V
            A,P,w = force(X)
            
        print energy(X,P,V)
        
    os.system("convert -delay %d -loop 0 %s/*.png %s/movie.gif" % (np.floor(100*dt), bname, bname))

euler_square_article_experiment(n=30, eps=.1, t=1., nt=50, vistype='cells')
#euler_square_article_experiment(n=70, eps=.1, t=1., nt=50, vistype='filled')
