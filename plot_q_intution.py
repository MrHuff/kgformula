import matplotlib.pyplot as plt
from pylab import *
rc('text', usetex=True)
font_size = 24
plt.rcParams['font.size'] = font_size
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['axes.labelsize'] = font_size
if __name__ == '__main__':
    mean = np.array([0.0,0.0])
    a= 1.5
    b = 0.75
    n=2500
    cov = np.array([[a,b],[b,a]])
    pxz = np.random.multivariate_normal(mean,cov,n)
    a= 2.0
    b = 0.0
    cov_2 = np.array([[a,b],[b,a]])

    px = np.random.multivariate_normal(mean,cov_2,n)
    px_scaled = px*0.5

    plt.scatter(pxz[:,0],pxz[:,1],alpha=0.25, marker='.',label=rf'$p(X\mid Z)$')
    plt.scatter(px[:,0],px[:,1],alpha=0.25, marker='.',label=rf'$p(X)$')
    lgnd_1 = plt.legend(prop={'size': 20}, markerscale=6.)
    # lgnd_1.legendHandles[0]._legmarker.set_markersize(20)

    plt.savefig('q_intuion_a.png',bbox_inches = 'tight')
    plt.clf()
    plt.scatter(pxz[:,0],pxz[:,1],alpha=0.25, marker='.',label=rf'$p(X\mid Z)$')
    plt.scatter(px_scaled[:,0],px_scaled[:,1],alpha=0.25, marker='.',label=rf'$q(X)$')
    lgnd_2 = plt.legend(prop={'size': 20}, markerscale=6.)
    # lgnd_2.legendHandles[0]._legmarker.set_markersize(20)

    plt.savefig('q_intuion_b.png',bbox_inches = 'tight')

    plt.clf()
