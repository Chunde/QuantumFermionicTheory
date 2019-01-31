import hfb_dir_init

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from uncertainties import ufloat
from importlib import reload  # Python 3.4+
import numpy as np
import homogeneous;reload(homogeneous)
import bcs;reload(bcs)
from bcs import BCS
import vortex_1d_aslda;reload(vortex_1d_aslda)
import itertools  
import matplotlib.pyplot as plt

import time
hbar = 1
m = 1
innerTest = False
plt.autoscale(enable=True, axis='both', tight=None)
plt.ion()
fig = None

class ASLDA_(vortex_1d_aslda.ASLDA):
# a modified class from ASLDA with different alphas which are constant, so their derivatives are zero
    def get_alphas(self, ns = None):
        alpha_a,alpha_b,alpha_p =np.ones(self.Nx),np.ones(self.Nx),np.ones(self.Nx)
        return (alpha_a,alpha_b,alpha_p)       
        return super().get_alphas(ns)
    #def _dp_dn(self,ns):
    #    return (ns[0] * 0, ns[1]*0)
    def _dalpha_p_dp(self,p):
        """return dalpha_p / dp"""
        return 0

    def _dalpha_m_dp(self,p):
        """return dalpha_m / dp"""
        return 0
    #def f(self, E, E_c=None):
        #return 1
    #def get_Ks_Vs(self, delta, mus=(0,0), ns=None, taus=None, kappa=0, ky=0, kz=0, twist=0):
    #    alphas = self.get_alphas(ns)
    #    return (self.get_Ks(twist=twist), self.get_modified_Vs(delta=delta,ns=ns,taus=taus,kappa=kappa,alphas=alphas))

def iterate(lda, mudelta, na_avg=0.5, nb_avg=0.5, N_twist=0, lines=None,**kw):
    mu_a, mu_b, na,nb, mu_a_eff, mu_b_eff, delta,taus,kappa = mudelta
    mus = (mu_a_eff, mu_b_eff)
    if na is None:
        na = np.ones(lda.Nx) * na_avg
    if nb is None:
        nb = np.ones(lda.Nx) * nb_avg
    ns_,taus_,kappa_ = lda.get_ns_taus_kappa_average_2d(mus=mus,delta = delta,ns=(na,nb),taus = taus,kappa=kappa,N_twist=N_twist) #lda.get_ns_taus_kappa(H) 
    gx = lda.gx(ns_,taus_,kappa_)
    na_,nb_ = ns_ # the new densities are not used in the iteration, just used for compute new mus
    if lines is not None:
        line1,line2 = lines
        line1.set_ydata(na_.real)
        line2.set_ydata(nb_.real)
        fig.canvas.draw()
        fig.canvas.flush_events()
    nomral_na = na_.mean() / (na_.mean() + nb_.mean())
    nomral_nb = nb_.mean() / (na_.mean() + nb_.mean())
    mu_a = mu_a*(1 + 0.1*(na_avg - nomral_na))
    mu_b = mu_b*(1 + 0.1*(nb_avg - nomral_nb))
    lr = 0.4
    lo = 1 - lr
    v_a,v_b = lda.get_modified_Vs(delta=delta,ns=(na_,nb_),taus=taus_,kappa=kappa_,alphas=lda.get_alphas((na_,nb_)))
    mu_a_eff = mu_a + v_a * nb
    mu_b_eff = mu_b + v_b * na
    #delta = lda.g_eff*kappa_ 
    print(("mu_a=%f\tmu_b=%f\tdelta=%f\tkappa=%f\tna=%f\tnb=%f\ttau_a=%f\ttau_b=%f")%(mus[0].real.max(),mus[1].real.max(),delta.real.max(),kappa_.real.mean(), na_.real.mean(), nb_.real.mean(),taus_[0].real.mean(),taus_[1].real.mean()))
    return (mu_a, mu_b,lo*na + lr*na_,lo*nb + lr*nb_, mu_a_eff, mu_b_eff, delta,taus_,kappa_)



def test_ASLDA_iterate():
    Nx = 4
    Lx = 0.46
    N_twist = 1
    delta = 1.0
    mu_eff = 1.0
    lda = ASLDA_(T=0,Nx=Nx,Lx=Lx)
    k_c = abs(lda.kx).max()
    E_c = (lda.hbar*k_c)**2/2/lda.m # 3 dimension, so E_c should have a factor of 3
    lda.E_c = E_c
    mu = 1# 0.59060550703283853378393810185221521748413488992993*E_c
    delta = 1# 0.68640205206984016444108204356564421137062514068346*E_c
    qT = (mu, mu) +(None,None)+ (mu_eff*np.ones(lda.Nx),)*2 + (delta * np.ones((lda.Nx),), None,None)
    max_iteration = 5
    v_0, n, mu, e_0 = homogeneous.get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)
    x = 0.75
    global fig
    fig = plt.figure()
    ax = fig.add_subplot(111)
    (line1,),(line2,) = ax.plot(lda.x, lda.x, 'r-') ,ax.plot(lda.x, lda.x, 'b-') 
    ax.set_title('Nx=%d Lx=%f N_twist=%d'%(Nx,Lx,N_twist))
    ax.set_ylim(0,1.2)
    while max_iteration > 0:
        qT = iterate(lda=lda,mudelta = qT, N_twist=N_twist,na_avg=1/(1+x), nb_avg=x/(1+x), lines=(line1,line2),abs_tol=1e-2)
        if not innerTest:
            break

if __name__ == '__main__':
    innerTest = True
    test_ASLDA_iterate()