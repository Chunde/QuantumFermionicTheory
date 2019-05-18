# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Simple Vortex in 2D

# + {"init_cell": true}
import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *                # Conveniences like clear_output
# -

# Here we generate some vortices.  These are regularized by fixing the coupling constant $g$ so that the homogeneous system in the box and on the lattice gives a fixed value of $\Delta$ at the specified chemical potential.  The self-consistent solution is found by simple iterations.

# +
from mmf_hfb import bcs, homogeneous;reload(bcs)
from mmfutils.math.special import mstep
from mmfutils.plot import imcontourf

class Vortex(bcs.BCS):
    barrier_width = 0.2
    barrier_height = 100.0
    
    def __init__(self, Nxyz=(32, 32), Lxyz=(10., 10.), **kw):
        self.R = min(Lxyz)/2
        bcs.BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, **kw)
    
    def get_v_ext(self):
        r = np.sqrt(sum([_x**2 for _x in self.xyz[:2]]))
        R0 = self.barrier_width * self.R
        V = self.barrier_height * mstep(r-self.R+R0, R0)
        return (V, V)

class VortexState(Vortex):
    def __init__(self, mu, dmu, delta, **kw):
        Vortex.__init__(self, **kw)
        self.delta = delta
        self.mus = (mu+dmu, mu-dmu)
        self.g = self.get_g(mu=mu, delta=delta)
        x, y = self.xyz
        self.Delta = delta*(x+1j*y)
        
    def get_g(self, mu=1.0, delta=0.2):
        h = homogeneous.Homogeneous(Nxyz=self.Nxyz, Lxyz=self.Lxyz) 
        res = h.get_densities(mus_eff=(mu, mu), delta=delta)
        g = delta/res.nu.n
        return g
    
    def solve(self, tol=0.01, plot=True):
        err = 1.0
        fig = None
        with NoInterrupt() as interrupted:
            display(self.plot())
            clear_output(wait=True)

            while not interrupted and err > tol:
                res = self.get_densities(mus_eff=self.mus, delta=self.Delta)
                Delta0, self.Delta = self.Delta, self.g*res.nu  # ....
                err = abs(Delta0 - self.Delta).max()
                if display:
                    plt.clf()
                    fig = self.plot(fig=fig, res=res)
                    plt.suptitle(f"err={err}")
                    display(fig)
                    clear_output(wait=True)
        
    
    def plot(self, fig=None, res=None):
        x, y = self.xyz
        if fig is None:
            fig = plt.figure(figsize=(20, 10))
        plt.subplot(231)
        imcontourf(x, y, abs(self.Delta), aspect=1)
        plt.title(r'$|\Delta|$'); plt.colorbar()
        
        if res is not None:
            plt.subplot(232)
            
            imcontourf(x, y, (res.n_a+res.n_b).real, aspect=1)
            plt.title(r'$n_+$'); plt.colorbar()
        
            plt.subplot(233)
            imcontourf(x, y, (res.n_a-res.n_b).real, aspect=1)
            plt.title(r'$n_-$'); plt.colorbar()
            
            plt.subplot(234)
            
            j_a = res.j_a[0] + 1j*res.j_a[1]
            j_b = res.j_b[0] + 1j*res.j_b[1]
            j_p = j_a + j_b
            j_m = j_a - j_b
            utheta = np.exp(1j*np.angle(x + 1j*y))
            imcontourf(x, y, abs(j_p), aspect=1)
            plt.title(r'$j_p$'); plt.colorbar()
            plt.quiver(x.ravel(), y.ravel(), j_a.real.T, j_a.imag.T)
            
            plt.subplot(235)
            imcontourf(x, y, abs(j_m), aspect=1)
            plt.title(r'$J_-$'); plt.colorbar()
            plt.quiver(x.ravel(), y.ravel(), j_m.real.T, j_m.imag.T)
        return fig      

v = Vortex(Nxyz=(16, 16))
# -

# $$
#   v_x +\I v_y = v e^{\I\phi}, \\
#   \uvect{\theta} = \frac{\I x - y}{r}\\
#   \uvect{\theta}\cdot \vect{v} = \frac{-yv_x + xv_y}{r} = - \Re(v \uvect{\theta})
# $$

# ## Potential Profile

V = v.get_v_ext()[0]
x, y = v.xyz
imcontourf(x, y, V, aspect=1);plt.colorbar()

# ## Large Gap
# ### Symmetric Vortex $\Delta=5$

v0 = VortexState(mu=10.0, dmu=0.0, delta=5.0, Nxyz=(32, 32))
v0.solve(plot=True)

# ### Polarized Vortex $\Delta=5$

v1 = VortexState(mu=10.0, dmu=2, delta=5.0, Nxyz=(32, 32))
v1.solve(plot=True)

# ## Smaller Gap
# ### Symmetric Vortex $\Delta=1$
# * smaller $\Delta$

v2 = VortexState(mu=10.0, dmu=0, delta=1.0, Nxyz=(32, 32))
v2.solve(plot=True)

# ### Polarized Vortex $\Delta=1$

v3 = VortexState(mu=10.0, dmu=1.2, delta=1.0, Nxyz=(32, 32))
v3.solve(plot=True)

v4 = VortexState(mu=10.0, dmu=0, delta=8, Nxyz=(32, 32))
v4.solve(plot=True)

# ## Compare to FF State
# * The FFVortex will compute FF State data with the same $\mu,d\mu$, and compare the results in plots
# * $\Delta$ should be continuous in a close loop, in rotation frame, the gap should satisfy:
# $$
# \Delta=\Delta e^{-2i\delta qx}
# $$
# with phase change in a loop by $2\pi$, which requires: $\delta q= \frac{1}{2r}$
#

# +
from mmf_hfb import FuldeFerrelState; reload(FuldeFerrelState)
import warnings
warnings.filterwarnings("ignore")

def FFVortex(bcs_vortex, delta=None):
    mu_a, mu_b=bcs_vortex.mus
    if delta is None:
        delta = bcs_vortex.delta
    mu, dmu = (mu_a + mu_b)/2, (mu_a - mu_b)/2
    N = bcs_vortex.Nxyz[0]
    L = bcs_vortex.Lxyz[0]
    k_c = np.sqrt(2)*np.pi*N/L
    args = dict(mu=mu, dmu=dmu, delta=delta, dim=1, k_c=np.inf)
    f = FuldeFerrelState.FFState(fix_g=True, **args)
    rs = np.linspace(0.01,1, 10)
    rs = np.append(rs, np.linspace(1.1, 4, 10))
    ds = [f.solve(mu=mu, dmu=dmu, dq=0.5/_r, a=0.001, b=2*delta) for _r in rs]
    ps = [f.get_pressure(mu_eff=mu, dmu_eff=dmu, delta=d, dq=0.5/r, use_kappa=False) for r, d in zip(rs,ds)]
    ps0 = [f.get_pressure(mu_eff=mu, dmu_eff=dmu, delta=0,q=0, dq=0, use_kappa=False) for r, d in zip(rs,ds)]

    
    r = np.sqrt(sum(_x**2 for _x in bcs_vortex.xyz))
    plt.figure(figsize(16,8))
    plt.subplot(321)
    plt.plot(r.ravel(), abs(bcs_vortex.Delta).ravel(), '+', label="BCS")
    plt.plot(rs, ds, label="TF Completion")
    plt.legend()
    plt.xlabel("r", fontsize=12)
    plt.ylabel(r'$\Delta$', fontsize=12)
    plt.subplot(322)
    plt.xlabel("r", fontsize=12)
    plt.ylabel("Pressure", fontsize=12)
    plt.plot(rs, ps, label="FF State/Superfluid State Pressure")
    plt.plot(rs, ps0,'--', label="Normal State pressure")
    plt.legend()

    na = np.array([])
    nb = np.array([])
    res = bcs_vortex.get_densities(mus_eff=bcs_vortex.mus, delta=bcs_vortex.Delta)
    for i in range(len(rs)):
        na_, nb_ = f.get_densities(delta=ds[i], dq=0.5/rs[i], mu=mu, dmu=dmu)
        na = np.append(na, na_.n)
        nb = np.append(nb, nb_.n)   
    plt.subplot(323)
    n_p = na + nb
    plt.plot(r.ravel(), abs(res.n_a + res.n_b).ravel(), '+', label="BCS")
    plt.plot(rs, n_p, label="TF Completion")
    plt.xlabel("r", fontsize=12), plt.ylabel(r"$n_p$", fontsize=12)
    plt.title("Total Density")
    plt.legend()
    plt.subplot(324)
    n_m = na - nb
    plt.plot(r.ravel(), abs(res.n_a - res.n_b).ravel(), '+', label="BCS")
    plt.plot(rs, n_m, label="TF Completion")
    plt.xlabel("r", fontsize=12), plt.ylabel(r"$n_m$", fontsize=12),plt.title("Density Difference")
    plt.legend()

    ja = np.abs(res.j_a[0] + 1j*res.j_a[1])
    jb = np.abs(res.j_b[0] + 1j*res.j_b[1]) 
    j_p_, j_m_ = ja + jb, ja - jb
    ja = []
    jb = []
    js = [f.get_current(mu=mu, dmu=dmu, delta=d,dq=0.5/r) for r, d in zip(rs,ds)]
    for j in js:
        ja.append(j[0].n)
        jb.append(j[1].n)
    ja, jb = np.array(ja), np.array(jb)
    j_p, j_m = -(ja + jb), ja - jb
    plt.subplot(325)
    plt.plot(r.ravel(), j_p_.ravel(), '+', label="BCS")
    plt.plot(rs, j_m, label="TF Completion")
    plt.xlabel("r", fontsize=12), plt.ylabel(r"$j_p$", fontsize=12),plt.title("Total Current")
    plt.legend()
    plt.subplot(326)
    plt.plot(r.ravel(), j_m_.ravel(), '+', label="BCS")
    plt.plot(rs, j_p, label="TF Completion")
    plt.xlabel("r", fontsize=12), plt.ylabel(r"$j_m$", fontsize=12),plt.title("Current Difference")
    plt.legend()
# -

rs = np.linspace(0.01,1, 10)
rs = np.append(rs, np.linspace(1.1, 4, 10))
mu=10
dmu=2.1
delta=2
args = dict(mu=mu, dmu=dmu, delta=delta, dim=1, k_c=np.inf)
ff = FuldeFerrelState.FFState(fix_g=True, **args)
qs = np.linspace(0, 5, 10)
ds = []
r = 20
def get_P(q):
    d = ff.solve(mu=mu, dmu=dmu, q=q*100, dq=0.5/r, a=0.001, b=1.2*delta)
    print(d)
    ds.append(d)
    P = ff.get_pressure(mu_eff=mu, dmu_eff=dmu, delta=d, q=q, dq=0.5/r, use_kappa=True)
    return P


Ps = [get_P(q) for q in qs]
plt.plot(qs, Ps)

ff.solve(mu=mu, dmu=dmu, q=0, dq=0, a=0.001, b=1.2*delta)

ff = FuldeFerrelState.FFState(fix_g=True, **args)
ds = np.linspace(0.8*delta, delta*2, 40)
gs = [ff.f(mu=mu,dmu=dmu, delta=d, q=0, dq=0.) for d in ds]
plt.axhline(0)
plt.axvline(delta)
plt.plot(ds, gs, "--")

# ### Symmetric case $\Delta=5$

FFVortex(v0,delta=5)

# ### Polarized case $\Delta=5$

FFVortex(v1, delta=5)

# ### Smaller coupling and symmetric case $\Delta=1$

FFVortex(v2, delta=1)

# ### Smaller coupling and polarized case $\Delta=1$

FFVortex(v3, delta=1)


