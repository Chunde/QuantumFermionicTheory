import numpy as np
from importlib import reload  # Python 3.4+
import numpy as np
from mmf_hfb import homogeneous; reload(homogeneous)
from mmf_hfb import bcs; reload(bcs)
from mmf_hfb.bcs import BCS
from mmf_hfb import vortex_1d_aslda;reload(vortex_1d_aslda)


class ASLDA_(vortex_1d_aslda.ASLDA):
    # a modified class from ASLDA with different alphas which are constant, so their derivatives are zero
    def get_alphas(self, ns=None):
        alpha_a, alpha_b, alpha_p =np.ones(self.Nx), np.ones(self.Nx), np.ones(self.Nx)
        return (alpha_a, alpha_b, alpha_p)       
    
    def _dp_dn(self,ns):
        return (ns[0] * 0, ns[1]*0)


def test_aslda_twisting():
    """[pass]"""
    L = 0.46
    N = 128
    delta = 1.0
    N_twist = 5
    mu_eff = 1.0
    v_0, n, mu, e_0 = homogeneous._get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)

    print("Test twisting validation...")
    b1 = ASLDA_(T=0,Nx=N,Lx = L)
    b2 = ASLDA_(T=0,Nx=N*2,Lx = L*2)
    for i in range(1,10):
        N_twist = i
        ns1, taus1, kappa1 = b1.get_ns_taus_kappa_average_1d(mus=(mu_eff*np.exp(np.ones((N)),), mu_eff*np.ones((N),)), delta=delta*np.ones((N),), N_twist=N_twist*2)
        ns2, taus2, kappa2 = b2.get_ns_taus_kappa_average_1d(mus=(mu_eff*np.exp(np.ones((b2.Nx)),), mu_eff*np.ones((b2.Nx),)), delta=delta*np.ones((b2.Nx),), N_twist=N_twist)
        assert np.allclose(np.concatenate((ns1[0], ns1[0])), ns2[0])
        assert np.allclose(np.concatenate((ns1[1], ns1[1])), ns2[1])
        print(f"Twisting Number={i+1} pass")

if __name__ == "__main__":
    test_aslda_twisting()
