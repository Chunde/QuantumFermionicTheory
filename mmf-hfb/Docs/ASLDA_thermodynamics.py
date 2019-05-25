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

from mmf_hfb import vortex_aslda
from mmf_hfb.xp import xp  # np
from mmf_hfb import homogeneous
import pytest
from mmf_hfb.xp import xp 


# # Test the ASLDA thermodynamics

def test_aslda_thermodynamic(dx=1e-3):
    L = 0.46
    N = 4
    N_twist = 32
    delta = 1.0
    mu=10
    dmu = 2.1
    v_0, n, _, e_0 = homogeneous.Homogeneous1D().get_BCS_v_n_e(
        delta=delta, mus_eff=(mu+dmu, mu-dmu))
    b = vortex_aslda.BDG(T=0, Nxyz=[N,N], Lxyz=[L,L])
    k_c = abs(xp.array(b.kxyz).max())
    b.E_c = 3 * (b.hbar*k_c)**2/2/b.m
    def get_ns_e_p(mu, dmu):
        ns, e, p = b.get_ns_e_p(mus_eff=(mu+dmu, mu-dmu), delta=delta, N_twist=N_twist, Laplacian_only=True, max_iter=12)
        return ns, e, p
    ns, e, p = get_ns_e_p(mu=mu, dmu=dmu)
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu)
    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu)
    n_p = (p1-p2)/2/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    print("-------------------------------------")
    print(n_p.max().real, sum(ns).max())
    print(mu_[0].max().real, mu)
    print("-------------------------------------")
    assert xp.allclose(n_p.max().real, sum(ns), rtol=1e-2)
    assert xp.allclose(mu_[0].max().real, mu, rtol=1e-2)


test_aslda_thermodynamic()


