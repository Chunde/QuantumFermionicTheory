
import sys
import numpy as np
import pytest
import os
from os.path import join
import inspect
from mmf_hfb.potentials import HarmonicOscillator
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, join(currentdir, '..', 'QuantumFriction'))
from bcs_cooling import BCSCooling


def Prob(psi):
    return np.abs(psi)**2


@pytest.fixture(params=[64, 225, 128])
def N(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def n(request):
    return request.param


@pytest.fixture(params=[0, 2, 3])
def da(request):
    return request.param


@pytest.fixture(params=[0, 2, 3])
def db(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def N_state(request):
    return request.param


def test_derivative_cooling(n, da, db):
    """
    Test that for free particle hamiltonian, when da=da, Vc is zero
    """
    args = dict(N=128, dx=0.1, beta_0=1, V=0, beta_K=0, beta_V=0, beta_D=1)
    b = BCSCooling(**args)
    k0 = 2*np.pi/b.L
    x = b.xyz[0]
    H0 = b._get_H(mu_eff=0, V=0)
    U0, E0 = b.get_psis_es(H0, transpose=True)
    psi_1 = b.Normalize(np.cos(k0*x))
    assert np.allclose(Prob(psi_1), Prob(b.Normalize(U0[1])))
    assert np.allclose(E0[1], k0**2/2.0)
    psi = np.exp(1j*n*(k0*x))
    E = n**2*k0**2/2
    psi_a = b.Del(psi, n=da)
    Hpsi = np.array(b.apply_H([psi]))[0]/(1j)
    Hpsi_a = b.Del(Hpsi, n=da)

    if da == db:
        psi_b = psi_a
        Hpsi_b = Hpsi_a
    else:
        psi_b = b.Del(psi, n=db)
        Hpsi_b = b.Del(Hpsi, n=db)
    Vc = psi_a*Hpsi_b.conj() + Hpsi_a*psi_b.conj()

    assert np.allclose(Vc, 0, atol=1e-7)
    assert np.allclose(
        psi_a*Hpsi_b.conj(),
        1j*(n*k0)**(da+db)*E*psi*psi.conj()*((-1j)**db)*(1j)**da)
    assert np.allclose(
        Hpsi_a*psi_b.conj(),
        (-1j)*(n*k0)**(da+db)*E*psi*psi.conj()*((-1j)**db)*(1j)**da)


def test_Vd_to_Vc(N):
    """if da=db=0, Vd should equal to Vc"""
    T = 0.5
    args0 = dict(N=N, dx=0.1, divs=(0, 0), beta_D=1, T=T, check_dE=True)
    args1 = dict(N=N, dx=0.1, divs=(0, 0), beta_V=1, T=T, check_dE=True)
    b0 = BCSCooling(**args0)
    h0 = HarmonicOscillator(w=1)
    x = b0.xyz[0]
    V = x**2/2
    psi = h0.get_wf(x, n=2)
    b1 = BCSCooling(**args1)
    b0.V = V
    b1.V = V
    assert np.allclose(b0.get_Vd([psi]), b1.get_Vc([psi]))


def test_apply_Vs(N_state=2):
    """
    check the equivalence of apply_H, apply_V, and apply_K for computing Vc and Kc
    """
    args = dict(
        N=128, dx=0.1, beta_0=1, divs=(1, 1), beta_K=0, beta_V=0, beta_D=1)
    b = BCSCooling(**args)
    # x = b.xyz[0]
    # V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    U0, E0 = b.get_psis_es(H0, transpose=True)
    psis = U0[:N_state]

    def get_kc(psis, fun):
        N = b.get_N(psis)
        Kc = 0
        # can also apply_H, result is unchanged
        Hpsis = fun(psis)  # [check]apply_V or apply_H
        for i, psi in enumerate(psis):
            psi_k = np.fft.fft(psi)*b.dV
            Vpsi_k = np.fft.fft(Hpsis[i])*b.dV
            Kc = Kc + 2*(psi_k.conj()*Vpsi_k).imag/N*b.dV/np.prod(b.Lxyz)
        return Kc

    def get_Vc(psis, fun):
        N = b.get_N(psis)
        Vc = 0
        Hpsis = fun(psis, psis_k=None)
        for i, psi in enumerate(psis):
            Vc = Vc + 2*(psi.conj()*Hpsis[i]).imag/N
        return Vc

    kc1 = get_kc(psis, b.apply_V)
    kc2 = get_kc(psis, b.apply_H)
    assert np.allclose(kc1, kc2)
    vc1 = get_Vc(psis, b.apply_K)
    vc2 = get_Vc(psis, b.apply_H)
    assert np.allclose(vc1, vc2)


def test_dE_dt(N=128, da=0, db=0, T=0.5):
    """check dE/dt, should be always negative"""
    if da > 1 or db > 1:
        return
    args = dict(N=N, dx=0.2, divs=(da, db), T=T, check_dE=True)
    b = BCSCooling(**args)
    b.restore_max_ks()
    h0 = HarmonicOscillator(w=1)
    x = b.xyz[0]
    V = x**2/2
    b.V = V
    psi = h0.get_wf(x, n=2) + 0j
    b.beta_V = 1
    b.beta_D = 0
    # test $V_c$
    b.solve([psi], T=T, rtol=1e-5, atol=1e-6, V=V, solver=None, method='BDF')
    b.beta_V = 0
    b.beta_K = 1
    # test $V_k$
    b.solve([psi], T=T, rtol=1e-5, atol=1e-6, V=V, solver=None, method='BDF')
    b.beta_K = 0
    b.beta_D = 1
    # test $V_d$
    b.solve([psi], T=T, rtol=1e-5, atol=1e-6, V=V, solver=None, method='BDF')
    b.beta_D = 0
    b.beta_Y = 1
    # test $V_y$
    # b.solve([psi], T=T, rtol=1e-5, atol=1e-6, V=V, solver=None, method='BDF')


def test_ImaginaryCooling_with_desired_energy():
    """test energy level setting code"""
    args = dict(N=128, dx=0.1, beta_0=-1j, beta_K=0, beta_V=0)
    s = BCSCooling(**args)
    s.E_stop = 0.75  # target energy should be no less than E0
    x = s.xyz[0]
    V = x**2/2
    s.V = V
    u0 = np.exp(-x**2/2)/np.pi**4
    u0 = u0/u0.dot(u0.conj())**0.5
    u1 = (np.sqrt(2)*x*np.exp(-x**2/2))/np.pi**4
    u1 = u1/u1.dot(u1.conj())**0.5
    psi_0 = s.Normalize(V*0 + 1+0*1j)
    E0 = s.get_E_Ns([psi_0])[0]
    s.E_stop = 0.75*E0
    _, psis, _ = s.solve([psi_0], T=10, rtol=1e-5, atol=1e-6, method='BDF')
    psi_ground = psis[-1]
    E = s.get_E_Ns(psi_ground)[0]
    assert np.allclose(E, s.E_stop, rtol=1e-10)


def test_uv_ir_Vc():
    """test the uv and ir error."""
    dx = 0.1
    base_value = None
    for Nx in [128, 256, 512]:
        uv = BCSCooling(N=256, dx=dx*256/Nx, beta_V=1)
        ir = BCSCooling(N=Nx, dx=dx, beta_V=1)
        for s in [uv, ir]:
            s.g = -1
            x = s.xyz[0]
            s.V = x**2/2
            psi0 = np.exp(-x**2/2.0)*np.exp(1j*x)
            Vc = s.get_Vc(s.apply_H([psi0]))
            if base_value is None:
                base_value = sum(abs(Vc))*s.dx
            else:
                new_value = sum(abs(Vc))*s.dx
                assert new_value > 1e-5
                assert base_value > 1e-5
                assert np.allclose(base_value, new_value, rtol=0.01)


def test_uv_ir_kc():
    """test the uv and ir error."""
    dx = 0.1
    base_value = None
    for Nx in [128, 256, 512]:
        uv = BCSCooling(N=256, dx=dx*256/Nx, beta_K=1)
        ir = BCSCooling(N=Nx, dx=dx, beta_K=1)
        for s in [uv, ir]:
            s.g = -1
            x = s.xyz[0]
            s.V = x**2/2
            psi0 = np.exp(-x**2/2.0)*np.exp(1j*x)
            Kc = s.get_Kc(s.apply_H([psi0]))
            if base_value is None:
                base_value = sum(abs(Kc))*s.dx
            else:
                new_value = sum(abs(Kc))*s.dx
                # assert new_value > 1e-5
                # assert base_value > 1e-5
                assert np.allclose(base_value, new_value, rtol=0.01)


def test_full_Hc():
    """Test the effect of full cooling potential"""
    b = BCSCooling(N=128, dx=0.1)
    x = b.xyz[0]
    V = x**2/2
    b.V = V
    H0 = b._get_H(mu_eff=0, V=V)
    psis0, Es0 = b.get_psis_es(H0, transpose=True)
    H = b._get_H(mu_eff=0, V=0)
    psis, _ = b.get_psis_es(H, transpose=True)
    for n, T in zip([2, 3, 4, 5], [5, 2, 2, 2]):
    
        psis_init = [b.Normalize(psis[i]) for i in range(n)]

        def compute_dy_dt(t, psis, **args):
            psis = b.unpack(y=psis)
            Hc = b.get_Hc(psis=psis)
            Hpsis = (Hc.dot(psis.T)).T
            for i, psi in enumerate(psis):
                Hpsis[i] -= b.dotc(psi, Hpsis[i])/b.dotc(psi, psi)*psi
            return b.pack(np.array(Hpsis)/(1j*b.hbar))
        _, ys, nfev = b.solve(
            psis=psis_init, T=T, dy_dt=compute_dy_dt,
            rtol=1e-5, atol=1e-6, method='BDF')
        E, N = b.get_E_Ns(psis=ys[-1])
        E0 = np.sum(Es0[:n])
        print(f"State Num={n}, E0={E0},E={E},N={N}, nfev={nfev}")
        assert np.allclose(E, E0, rtol=1e-2)


def test_cooling_with_pairing():
    b = BCSCooling(N=64, dx=0.25, beta_V=1, delta=1, mus=(2, 2))
    x = b.xyz[0]
    V0 = x**2/3
    V1 = x**2/2
    H0 = b.get_H(mus_eff=b.mus, delta=b.delta, Vs=(V0, V0))
    U0, _ = b.get_psis_es(H0, transpose=True)
    psi = U0[10]
    b.V = V1
    psis = [psi]
    E_old = 1e10
    for _ in range(100):
        psis = b.step(psis=psis, n=100)
        E, _ = b.get_E_Ns(psis=psis)
        if abs((E - E_old)/E_old) > 1e-2:
            assert E <= E_old
        E_old = E
