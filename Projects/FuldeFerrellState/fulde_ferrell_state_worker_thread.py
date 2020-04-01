"""
A function will be called in the vortex.py notebook.
This is due to the fact that the multiple processing
not works as expected in jupyter notebook. By stripe
the thread routine to a separate file will solve the
problem.
"""


def fulde_ferrell_state_solve_thread(obj_mu_dmu_delta_r):
    f, mu, dmu, delta, r = obj_mu_dmu_delta_r
    return f.solve(mu=mu, dmu=dmu, dq=0.5/r, a=0.001, b=2*delta)
