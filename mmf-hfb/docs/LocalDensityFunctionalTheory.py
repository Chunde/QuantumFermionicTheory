# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all

# # Local-density-functional theory for superﬂuid fermionic systems

# * In this notebook, I will try to understand the paper "Local-density-functional theory for superﬂuid fermionic systems" which is implemented using DVR basis

# Start with the energy functional:
# $$
# \mathcal{E}(\mathbf{r})=\alpha \frac{\tau(\mathbf{r})}{2}+\beta \frac{3\left(3 \pi^{2}\right)^{2 / 3} n^{5 / 3}(\mathbf{r})}{10}+\gamma \frac{|\nu(\mathbf{r})|^{2}}{n^{1 / 3}(\mathbf{r})}
# $$

# where:
#
# $$
# n(\mathbf{r})=\sum_{k}\left|v_{k}(\mathbf{r})\right|^{2} + \sum_{k}\left|u_{k}(\mathbf{r})\right|^{2}\\
# \tau(\mathbf{r})=\sum_{k}\left|\nabla v_{k}(\mathbf{r})\right|^{2}+\sum_{k}\left|\nabla u_{k}(\mathbf{r})\right|^{2}
# $$
#
# $$
# \nu(\mathbf{r})=\sum_{k} v_{k}^{*}(\mathbf{r}) u_{k}(\mathbf{r})
# $$

# ## Regularization
# * The kinetic and anomalous densities diverge, one needs renormalization procedure to the pairing gap and for the energy density.
#
# $$
# \mathcal{E}(\mathbf{r})=\alpha \frac{\tau_{c}(\mathbf{r})}{2}+\beta \frac{3\left(3 \pi^{2}\right)^{2 / 3} n^{5 / 3}(\mathbf{r})}{10}+g_{e f f}(\mathbf{r})\left|\nu_{c}(\mathbf{r})\right|^{2}+V_{e x t}(\mathbf{r}) n(\mathbf{r})
# $$
#
# where
#
# $$
# \frac{1}{g_{e f f}(\mathbf{r})}=\frac{n^{1/ 3}(\mathbf{r})}{\gamma}+\Lambda_{c}(\mathbf{r})
# $$
# What's the difference from another expression?
# $$
# \frac{m}{4 \pi \hbar^{2} a}=\frac{1}{g}+\frac{1}{2} -\!\!\!\!\!\!\int \frac{\mathrm{d}^{3} \mathbf{k}}{(2 \pi)^{3}} \frac{1}{\frac{h^{2} k^{2}}{2 m}+\mathrm{i} 0^{+}}
# $$
# The regularized kinetic and anomalous densities are:
# $$
# \tau_{c}(\mathbf{r})=\sum_{E_{k}<E_{c}}\left|\nabla v_{k}(\mathbf{r})\right|^{2} + \left|\nabla u_{k}(\mathbf{r})\right|^{2}, \quad \nu_{c}(\mathbf{r})=\sum_{E_{k}<E_{c}} v_{k}^{*}(\mathbf{r}) u_{k}(\mathbf{r})
# $$

# ## Transfer to matrix form
# Compute $U(r)=\delta \mathcal{E}(r)/\delta n(r)$

# Rewrite each of the the kinetic term as:
# $$
# \tau = \left|\nabla v_{k}(\mathbf{r})\right|^{2}=\nabla v_k(r)\nabla v_k^*(r)
# $$
# Since the kinetic terms do not dependent on the densities, then
# $$
# \frac{\delta \tau}{\delta n}=\frac{\delta (\nabla v \nabla v^*)}{\delta n}=0
# $$
# $$
# \frac{\delta \tau_c}{\delta n}=0
# $$


