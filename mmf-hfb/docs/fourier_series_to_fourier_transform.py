# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Fourier Series

# Here I try to derive the Fourier transform(FT) from the Fourier series(FS).
# ## 1D Fourier Series
# Let $f(x)$ be a function defined in the range from $0-L$, then $f(x)$ can be expanded as:

# $$\begin{align}
# f(x)&=\sum_{n=0}^{\infty}a_n cos(\frac{2\pi nx}{L}) + b_nsin(\frac{2\pi nx}{L})\\
# a_0&= \frac{1}{L}\int_0^{L}f(x)dx\\
# a_n&= \frac{2}{L}\int_0^{L}f(x)cos(\frac{2\pi nx}{L})dx\\
# b_n&= \frac{2}{L}\int_0^{L}f(x)sin(\frac{2\pi nx}{L})dx\\
# \end{align}\tag{1}
# $$
# From above identities, it's also obvious that:
# $$
# \begin{align}
# a_n &= a_{-n} \qquad b_n=-b_{-n}
# \end{align}\tag{2}$$

# $$
# \begin{align}
# f(x)&=\sum_{n=0}^{\infty}a_n cos(\frac{2\pi nx}{L}) + b_nsin(\frac{2\pi nx}{L})\\
# &=\sum_{n=0}^{\infty}a_n\frac{e^{i2\pi nx/L}+e^{-i2\pi nx/L}}{2}+b_n\frac{e^{i2\pi nx/L}-e^{-i2\pi nx/L}}{2i}\\
# &=\sum_{n=0}^{\infty}(\frac{a_n}{2}-\frac{b_n}{2i})e^{i2\pi nx/L}+(\frac{a_n}{2}+\frac{b_n}{2i})e^{-i2\pi nx/L}\\
# &=\frac{1}{2}\sum_{n=0}^{\infty}(a_n-ib_n)e^{i2\pi nx/L}+(a_n+ib_n)e^{-i2\pi nx/L}\\
# &=\sum_{n=-\infty}^{\infty}\frac{1}{2}(a_n-ib_n)e^{i2\pi nx/L}
# \end{align}\tag{3}
# $$

# # Fourier Transform

# From now on, define $k_n=\frac{2\pi n}{L}$, and $\Delta k=\frac{2\pi}{L}$ subsitute (2) into (3) yields:
# $$
# \begin{align}
# f(x)
# &=\sum_{n=-\infty}^{\infty}\frac{1}{2}(a_n-ib_n)e^{ik_nx}\\
# &=\sum_{n=-\infty}^{\infty}\frac{1}{L}\int_0^L\bigl[f(y)cos(k_ny)-if(y)sin(k_ny)\bigr]e^{ik_nx}dy\\
# &=\sum_{n=-\infty}^{\infty}\frac{1}{L}\int_0^Lf(y)\bigl[cos(k_ny)-isin(k_ny)\bigr]e^{ik_nx}dy\\
# &=\sum_{n=-\infty}^{\infty}\frac{1}{L}\int_0^Lf(y)e^{-ik_ny}e^{ik_nx}dy\\
# &=\frac{1}{2\pi}\sum_{n=-\infty}^{\infty}\Delta k\int_0^Lf(y)e^{-ik_ny}e^{ik_nx}dy\\
# \end{align}
# $$

# Shift the integral range for dy from $0-L$ to $-L/2-L/2$
# $$
# f(x)=\frac{1}{2\pi}\sum_{n=-\infty}^{\infty}\Delta k\int_{-L/2}^{L/2}f(y)e^{-ik_ny}e^{ik_nx}dy
# $$
# If $L\rightarrow \infty$, $\Delta k \rightarrow dk$, the summation changes to integral $\sum \rightarrow \int$:
# $$
# f(x)=\frac{1}{2\pi}\int_{-\infty}^{\infty}dk\int_{-\infty}^{\infty}f(y)e^{-ik_ny}e^{ik_nx}dy
# $$

# Define:
# $$
# f(k)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} f(x)e^{-ik_nx}dx
# $$
# Then:
# $$
# f(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} f(k)e^{ik_nx}dk
# $$
#
# Compare to the definition of Fourier transform, it's found that FT is the limit of FS. To extend the 1D relation to higher dimension should be straightforward.


