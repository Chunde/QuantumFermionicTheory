% SOLVES THE 1D GPE VIA THE SPLIT-STEP FOURIER METHOD
clear all;clf; %Clear workspace and figure

hbar=1.054e-34;amu=1.660538921e-27; %Physical constants
m=87*amu;as=5.8e-9; %Atomic mass; scattering length
N=1000;wr=100*2*pi;wx=40*2*pi; %Atom number; trap frequencies

M=200; Nx=2*M+1;
dx=double(2e-7); x=(-M:1:M)*dx; %Define spatial grid
dk=pi/(M*dx); k=(-M:1:M)*dk; %Define k-space grid
dt=double(10e-8); Nt=200000; %Define time step and number

lr=sqrt(hbar/(m*wr)); lx=sqrt(hbar/(m*wx)); %HO lengths
g1d=2*hbar*hbar*as/(m*lr^2); %1D interaction coefficient

V=0.5*m*wx^2*x.^2/hbar; %Define potential
psi_0=sqrt(N/lx)*(1/pi)^(1/4)*exp(-x.^2/(2*lx^2)); %Initial wavefunction

% [psi_0,mu] = get_ground_state(psi_0,dt,g1d,x,k,m,V); %Imaginary time

Nframe=100; %Data saved every Nframe steps
t=0; i=1; psi=psi_0; spacetime=[]; %Initialization

for itime=1:Nt %Time-stepping with split-step Fourier method
    psi=psi.*exp(-0.5*1i*dt*(V+(g1d/hbar)*abs(psi).^2));
    psi_k=fftshift(fft(psi)/Nx);
    psi_k=psi_k.*exp(-0.5*dt*1i*(hbar/m)*k.^2);
    psi=ifft(ifftshift(psi_k))*Nx;
    psi=psi.*exp(-0.5*1i*dt*(V+(g1d/hbar)*abs(psi).^2));
    if mod(itime,Nt/Nframe) == 0 %Save wavefunction every Nframe steps
        spacetime=vertcat(spacetime,abs(psi.^2)); t
    end
    t=t+dt;
end

subplot(1,3,1); %Plot potential
plot(x,V,'k'); xlabel('x (m)'); ylabel('V (J/hbar)');

subplot(1,3,2); %Plot initial and final density
plot(x,abs(psi_0).^2,'k',x,abs(psi).^2,'b');
legend('\psi(x,0)','\psi(x,T)');xlabel('x (m)');ylabel('|\psi|^2 (m^{-1})');

subplot(1,3,3); % Plot spacetime evolution as pcolor plot
dt_large=dt*double(Nt/Nframe);
pcolor(x,dt_large*(1:1:Nframe),spacetime); shading interp;
xlabel('x (m)'); ylabel('t (s)');