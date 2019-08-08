%--------------------------------------------------------------------------------
% Compute energies and <r^2> for N=(1:30) and compare with ab initio
% results of: 
% - Chang and Bertsch, PRA 76, 021603(R) (2007) [arXiv:physics/0703190v1]
% - von Stecher, Greene and Blume, PRL 99, 233201 (2007) [arXiv:0708.2734v2]
% and arXiv:0801.2747v1
%
% 
% SLDA with Bessel-DVR method, Littlejohn and Cargo, J.Chem.Phys. 117, 27 (2002)
% Broyden updates on all sites at once 
% fixed particle numbers for both spin up and down species
% smoothing around Ec and finite temperature
%
% Units: m=hbar=omega=1 since a gas at unitarity in a harmonic trap has
% universal properties as well and m, hbar and omega factor out of the
% problem, see Bulgac, PRA, 040502(R) (2007)
% 
% This is the acutal program used for calculations in the above publication
%
%  
%--------------------------------------------------------------------------------

clear
format short g
 
amix    = 1.0;      %  weight in the usual iterative procedure with linear mixing
bmix    = 1.0;      %  size of dx step with Broyden 
cmix    = 1.0;      %  initial Jacobian

convergence_limit = 1.0e-9;
deps    = 1.0e-15;  % essentially defines machine precisions 
ddmu    = 1.0e-6;
part_c  = 0;
ener_c  = 0;
nlpr    = 101;
nplot   = 101;

Nmax    = 100;
Niter   = 0*Nmax;
Kmax    = sqrt(2*Nmax+3)+3;
% Rmax    = Kmax;
Rmax    = 9.0;


%---------------------------------------------------------------------------------
%  The energy cutoff Ec is made smooth with a Fermi function 
%
%  A small finite temperature is introduced in order to faciltitate convergence 
%  especially in the case of odd or polarized systems. The actual value of
%  T is so small as to be insignificant physically and it can be put T=0
%  at the end of calculations
%---------------------------------------------------------------------------------
Ec      = 0.95*(Nmax+1.5); 
dEc     = Ec/50;
Temp    = 0.025;
four_pi = 4*pi;

%------------------------------------------------
%           SLDA parameters
%------------------------------------------------

% %------------------------------------------------
% % the original Carlson et al estimates
% %------------------------------------------------
% xi   = 0.44;
% zeta = xi;
% eta  = 0.486;
% alpha  =  1.1175;
% beta   = -0.51955;
% igamma = -0.095509;
%---------------------------------------------------
% these are likely the best estimates
%---------------------------------------------------
xi   = 0.42;
zeta = xi;
eta  = 0.504;
alpha  =  1.14;
beta   = -0.55269;
igamma = -0.090585;
% beta bar in equation (20)
bbar   = beta - eta^2*(3*pi^2)^(2/3)*igamma/6; 
%--------------------------------------------------- 


%-------------------------------------------------------------
% here I construct various things needed for the DVR method 
%-------------------------------------------------------------
% determine the zeros of the Bessel functions j0(z) and j1(z)
%-------------------------------------------------------------
Nzeros = 500;
nn  = (1:Nzeros);
Z0  = pi*nn;  %zeros for 0th bessel function
Z1  = pi*(nn + 0.5);  % compute zeros for the first order bessel function
for in = 1:20
   Z1 = nn*pi + atan(Z1);  % use the relation tg(x)=x
end
%-------------------------------------------------------------

iz0 = find( Z0 <= Rmax*Kmax);  % max index of the zero solution for given max
iz1 = find( Z1 <= Rmax*Kmax);

%----------------------------------------------------------------------
% Here I construct kinetic energy matrix for l=0 and l=1,[angular momentum]
% see Littlejohn and Cargo, J.Chem.Phys. 117, 27 (2002) for Bessel-DVR.
% I am using the l=0 lattice points for even ls and l=1 for odd ones,
% see Nygaard, Bruun, Schneider, Clark and Feder, PRA 69, 053622 (2004) 
% [arXiv:cond-mat/0312258v1]. I have also checked rather extensively 
% on a number of exact solutions.
%----------------------------------------------------------------------
% see equation (7.1)
for l0 = 0:1
    if l0 == 0
        n0 = iz0';
        z0 = Z0(n0)';       % zeros of spherical Bessel function
        nu0 = 1/2;
        [m1,m2]  = meshgrid(n0,n0);
        [zz1, zz2] = meshgrid(z0,z0);
        t0 = (1+2*(nu0^2-1)./z0.^2)/3; % compute diagonal terms
        t1 = 8*(-1).^(m1-m2).*zz1.*zz2./( (zz1.^2-zz2.^2).^2 + eps ); % compute the off-diagonal terms
        t1 = t1-diag(diag(t1)); % set the diagonal terms to zero as it's not the right values
        T0    = Kmax^2*(diag(t0) + t1)/2*alpha;
        zmax0 = z0(end); % not used
        r0    = z0/Kmax;
        r0_2  = r0.^2;
        mm0 = size(n0,1);
    elseif l0 == 1
        n1 = iz1';
        z1 = Z1(n1)';
        nu0 = 3/2;
        [m1,m2]  = meshgrid(n1,n1);
        [zz1, zz2] = meshgrid(z1,z1);
        t0 = (1+2*(nu0^2-1)./z1.^2)/3;
        t1 = 8*(-1).^(m1-m2).*zz1.*zz2./( (zz1.^2-zz2.^2).^2 + eps );
        t1 = t1-diag(diag(t1));
        T1    = Kmax^2*(diag(t0) + t1)/2*alpha;
        zmax1 = z1(end); % not used
        r1    = z1/Kmax;
        r1_2  = r1.^2;
        mm1 = size(n1,1);
    end
end
%--------------------------------------------------------------------

%=================================================================
%  convert from one coordinate set f(r_a) to another f(r_b) 
%  calculate weights
%=================================================================
U1_0 = zeros(mm1,mm0);
U0_1 = zeros(mm0,mm1);
C0   = zeros(mm0,1);         % 1/F0(z0)
C1   = zeros(mm1,1);         % 1/F1(z1)
for i0 = 1:mm0
a          = cos(z0(i0))/sqrt(z0(i0));
b          = sin(z1)./sqrt(z1);
U1_0(:,i0) = 2*sqrt(z0(i0)*z1)./(z1.^2-z0(i0).^2).*b/a;   
C0(i0)     = sqrt(pi/Kmax);
end
C0_2 = C0.^2;

for i1 = 1:mm1
a          = sin(z1(i1))/sqrt(z1(i1));
b          = -cos(z0)./sqrt(z0); 
U0_1(:,i1) = 2*sqrt(z1(i1)*z0)./(z0.^2-z1(i1).^2).*b/a;
C1(i1)     = sqrt(pi/Kmax/(sin(z1(i1)))^2);
end
C1_2 = C1.^2;  % not used
%-----------------------------------------------------------------------
% clear Nzeros nn Z0 Z1 m1 m2 zz1 zz2 iz0 iz1 t0 t1 nu0 a b 
%-----------------------------------------------------------------------
% this concludes the determination of various quantities needed for DVR
%-----------------------------------------------------------------------

Number_Density = []; % not used

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
for  N = 2:30          % total particle number
    
    tic % tic starts a stopwatch timer

    N_a     = ceil(N/2);   % number of spin-up particles
    N_b     = floor(N/2);  % number of spin-down particles

    %----------------------------------------------------------------------
    % Start iterations
    %
    % The intial guesses for densities, fields and chemical potentials
    % are rather crude. Note that the particle numbers obtained from these 
    % initial guess for number densities are rather far from the desired
    % values (almost by a factor of two sometimes), but that has little
    % effect on the convergence 
    %
    %
    %----------------------------------------------------------------------

    mu      = (3*N)^(1/3)*sqrt(xi);
    mu_a    = mu;                      % chemical potential for spin-up
    mu_b    = mu;                      % chemical potential for spin-down
    mu0     = mu;                      
    E0      = (3*N)^(4/3)/4*sqrt(xi); % useless
    ir_0    = ( (2*mu - r0_2)>0 );
    rho_0_a = ( (2*mu - r0_2)/(alpha*(1+bbar)) ).^1.5/(6*pi^2).*ir_0;
    rho_0_b = ( (2*mu - r0_2)/(alpha*(1+bbar)) ).^1.5/(6*pi^2).*ir_0;
    rho_0   = rho_0_a + rho_0_b;
    D_0     =  eta*(3*pi^2*rho_0).^(2/3)/2.*ir_0;  
    V_0     =  bbar*(3*pi^2*rho_0).^(2/3)/2.*ir_0;
    ir_1    = ( (2*mu - r1_2)>0 );
    rho_1_a = ( (2*mu - r1_2)/(alpha*(1+bbar)) ).^1.5/(6*pi^2).*ir_1;
    rho_1_b = ( (2*mu - r1_2)/(alpha*(1+bbar)) ).^1.5/(6*pi^2).*ir_1;
    rho_1   = rho_1_a + rho_1_b;
    D_1     =  eta*(3*pi^2*rho_1).^(2/3)/2.*ir_1;  
    V_1     =  bbar*(3*pi^2*rho_1).^(2/3)/2.*ir_1;

    x0 = [V_0;D_0;V_1;D_1;mu_a;mu_b];
    x1 = x0;
    dx = x0;
    G0 = x0;
    G1 = x0;
    dG = x0;
    K0 = cmix*eye(size(x0,1));
    K1 = K0;

    convergence = 1;
    iter    = 0;
    while convergence > convergence_limit
        %--------------------------------------    
        iter    = iter + 1;
        Etot    = 0;
        Eqs     = [];
        Ll      = [];
        % nocc_a  = [];
        % nocc_b  = [];
        rho_0_a = zeros(mm0,1);
        rho_0_b = zeros(mm0,1);
        kappa_0 = zeros(mm0,1);
        rho_1_a = zeros(mm1,1);
        rho_1_b = zeros(mm1,1);
        kappa_1 = zeros(mm1,1);
        %--------------------------------------
        for l=0:Nmax         % angular momentum 

            l0 = mod(l,2);
            if l0 == 0
                ll   = alpha*(l*(l+1) - l0*(l0+1))/2;
                Vl   = ll./r0_2;
                Hh_a = T0 + diag(Vl + V_0 + r0_2/2-mu_b);
                Hh_b = T0 + diag(Vl + V_0 + r0_2/2-mu_a);
                Dh   = diag(D_0);
                HH   = [[Hh_a,Dh];[Dh,-Hh_b]];
             elseif l0 == 1
                ll    = alpha*(l*(l+1) - l0*(l0+1))/2;
                Vl    = ll./r1_2;
                Hh_a  = T1 + diag(Vl + V_1 + r1_2/2-mu_b);
                Hh_b  = T1 + diag(Vl + V_1 + r1_2/2-mu_a);
                Dh  = diag(D_1);
                HH = [[Hh_a,Dh];[Dh,-Hh_b]];
            end
            [phi, eigen] = eig(HH);
            ener = diag(eigen);
            al =   (2*l+1)/four_pi;  
            e = 0;
                for ie = 1:size(ener,1)
                    en_t    = ener(ie)/(Temp+eps);
                    if en_t > 40
                        fe = 0;
                    elseif en_t < -40
                        fe = 1;
                    else
                        fe      = 1/(1+exp(en_t));
                    end
                    en_c    = (abs(ener(ie))-Ec)/dEc;
                    if en_c > 40
                        fc = 0;
                    elseif en_c < -40
                        fc = 1;
                    else
                        fc      = 1/(1+exp(en_c));
                    end
                    if fc > 0
                        if l0 == 0
                            ur0     = phi(    1:  mm0,ie)./C0;
                            vr0     = phi(mm0+1:2*mm0,ie)./C0;
                            vr1     = U1_0*vr0;
                            ur1     = U1_0*ur0;
                        elseif l0 == 1
                            ur1     = phi(    1:  mm1,ie)./C1;
                            vr1     = phi(mm1+1:2*mm1,ie)./C1;
                            vr0     = U0_1*vr1;
                            ur0     = U0_1*ur1;
                        end

                        rho_0_a = rho_0_a +   (1-fe)*fc*al*vr0.*vr0;
                        rho_0_b = rho_0_b +       fe*fc*al*ur0.*ur0;
                        kappa_0 = kappa_0 + (1-2*fe)*fc*al*vr0.*ur0;

                        rho_1_a = rho_1_a +   (1-fe)*fc*al*vr1.*vr1;
                        rho_1_b = rho_1_b +       fe*fc*al*ur1.*ur1;
                        kappa_1 = kappa_1 + (1-2*fe)*fc*al*vr1.*ur1;

                        ev      = vr0.^2.*(mu_a-ener(ie)-V_0) + vr0.*ur0.*D_0;
                        eu      = ur0.^2.*(mu_b+ener(ie)-V_0) - vr0.*ur0.*D_0;
                        e = e + four_pi*al*sum(((1-fe)*ev+fe*eu).*C0_2)*fc;
                        
                    end
        %             nocc_a    = [nocc_a, (1-fe)*fc];
        %             nocc_b    = [nocc_b,     fe*fc];
                end
          Etot    = Etot + e;

          Eqs = [Eqs, ener'];
          Ll  = [Ll,  l*ones(size(ener))']; 

        end       % ------------ end l loop

        rho_0_a = rho_0_a./r0_2;
        rho_0_b = rho_0_b./r0_2;
        rho_0   = rho_0_a + rho_0_b;   
        kappa_0 = kappa_0./r0_2/2;

        rho_1_a = rho_1_a./r1_2;
        rho_1_b = rho_1_b./r1_2;
        rho_1   = rho_1_a + rho_1_b;
        kappa_1 = kappa_1./r1_2/2;

        %--------------------------------------------------------------
        %        l=0 sites 
        %--------------------------------------------------------------
        k0  = sqrt(2*(mu - r0_2/2 - V_0)/alpha);
        kc  = sqrt(2*(Ec + mu  - r0_2/2 - V_0)/alpha);
        Lc  = real( (kc-k0/2.*log((kc+k0)./(kc-k0)))/(2*pi^2*alpha) );
        g_eff     = 1./( rho_0.^(1/3)*igamma - Lc + eps);
        last_corr = 1 - D_0.^2/(6*pi^2*alpha^2) ...
                 .*real( log((kc+k0)./(kc-k0))./(k0.*rho_0+eps) );

        D_0 = -g_eff.*kappa_0;                           % pairing field
        V_0 =  beta*(3*pi^2*rho_0).^(2/3)/2 -  ...
                D_0.^2*igamma./(3*(rho_0+eps).^(2/3));   % meanfield
        V_0 = V_0./last_corr;
        %------------------------------------------------------------------------
        Etot = Etot   ... 
         + 0.3*beta*(3*pi^2)^(2/3)*four_pi*sum(r0_2.*rho_0.^(5/3).*C0_2) ...
                                 - four_pi*sum(r0_2.*D_0.*kappa_0.*C0_2);
        %------------------------------------------------------------------------                    
        %--------------------------------------------------------------
        %        l=1 sites 
        %--------------------------------------------------------------
        k0  = sqrt(2*(mu - r1_2/2 - V_1)/alpha);
        kc  = sqrt(2*(Ec + mu - r1_2/2 - V_1)/alpha);
        Lc  = real( (kc-k0/2.*log((kc+k0)./(kc-k0)))/(2*pi^2*alpha) );
        g_eff     = 1./( rho_1.^(1/3)*igamma - Lc + eps);
        last_corr = 1 - D_1.^2/(6*pi^2*alpha^2) ...
                 .*real( log((kc+k0)./(kc-k0))./(k0.*rho_1+eps) );

        D_1 = -g_eff.*kappa_1;                           % pairing field
        V_1 =  beta*(3*pi^2*rho_1).^(2/3)/2 -  ...
                D_1.^2*igamma./(3*(rho_1+eps).^(2/3));   % meanfield
        V_1 = V_1./last_corr;

        %-------------------------------------------------------------------   
        N0_a =  four_pi*sum(r0.^2.*C0.^2.*rho_0_a);
        N0_b =  four_pi*sum(r0.^2.*C0.^2.*rho_0_b);
        R2_0 =  four_pi*sum(r0.^4.*C0.^2.*rho_0);
        %-------------------------------------------------------------------
        % This is where Broyden method is applied to perform the update
        % Notice that the meanfield (V_0,V_1), the pairing field (D_0, D_1)
        % and the chemical potentials (mu_a, mu_b)are updated, until 
        % convergence is acheived for the desired values of specific particle 
        % numbers (N_a, N_b) and accuracy. The first update (iter==1) 
        % is simple, apart from weight, which is of little relevance.
        % Notice also that one can run the program with the simple linear 
        % mixing by choosing Niter >0 accordingly.
        %-------------------------------------------------------------------
        if iter == 1
            G0 = x0 - [V_0;D_0;V_1;D_1; ...
                       mu_a+mu_a*(N_a/N0_a-1);mu_b+mu_b*(N_b/N0_b-1)];
        %     G0 = x0 - [V_0;D_0;V_1;D_1; ...
        %                mu_a+mu0*(N_a/N0_a-1);mu_b+mu0*(N_b/N0_b-1)];
            x1 = x0 - amix*G0;
            dx = x1 - x0;
            x0 = x1;
        elseif iter > 1
            G1  = x0 - [V_0;D_0;V_1;D_1;  ...
                       mu_a+mu_a*(N_a/N0_a-1);mu_b+mu_b*(N_b/N0_b-1)];
        %    G1  = x0 - [V_0;D_0;V_1;D_1;  ...
        %               mu_a+mu0*(N_a/N0_a-1);mu_b+mu0*(N_b/N0_b-1)];
            dG  = G1 - G0;
            ket = dx - K0*dG;
            ket = ket.*(abs(ket)>deps*(abs(G0)+abs(G1)));
            bra = dx'*K0;
            inorm = bmix/(bra*dG);
            K1  = K0 + ket*bra*inorm;
            if iter < Niter
                x1  = x0 - amix* G1;     % linear mixing
            else
                x1  = x0 - K1*G1;
            end
            K0  = K1;
            dx  = x1 - x0;
            x0  = x1;
            G0  = G1;
        end

        V_0  = x0(          1:  mm0,1);
        D_0  = x0(  mm0    +1:2*mm0,1);
        V_1  = x0(2*mm0    +1:2*mm0+  mm1,1);
        D_1  = x0(2*mm0+mm1+1:2*mm0+2*mm1,1);
        mu_a = x0(end-1);
        mu_b = x0(end  );
        mu   = (mu_a+mu_b)/2;
        dmu  = (mu_a-mu_b)/2;
        convergence = max(abs([G0;dx]))
        

    end    % --------- end iterations

    [N, toc, iter, Nmax, Etot, R2_0, mu, dmu*(abs(dmu)>ddmu)]   

    En(N) = Etot;
    R2(N) = R2_0;
    Nn(N) = N;

    Number_Density = [Number_Density, N*ones(size(r0)), r0, rho_0_a, rho_0_b]; 

end    % loop on particle number

%------------------------------------------------------------------------
% The rest of the program is for plotting and comparing SLDA results
% ab initio results.
%------------------------------------------------------------------------


En(1) = 1.37;
R2(1) = 1.37;
Nn(1) = 1;



%--------------------------------------------------------------------------
% Chang and Bertsch, PRA 76, 021603(R) (2007) [arXiv:physics/0703190v1]
%--------------------------------------------------------------------------
np_0  = (1:22);
en_0  = [1.5, 2.01, 4.28, 5.1, 7.6, 8.7, 11.3, 12.6, 15.6, 17.2, 19.9, 21.5, ... 
       25.2, 26.6, 30.0, 31.9, 35.4, 37.4, 41.1, 43.2, 46.9, 49.3];
den_0 = [0, 0.02, 0.04, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.2, ...
        0.4, 0.1, 0.3, 0.2, 0.3, 0.3, 0.4, 0.2, 0.1];     
np_v = [2, 3, 8, 14, 20];
en_v = [ 1.9463, 4.1944, 11.914, 25.965, 40.765];
en_vv= (3*np_v).^(4/3)/4*sqrt(xi);  % not used
%--------------------------------------------------------------------------

in = 0;
for N = 3:2:21
   in = in + 1;
   Gap_slda(in) = En(N)  -(En(N+1)   + En(N-1))/2; % not used
   Gap_gfmc(in) = en_0(N)-(en_0(N+1) + en_0(N-1))/2;  %not used
end
%--------------------------------------------------------------------------
% J. von Stecher, C.H. Greene, and D. Blume, 
% PRL 99, 233201 (2007) [arXiv:0708.2734v2] and arXiv:0801.2747v1 
% NB There might be slight differences between the two works in the 
% last digits.
%--------------------------------------------------------------------------

doerte = [
1   1.5     0.0; 
2	2.002	0.0;
3   4.281   0.004;   
4	5.051	0.009;
5	7.610	0.01;
6	8.639	0.03;
7	11.362	0.02;
8	12.573	0.03;
9	15.691	0.05;
10	16.806	0.04;
11	20.102	0.07;
12	21.278	0.05;
13	24.787	0.09;
14	25.923	0.05;
15	29.593	0.1;
16	30.876	0.06;
17	34.634	0.12;
18	35.971	0.07;
19	39.820	0.15;
20	41.302	0.08;
21	45.474	0.15;
22	46.889	0.09;
23	51.010	0.18;
24	52.624	0.20;
25	56.846	0.22;
26	58.545	0.18;
27	63.238	0.22;
28	64.388	0.31;
29	69.126	0.31;
30	70.927	0.3];

np_1  = doerte(:,1);
en_1  = doerte(:,2);
den_1 = doerte(:,3);
%--------------------------------------------------------------------------

F1 = figure(1);

plot(Nn,En,'bo-')
hold on

plot(np_0,en_0,'r^-')
plot(np_1,en_1,'ks-')

legend('SLDA','GFMC','FN-DMC','Location','SouthEast')   
xlabel('N')
ylabel('E(N)')

% plot(Nn,R2,'bs-')
% plot(np_v,en_v,'rs')
% legend('SLDA E','SLDA <r^2>','GFMC E','FN-DMC','GFMC <r^2>',...
%        'Location','SouthEast') 
hold off
D_En_0 = en_0./En(1:22)-1;
D_En_1 = en_1'./En-1;
axes('position',[0.25,0.6,0.3,0.35])
plot(Nn(1:22),D_En_0,'r^-',np_1,D_En_1,'ks-',[0,30],zeros(2,1),'k')
xlabel('N')
ylabel('\delta E(N)')


F2 = figure(2);
Gap_s = (En(3:2:29) - (En(2:2:28)+En(4:2:30))/2)';
Gap_0 = (en_0(3:2:21) - (en_0(2:2:20)+en_0(4:2:22))/2)';
Gap_1 = (en_1(3:2:29) - (en_1(2:2:28)+en_1(4:2:30))/2);

d_Gap_0 = (den_0(3:2:21)+(den_0(2:2:20)+den_0(4:2:22))/2)';
d_Gap_1 = (den_1(3:2:29)+(den_1(2:2:28)+den_1(4:2:30))/2);

plot(Nn(3:2:29)',Gap_s,'bo-')
hold on
errorbar(Nn(3:2:21)',Gap_0,d_Gap_0,'r^-')
errorbar(Nn(3:2:29)',Gap_1,d_Gap_1,'ks-')
xlabel('N')
ylabel('\Delta(N)')
legend('SLDA','GFMC','FN-DMC','Location','NorthWest')
hold off
