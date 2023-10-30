%% Introduction

% Algorithm name: Sample_convergence.m
% Algorithm author: Cruz Y. Li
%
% CAUTION: Running this algorithm with the example dataset may take days
% and consumes significant RAM. It took more than 15 hours and ~300GB of
% RAM on a 64-core, 512GB-RAM server.
%
% Sample_convergence.m intakes the data matrix sampled on the mid-xy-plane of
% the test subject and iteratively performs the DMD (a Lite version) to
% calculate the grandmean reconstruction error, which is essential for
% assessing the convergence for sampling range. The algorithm outputs a
% grandmean_v1.mat that stores the grand mean reconstruction error versus
% sampling range in a vector. This algorithm takes sampling range as the
% number of oscillation cycles because the flow is highly periodic.
%
% More details on the example case can be found in:
% Li, C. Y., Chen, Z., *Tse, T. K. T., Weerasuriya, A. U., Zhang, X., Fu,
% Y., & Lin, X. (2022). A parametric and feasibility study for data
% sampling of the dynamic mode decomposition: range, resolution, and
% universal convergence states. Nonlinear Dynamics, 107(4), 3683-3707.
% https://doi.org/10.1007/s11071-021-07167-8.
%
% For details on sampling convergence, please refer to:
% Li, C. Y., Chen, Z., *Tse, T. K. T., Weerasuriya, A. U., Zhang, X., Fu,
% Y., & Lin, X. (2022). A parametric and feasibility study for data
% sampling of the dynamic mode decomposition: range, resolution, and
% universal convergence states. Nonlinear Dynamics, 107(4), 3683-3707.
% https://doi.org/10.1007/s11071-021-07167-8.
%
% AND
%
% Li, C. Y., Chen, Z., *Tse, T. K. T., Weerasuriya, A. U., Zhang, X., Fu,
% Y., & Lin, X. (2022). A Parametric and Feasibility Study for Data
% Sampling of the Dynamic Mode Decomposition: Spectral Insights and Further
% Explorations. Physics of Fluids, 34(3), 035102.
% https://doi.org/10.1063/5.0082640
%
% Thanks and please cite :)

%% Step 0: Initialization

clc; clearvars; close all       % clears command window, workspace & close all windows

%% Step 1: Load Data

load('C:\Users\Matheus Seidel\OneDrive\Doutorado\1_Data\Li et al. 2023\Convergence Test\Midplane_xy_Vm_25_cycles_nointerp.mat')  % load the data matrix

%% Step 2: Define Control Parameters

ncycle = 12;      % number of oscillation cycles to consider (max 24)

sf = 1;         % sample every sf snapshot
dt = sf*1e-5;   % uniform inter-snapshot time step (1e-5 is 10x the CFD time step)

% Case specific
Start_frame = 120080;       % starting frame from lift history
Cycle_frame = [121000;121928;122808;123775;124737;125695;126613;127582;...
    128599;129505;130549;131559;132541;133637;134581;135546;136552;...
    137621;138638;139595;140539;141469;142494;143480];      % frame number corresponding to cycles 1-24

%% Step 3: Core Calculation

for i = 1:length(ncycle)       % serial for loop
%parfor i = 1:length(ncycle)     % parallel parfor loop
    tic         % start timing
    
    End_frame = Cycle_frame(ncycle(i));         % end frame
    
    Num_begin = Start_frame-Start_frame+1;      % corresponding starting column in data matrix
    Num_end = End_frame-Start_frame+1;          % corresponding ending column in data matrix
    
    
    Vm_all_mean = mean(Vm_all(:,Num_begin:sf:Num_end),2);   % compute mean  data matrix
    Vm_all_f = Vm_all(:,Num_begin:sf:Num_end)-Vm_all_mean;  % compute fluctuating data matrix
    
    X1 = Vm_all_f(:,Num_begin:end-1);               % truncated data matrix from 1-to-m-1
    X2 = Vm_all_f(:,Num_begin+1:end);               % truncated data matrix from 2-to-m
    
    r = size(X1,2);                                 % no trucation (default), or specify r
    
    %------------------------- Running DMD Lite -------------------------%
    
    tic
    
    [Phi_p,alpha_p,omega,Atilde,U,S,V,W_r,D] = DMD_Lite(X1,X2,r,dt);
    
    %Output
    %   Phi_p - project DMD mode (from Ritz pair)
    %   alpha_p - alpha amplitude from projected modes
    %   omega - continuous-time eigenvalues
    
    %Input
    %   X1 - snapshot 1
    %   X2 - snapshot 2
    %   r - trucation order
    %   dt - inter-snapshot time interval
        
        [eDMD_mean,rss,tr,time_dynamics,Xdmd,eDMD] = DMDReconstruction_Lite(alpha_p,omega,Phi_p,X1,dt);    % reconstruction 
                                                                            % by projected DMD
    
    grandmean(1,i) = mean(eDMD_mean);       % calculating spatiotemporal-average reconstruction error
    
    disp(['-----Completed for ',num2str(i),'/',num2str(ncycle(end))...
        ,' oscillation cycle(s) -----'])  % display oscillation cycle
    toc
 
end

%% Step 4: Plot Data

f1 = figure();
set(gcf,'Position',[20 50 1400 500]);
p(1)=plot(ncycle,grandmean(1,:)./100,'.:','Color',[0 0.4470 0.7410],'LineWidth',2,...
    'MarkerSize',20);

set(gca,'FontSize',26);set(gca,'FontName','Times New Roman','FontAngle','italic');

xlim([0 24])
grid on

ylabel('\it G_{\mid\mid e \mid\mid_2}')
xlabel('Cycle')

%% Step 5: Save Data
  
    save('C:\Users\Matheus Seidel\OneDrive\Doutorado\1_Data\Li et al. 2023\grandmean_proj_a_phi.mat','grandmean','ncycle')
    
%    exportgraphics(f1,['D:\Cruz\MatLab File Upload\Best Practice for DMD\Convergence Test\'...
%        'grandmean_projected.tif'],'Resolution',300)


%% Embedded Functions
function [Phi_p,alpha_p,omega,Atilde,U,S,V,W_r,D] = DMD_Lite(X1,X2,r,dt)
%   Phi - exact DMD mode
%   Phi_p - project DMD mode (from Ritz pair)
%   alpha - alpha amplitude from exact modes
%   alpha_p - alpha amplitude from projected modes
%   omega - continuous-time eigenvalues

%Singular-value Decomposition
[U,S,V] = svd(X1,0);
%                             %U: POD modes
%                             %S: Diagonals contian non-zero singular values

%Truncation (Optional)
r = min(r,size(U,2));
U_r = U(:,1:r);
S_r = S(1:r,1:r);
V_r = V(:,1:r);

%
%Compute A_tilde
%A_tilde defines a low-dimensional linear model of the dynamic system on
%POD coordinates
Atilde = U_r'*X2*V_r/S_r;

%Eigen-decomposition
%W_r: eigenvectors
%D: diagonal matrix containing eigenvalues
[W_r,D] = eig(Atilde);

%Projected DMD modes

Phi_p = U_r*W_r;

lambda = diag(D);   %Discrete-time eigenvalues
omega = log(lambda)/dt; %Continuous-time eigenvalues

% Modal Amplitude
%b is the initial amplitude of each mode, represents the modal contribution
%on the initial snapshot x1. x1 = Phi*alpha.
x1 = X1(:,1);
alpha_p = Phi_p\x1;  %from projected DMD mode

% Data-Driven Residual
rs = zeros(length(alpha_p),1);

for i = 1:length(alpha_p)
    rs(i) = norm((Phi(:,i)-Phi_p(:,i)),2);
end
end

function [eDMD_mean,rss,tr,time_dynamics,Xdmd,eDMD] = DMDReconstruction_Lite...
    (alpha_rank,omega_rank,Phi_rank,X1,dt)

rss = length(alpha_rank);   %selected mode, if rs=size(X1,2) full

tr = size(X1,2);                %time range to reconstruct

%n*size(X1,2), if n=1, sampling time; n>1 prediction

time_dynamics = zeros(rss,tr);
t = (0:tr-1)*dt;

for iter = 1:tr
    time_dynamics(:,iter) = (alpha_rank.*exp(omega_rank*t(iter)));
end

Xdmd = Phi_rank*time_dynamics;

%eDMD = abs((X1(:,1:end-1)-Xdmd(:,2:end))./(X1(:,1:end-1)+1e-20))*.100;
eDMD = abs((X1(:,1:end)-Xdmd(:,1:end))./(X1(:,1:end)+1e-20))*.100;

eDMD_mean = zeros(1,size(X1,2));

for i=1:size(X1,2)
    eDMD_mean(i) = mean(eDMD(:,i));
end
end
