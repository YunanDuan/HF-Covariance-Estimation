function [x_filt,x_smooth,Q,R,Q_old,c_conv,l,i,q,r]=KEM(y,Q_init,R_init,maxiter,eps,showplot)    

% KEM algorithm
% INPUTS:  y        = dxT data matrix with NaN of d subjects for T observations
%          Q_init   = initial value for innovation error variance
%          R_init   = initial value for observation error variance
%          maxiter  = max n of iterations allowed before convergence
%          eps      = min tolerance change in log-likelihood for convergence
%          showplot = plots are (not) shown if (0) 1
% OUTPUTS: x_filt   = filtered data at last iteration
%          x_smooth = smoothed data at last iteration
%          Q        = estimated innovation error variance at last iteration
%          R        = estimated observation error variance at last iteration
%          Q_old    = estimated innovation error variance before the last iteration
%          c_conv   = equal to 1 if convergence has been reached
%          l        = log-likelihood path
%          i        = num of steps done until convergence (or maxiter)
%          q        = vectorized innovation error variance path
%          r        = vectorized observation error variance path

% preliminaries
[d,T]=size(y);
a=isnan(y)==0;
c_conv=0;

% memory allocation
l=NaN*ones(1,maxiter);
deltalog=NaN*ones(1,maxiter);
q=NaN*ones(d*(d+1)/2,maxiter);
r=NaN*ones(d*(d+1)/2,maxiter);

% initialization
l_old=-10e10;
Q=Q_init; R=R_init; I=eye(d); Phi=I;

for i=1:maxiter
    tic
    disp(['iteration ', int2str(i)])

    % Kalman filtering and smoothing
    y(a==0)=NaN;
    clear x_filt x_smooth V_smooth Vt_smooth loglik 
    [x_filt,x_smooth,V_smooth,Vt_smooth,loglik]=Ksmoother_miss_fast(y,Q,R);
    y(a==0)=0;
    
    % compute incomplete data log-likelihood
    l(i)=loglik;
    
    % E step
    S=zeros(d); S10=zeros(d); eps_smooth=zeros(d);
    for t=1:T
        D=diag(a(:,t));
        S=S+x_smooth(:,t)*x_smooth(:,t)'+V_smooth(:,:,t);
        if t>1,S10=S10+x_smooth(:,t)*x_smooth(:,t-1)'+Vt_smooth(:,:,t);end
        eps_smooth=eps_smooth+D*(y(:,t)-x_smooth(:,t))*(y(:,t)-x_smooth(:,t))'*D'...
                   +D*V_smooth(:,:,t)*D'+(I-D)*R*(I-D)';
    end
    S00=S-x_smooth(:,T)*x_smooth(:,T)'-V_smooth(:,:,T);
    S11=S-x_smooth(:,1)*x_smooth(:,1)'-V_smooth(:,:,1);
    
    % M step
    Q_old=Q; R_old=R; Phi_old=Phi; %#ok<NASGU>
    Q=(S11-S10/S00*S10')/(T-1);
    Q=tril(Q)+tril(Q,-1)';
    q(:,i)=vech2(Q);
    R=eps_smooth/T;
    R=tril(R)+tril(R,-1)';
    r(:,i)=vech2(R);
    Phi=S10/S00;
    
    % relevant plots
    if showplot==1,
    subplot(2,2,1), plot(l(1:i)),title('Incomplete data Log-likelihood');
    subplot(2,2,2), plot(q(d,1:i)*(T-1)*252),title(['q(',int2str(d),',',int2str(d),') path']);
    subplot(2,2,3), plot(q(d*(d+1)/2,1:i)*(T-1)*252),title(['q(',int2str(d-1),',',int2str(d),') path']);
    subplot(2,2,4), plot(r(d,1:i)*T*252),title(['r(',int2str(d),',',int2str(d),') path']);
    pause(0.0001)
    end

    % break for convergence
    deltalog(i)=abs(l(i)-l_old)/abs(l_old);
    if deltalog(i)<eps,
        str=int2str(i);
        disp(['break for convergence at iteration ', str])
        c_conv=1;
        break
    end
    l_old=l(i);
    toc
end

% display if missed convergence
if i==maxiter,disp('convergence not reached');end

% reshaping for convergence before maxiter
l=l(isnan(l)==0);
q=q(isnan(q)==0);
q=reshape(q,d*(d+1)/2,[]);
r=r(isnan(r)==0);
r=reshape(r,d*(d+1)/2,[]);

Q=Q*(T-1);
R=R*T;
Q_old=Q_old*(T-1);