function [y1,P_y]=f_gibbs(P,nsim)        %P¼´ÎªÎÄÖÐYt

% preliminaries and memory allocation
[T,d]     = size(P);                     % dimensions of price matrix
y1        = NaN*ones(d*(d+1)/2,nsim);    % sampled Var elements
a         = isnan(P)==0;                 % data indicator          
tol       = 1e-04;

% missings initialization
P_y=P; n_miss=sum(a==0);
if sum(n_miss)~=0
    lag = 10*T./(T-n_miss); lag=round(lag);
    for j = 1:d        
        P_y(1,j)=P(find(a(:,j)==1,1,'first'),j);
        P_y(1:lag(j),j)=nanmean(P_y(1:lag(j),j));
        for i = (1+lag(j)):T,P_y(i,j) = nanmean(P_y((i-lag(j)):min(i+lag(j),T),j)); end
    end
end
P_y(a==1)=P(a==1);
    
% covariances initialization
sigma_new = HY(P);
omega_new = diag(max(diag(diff(P_y)'*diff(P_y)-sigma_new),tol));

% vectorization of covariance parameters
y1(:,1)=vech2(sigma_new);

% projection to positive matrix with min Frobenius
sigma_new=frobproj(sigma_new,0.001);
    
%%%%%%%%%% GIBBS SAMPLER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=2:nsim,
    disp(t)
    tic
    
    % sampling latent process
    x_sim    = FFBS_fast(P_y',sigma_new/T,omega_new/T)';
    SS_sigma = diff(x_sim)'*diff(x_sim);
    SS_omega = (P_y-x_sim)'*(P_y-x_sim);
    SS_sigma = tril(SS_sigma)+tril(SS_sigma,-1)';
    SS_omega = tril(SS_omega)+tril(SS_omega,-1)';
    
    % sampling sigma
    sigma_new = iwishrnd(SS_sigma,T)*T;
    y1(:,t)   = vech2(sigma_new);
    
    % sampling omega
    omega_new = iwishrnd(SS_omega,T)*T;
    y3        = vech2(omega_new); disp(y3')
    
    % sampling missing observations
    if sum(n_miss)~=0,
        s=y3(1:d)/T;
        for i=1:d,
            ind = setdiff(1:d,i); a_i = a(:,i)==0;
            Omega22=omega_new(ind,ind)/T;
            Omega12=omega_new(i,ind)/T;
            P_y(a_i,i)=normrnd(x_sim(a_i,i)+(Omega12/Omega22*(P_y(a_i,ind)-...
                x_sim(a_i,ind))')',sqrt(s(i)-Omega12/Omega22*Omega12'));
        end
        P_y=real(P_y);
    end
    
    % plots
    for j=1:3,subplot(3,1,j),plot(y1(j,1:t));end;
    pause(0.0001)
    
    toc
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%