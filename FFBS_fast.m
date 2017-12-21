function [x_sim,x_filt,loglik]=FFBS_fast(y,Q,R)

% preliminaries and memory allocation
[d,T]  = size(y);
loglik = 0;
tol    = 1.0e-18;
x_filt = zeros(d,T);
x_sim  = zeros(d,T);
V_filt = zeros(d,d,T);

% filtering initialization
V0            = eye(d)*0.001;
K             = V0/(V0+R);
V_filt(:,:,1) = V0-K*V0;
x_filt(:,1)   = y(:,1);

% Kalman filter
for t=2:T,
    V_pred        = V_filt(:,:,t-1)+Q;
    K             = V_pred/(V_pred+R);
    x_filt(:,t)   = x_filt(:,t-1)+K*(y(:,t)-x_filt(:,t-1));
    V_filt(:,:,t) = V_pred-K*V_pred;
end

% simulation initialization
V_filt(:,:,T) = tril(V_filt(:,:,T))+tril(V_filt(:,:,T),-1)';
V_filt(:,:,T) = frobproj(V_filt(:,:,T),tol);
x_sim(:,T)    = mvnrnd(x_filt(:,T),V_filt(:,:,T));

% backward simulation
for t=T-1:-1:1,
    h          = x_filt(:,t)+V_filt(:,:,t)/(V_filt(:,:,t)+Q)*(x_sim(:,t+1)-x_filt(:,t));
    H          = V_filt(:,:,t)-V_filt(:,:,t)/(V_filt(:,:,t)+Q)*V_filt(:,:,t);
    H          = tril(H)+tril(H,-1)'; H = frobproj(H,tol);
    x_sim(:,t) = mvnrnd(h,H);
    loglik     = loglik+sum(log(mvnpdf(x_sim(:,t),h,H)));
end