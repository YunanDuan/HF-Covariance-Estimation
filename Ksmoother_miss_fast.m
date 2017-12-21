function [x_filt,x_smooth,V_smooth,Vt_smooth,loglik]=Ksmoother_miss_fast(y,Q,R)

% Kalman filter and smoother for the dynamic linear model
% y_t = x_t     + eta_t, eta_t~N(0,R)
% x_t = x_{t-1} + eps_t, eps_t~N(0,Q)

% INPUTS:  y         = dxT matrix of T observations for d subjects
%          Q         = dxd observational error matrix
%          R         = dxd innovation error matrix
% OUTPUTS: x_pred    = predicted value of the latent process
%          x_smooth  = smoothed value of the latent process
%          V_pred    = variance of the prediction error
%          V_smooth  = variance of the smoothing error
%          Vt_smooth = one-lag autocovariance of the smoothing error
%          D         = instrumental matrix expressing the presence of missings
%          V_filt    = variance of the filtering error

% preliminaries and memory allocation
[d,T]=size(y);
a=isnan(y)==0;
I=eye(d);
x_pred=zeros(d,T);x_filt=zeros(d,T);x_smooth=zeros(d,T);
V_filt=zeros(d,d,T);V_pred=zeros(d,d,T);V_smooth=zeros(d,d,T);
Vt_smooth=zeros(d,d,T);J=zeros(d,d,T);
loglik=0;

% filtering initialization
y(a==0)=0;
for j=1:d,x_filt(j,1)=y(j,find(y(j,:),1,'first')); end % fix to 1st obs if 1st tick is missing
V_filt(:,:,1)=zeros(d);
x_pred(:,1)=x_filt(:,1);
V_pred(:,:,1)=V_filt(:,:,1);

for t=2:T,
    
    % "Obervations" matrix
    D=diag(a(:,t));
    
    % Kalman filter
    x_pred(:,t)=x_filt(:,t-1);
    V_pred(:,:,t)=V_filt(:,:,t-1)+Q;    
    K=V_pred(:,:,t)*D'/(D*V_pred(:,:,t)*D'+R);
    x_filt(:,t)=x_pred(:,t)+K*(y(:,t)-D*x_pred(:,t));
    V_filt(:,:,t)=V_pred(:,:,t)-K*D*V_pred(:,:,t);
    V_temp=V_pred(a(:,t),a(:,t));
    R_temp=R(a(:,t),a(:,t));
    if ne(sum(a(:,t)),0)  
        if ne(det(V_temp+R_temp),0)
        loglik=loglik+(-1/2*log(det(V_temp+R_temp))-1/2*((y(a(:,t),t)-x_filt(a(:,t),t-1))'...
            /(V_temp+R_temp)*(y(a(:,t),t)-x_filt(a(:,t),t-1))));
        end
    end
end

% smoothing initialization
x_smooth(:,T)=x_filt(:,T);
V_smooth(:,:,T)=V_filt(:,:,T);
Vt_smooth(:,:,T)=(I-K*D)*V_filt(:,:,T-1);
J(:,:,T-1)=V_filt(:,:,T-1)/V_pred(:,:,T);
x_smooth(:,T-1)=x_filt(:,T-1)+J(:,:,T-1)*(x_smooth(:,T)-x_filt(:,T-1));
V_smooth(:,:,T-1)=V_filt(:,:,T-1)+J(:,:,T-1)*(V_smooth(:,:,T)-V_pred(:,:,T))*J(:,:,T-1)';

% Rauch recursions
for t=T-1:-1:2,
    J(:,:,t-1)=V_filt(:,:,t-1)/V_pred(:,:,t);
    x_smooth(:,t-1)=x_filt(:,t-1)+J(:,:,t-1)*(x_smooth(:,t)-x_filt(:,t-1));
    V_smooth(:,:,t-1)=V_filt(:,:,t-1)+J(:,:,t-1)*(V_smooth(:,:,t)-V_pred(:,:,t))*J(:,:,t-1)';
    Vt_smooth(:,:,t)=V_filt(:,:,t)*J(:,:,t-1)'+J(:,:,t)*(Vt_smooth(:,:,t+1)-V_filt(:,:,t))*J(:,:,t-1)';
end
