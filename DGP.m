function [Y,X,RCQ]=DGP(Q0,R0,d,T,X0,v)
% Data Generating Process
% INPUTS:  QO       = d*d covariance matrix describing asynchronicity
%          R0       = d*d diagnal matrix describing microstructure noise
%          d        = number of asset class
%          T        = total number of points in the price process
%          X0       = initial price process 
%          v        = missing probability vector
% OUTPUTS: Y        = observed price process with values missing at random
%          X        = latent price process
%          RCQ      = realized covariance matrix of price process X

% Preliminary %
X=zeros(d,T);sigma=Q0/240;Y=zeros(d,T);
W=randn(d,T);B=randn(d,T);

% Generate latent price process X using Heston Model %
V=diag(sigma);
for s=1:d
    X(s,1)=X0(s)+sqrt(V(d))*W(s,1);
    sigma(s,1)=V(d)+2*(0.01-V(d))+0.1*sqrt(V(d))*B(s,1);
    sigma(s,1)=abs(sigma(s,1));
end
for i=1:d
    for j=2:T
        X(i,j)=X(i,j-1)+sigma(i,j-1)*W(i,j);
        sigma(i,j)=sigma(i,j-1)+2*(0.01-sigma(i,j-1))+0.1*sqrt(sigma(i,j-1))*B(i,j);
        sigma(i,j)=abs(sigma(i,j));
    end
end

% Calculate Realized Covariance Matrix RCQ %
RCQ=zeros(10);
for i=2:T
    m=X(:,i)-X(:,i-1);
    RCQ=RCQ+m*m';
end

% Generate observed price process Y with missing values%  
for i=1:T
    L=(mvtrnd(R0,d,1))';
    Y(:,i)=X(:,i)+L;
end

for r=1:d
    indices=randperm(T);
    indices=indices(1:v(r)*T);
    for c=1:v(r)*T
        Y(r,indices(c))=NaN;
    end
end

for t=1:T
    j=isnan(Y(:,t));% isnan: return 1 if NaN, o if number %
    for s=1:10   
    if j(s)==1 
        Y(s,t)=0;
    end
    end
end
    




