% Calculate the Frobenius distance between the estimated and realized
% covariance 
Q0=initialQ;
R0=diag([0.0505,0.0222,0.2011,0.0937,0.1425,0.0822,0.0606,0.1040,0.1719,0.0072]);
d=10;T=240;
X0=(log([100,40,60,80,40,20,90,30,50,60]))';
v=[1/2,1/3,1/2,1/4,1/4,1/3,1/5,1/4,1/3,1/4];
[Y,X,RCQ]=DGP(Q0,R0,d,T,X0,v);
Q_init=initialQ;R_init=eye(d);
[x_filt,x_smooth,Q,R,Q_old,c_conv,l,i,q,r]=KEM(Y,Q_init,R_init,1000,0.001,0);
sum=0;
for i=1:10
    for j=1:10
        sum=sum+(RCQ(i,j)/239-Q(i,j))^2;
    end
end
disp(sqrt(sum));