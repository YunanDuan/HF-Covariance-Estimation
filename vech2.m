function trandformeddata=vech2(x)
% PURPOSE:
%        Transform a k by k matrix into a vector of size k*(k+1)/2 by 1,
%        with diagonal elements first, and then column-wise
% INPUTS:
%      data:   A k by k matrix
% OUTPUTS:
%      transformeddata - a k*(k+1)/2 by 1 vector for the form
%        [data(1,1) data(2,2) ... data(k,k) data(2,1)...data(k,1)...data(k,k-1)]'
% 
% EXAMPLE:
%      vech2([0.16,0.06,-0.01;
%             0.06,0.09, 0.04;
%            -0.01,0.04, 0.12])=
%            [0.16,0.09,0.12,0.06,-0.01,0.04]' 
% 

trandformeddata=x(logical(tril(ones(size(x)),-1)));
trandformeddata=[diag(x);trandformeddata];