% Projection onto the space of positive or semi-positive definite 
% matrices minimizing the Frobenius distance.
% INPUTS: x=initial matrix
%         tol=values to assign to negative eigenvalues. If tol=0, y will be
%         singular
% OUTPUT: y=resulting matrix

function y=frobproj(x,tol)

if min(eig(x))<=0,
    [A,B]=eig(x);
    B=diag(real(B));
    B(B<=0)=tol;
    y=A*diag(B)*A';
    y=tril(y)+tril(y,-1)';
else
    y=x;
end