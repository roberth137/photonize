function [L, ChiMLE] = PoissonLogL_Chi_MLE(counts, model)
% s = counts > 0 & model > 0;
% n = counts(s);
% g = model(s);
% L = 2 * sum(n.*log(n./g) - (n - g));
% ChiMLE = L/sum(n>0);
% s = counts > 0 & model > 0;
n = counts;
g = model;
tmp = n.*log(n./g) - (n - g);
s = isfinite(tmp);
tmp(~s)=0;
L = 2 * sum(tmp);
ChiMLE = L./size(n,1);
end