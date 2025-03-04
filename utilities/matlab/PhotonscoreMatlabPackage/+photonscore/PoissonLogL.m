function res = PoissonLogL(counts, model)
s = counts > 0 & model > 0;
n = counts(s);
g = model(s);
res = 2 * sum(n.*log(n./g) - (n - g));
end