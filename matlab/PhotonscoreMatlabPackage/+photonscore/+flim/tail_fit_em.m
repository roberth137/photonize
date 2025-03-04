function [x1, m] = tail_fit_em(decays, x0, iterations, accelerate)
if nargin == 4
  arg_iterations = iterations;
  arg_accelerate = accelerate;
end

if nargin == 3
  arg_iterations = iterations;
  arg_accelerate = 1;
end

if nargin == 2
  arg_iterations = 100;
  arg_accelerate = 1;
end

if size(decays, 1) == 1
  decays = decays(:);
end
fits = size(decays, 2);

x0 = x0(:);
if mod(length(x0), fits) ~=0
  error 'Inconsistent X0 size'
end

x = photonscore_mex(photonscore.Const.FN_FLIM_TAIL_FIT_EM, ...
    double(decays), double(x0), arg_iterations, arg_accelerate);

x1.tau = x(2, :);

m0 = photonscore.flim.tail_decay(size(decays, 1), x1.tau);
v = sum(m0);
x1.a = x(1, :) .* (1./v) * sum(decays);
m.model = m0 * x1.a';
m.residuals = photonscore.flim.residuals(decays, m.model);

end
