function d = tail_decay(channels, tau)
tau = tau(:);
d = zeros(channels, length(tau));
factor = exp(-1./tau)';
d(1, :) = 1 - factor;
for i=2:channels
  d(i, :) = d(i - 1, :) .* factor;
end
