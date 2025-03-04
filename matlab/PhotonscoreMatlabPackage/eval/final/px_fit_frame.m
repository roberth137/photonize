function a = px_fit_frame(px_tau, irf_shift, tau_ref, fr, rdecay, tr, x, y, dt, dim)
% % a = px_fit_frame(px_tau, irf_shift, tau_ref, fr, rdecay, tr, x, y, dt, dim)
fl = photonscore.flim.sort(x, 0, 4096, dim, y, dt);
mask = int32(fl.image > tr);
fm = find(mask);
mask(fm) = 1:length(fm);
dd = photonscore.flim.decay_from_mask(fl, mask, ...
    rdecay(1), rdecay(end), rdecay(end)-rdecay(1));
rr = photonscore.flim.decay_from_mask(fr, mask, ...
    rdecay(1), rdecay(end), rdecay(end)-rdecay(1));
dd = dd(:, 2:end);
rr = rr(:, 2:end);
for i=1:size(rr, 2)
    rr(:, i) = rr(:, i) / sum(rr(:,i));
end
% form the model matrix
m = photonscore.flim.convolve(rr, irf_shift,...
    size(dd, 1), ...
    px_tau,...
    tau_ref);
x3 = ones(size(m,2), size(dd, 2));
tic;
x4 = photonscore.flim.admixture_em(m, dd, x3, ...
    'iterations', 50, 'method', 3);
toc;
a = zeros([size(mask), size(m, 2)]);
for i=1:size(m, 2)
    tmp = a(:, :, i);
    tmp(fm) = x4(i, :);
    a(:, :, i) = tmp;
end
end

