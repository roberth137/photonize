function AW_IntTauPng(data1, data2, lim1, lim2, cmap, fname, varargin)
%
% Usage: AW_IntTauPng(data1, data2, lim1, lim2, cmap, fname)
%
% Usage: AW_IntTauPng(data1, data2, lim1, lim2, cmap, fname, rin1, rin2)
%
% Examples:
%   lntTauPng(data1, data2, lim1, lim2, cmap, fname);
%   lntTauPng(data1, data2, lim1, lim2, cmap, fname, [a b], [c d]);
%
%   If you want to specify only rin2 but not rin1, set rin1 to []:
%   lntTauPng(data1, data2, lim1, lim2, cmap, fname, [],    [c d]);

assert(nargin == 6 || nargin == 8);
rin1 = []; rin2 = [];
if nargin == 8; rin1 = varargin{1}; rin2 = varargin{2}; end

data1 = clipData(flipud(data1), [lim1(1) NaN]);
data2 = cutData(flipud(data2), [lim2(1) NaN], 0);

data1 = clipData(data1, [NaN lim1(2)]);
data2 = clipData(data2, [NaN lim2(2)]);

if ~isempty(rin1)
    rgb = cmapData(data1, cmap, rin1);
else
    rgb = cmapData(data1, cmap);
end

if ~isempty(rin2)
    a = rescaleData(data2, [0 1], rin2);
else
    a = rescaleData(data2, [0 1]);
end

imwrite(flipdim(rgb,1),fname, 'Alpha', flipdim(a,1));
end