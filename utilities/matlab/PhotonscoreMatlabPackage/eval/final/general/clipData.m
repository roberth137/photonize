function data = clipData(data,lims)
%CLIPDATA Clip/winsorize data according to the specified clipping limits.
%
%   Usage: clipData(data,lims)
%   Values < lims(1) are replaced with lims(1).
%   Values > lims(2) are replaced with lims(2).
%
%   If lims(1) or lims(2) is NaN, then the corresponding replacement
%   is skipped. By setting one lim-value to NaN, one indicates that
%   only the other lim-value is to be used.
%
%   File   : clipData.m
%   Author : Kristian Loewe

assert(numel(lims) == 2);

if ~isnan(lims(1))
  data(data < lims(1)) = lims(1);
end
if ~isnan(lims(2))
  data(data > lims(2)) = lims(2);
end

end
