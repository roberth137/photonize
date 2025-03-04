function data = cutData(data,lims,repl)
%CUTDATA Cut data according to the specified limits.
%
%   Usage: cutData(data,lims)
%   Values < lims(1) are replaced with NaN.
%   Values > lims(2) are replaced with NaN.
%
%   Usage: cutData(data,lims,subst)
%   Values < lims(1) are replaced with the value specified by 'repl'.
%   Values > lims(2) are replaced with the value speficied by 'repl'.
%
%   If lims(1) or lims(2) is NaN, then the corresponding replacement
%   is skipped. By setting one lim-value to NaN, one indicates that
%   only the other lim-value is to be used.
%
%   File   : cutData.m
%   Author : Kristian Loewe

assert(numel(lims) == 2);

if exist('repl', 'var')
  assert(isscalar(repl) && isnumeric(repl));
else
  repl = NaN;
end

if ~isnan(lims(1))
  data(data < lims(1)) = repl;
end
if ~isnan(lims(2))
  data(data > lims(2)) = repl;
end

end
