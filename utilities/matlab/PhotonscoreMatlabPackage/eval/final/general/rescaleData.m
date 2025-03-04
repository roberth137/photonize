function data = rescaleData(data,rout,rin)
%RESCALEDATA Rescale (linearly transform) data to the specified range.
%
%   Usage: rescaleData(data,rout)
%   Transforms data according to
%                                 rout(2) - rout(1)
%     out = (data - min_data) * --------------------- + rout(1), where
%                                max_data - min_data
%
%   min_data and max_data are computed using nanmin and nanmax, respectively.
%
%   Usage: rescaleData(data,rout,rin)
%   Transforms data according to
%                                rout(2) - rout(1)
%     out = (data - rin(1))   * -------------------   + rout(1)
%                                 rin(2) - rin(1)
%   If rin(1) or rin(2) are set to NaN, they are replaced by min_data and
%   max_data, respectively. As mentioned above, min_data and max_data are
%   computed using nanmin and nanmax, respectively.
%
%   File   : rescaleData.m
%   Author : Kristian Loewe

assert(numel(rout) == 2);
dtype = class(data);
assert(ismember(dtype, {'single','double'}));

if all(isnan(data(:)))
  return;
end

if strcmp(dtype, 'single')
  rout = single(rout);
end

if ~exist('rin', 'var')
  rin = [nanmin(data(:)) nanmax(data(:))];
else
  assert(numel(rin) == 2);
  if strcmp(dtype, 'single')
    rin = single(rin);
  end
  if ~isnan(rin(1))
    assert(rin(1) <= nanmin(data(:)));
  else
    rin(1) = nanmin(data(:));
  end
  if ~isnan(rin(2))
    assert(rin(2) >= nanmax(data(:)));
  else
    rin(2) = nanmax(data(:));
  end
end

data = (data - rin(1)) * (rout(2) - rout(1)) ./ (rin(2) - rin(1)) + rout(1);

end
