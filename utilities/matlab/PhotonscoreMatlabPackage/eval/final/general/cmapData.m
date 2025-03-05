function varargout = cmapData(data,cmap,varargin)
%CMAPDATA Map data to RGB values, given a color map.
% modify AW: if max(indCmap(:)) > size(cmap,1) Rounding error
% File   :  cmapData.m
% Author :  Kristian Loewe


% rescale data to the range 0 ... n, given that 'cmap' has n RGB values
if(isempty(varargin))
    data = rescaleData(data,[0 size(cmap,1)]);
elseif(numel(varargin)==1)
    data = rescaleData(data,[0 size(cmap,1)],varargin{1});
else
    error('Unexpected number of input arguments.');
end

% initialize variables for rgb-channels
r = NaN(size(data));
g = NaN(size(data));
b = NaN(size(data));

% indCmap (same size as data): index into cmap for each voxel
indCmap = ceil(data);    % bin data, one bin per rgb-value in cmap; 
                         % 0 < b1 <= 1 < b2 <= 2 < b3 <= ... < bn <= n
indCmap(indCmap==0) = 1; % adjustment for first bin, resulting in:
                         % 0 <= b1 <= 1 < b2 <= 2 < b3 <= ... < bn <= n
                         %      |         |         |           |
              % cmap index:     1         2         3           n
if max(indCmap(:)) > size(cmap,1)
    indCmap(indCmap > size(cmap,1) )=size(cmap,1);
end 
% See also: 
% http://stat.ethz.ch/R-manual/R-patched/library/graphics/html/hist.html :
% "If right = TRUE (default), the histogram cells are intervals of the 
% form (a, b], i.e., they include their right-hand endpoint, but not their 
% left one, with the exception of the first cell when include.lowest is TRUE."


% indNotNan: logical index for exclusion of NaNs
indNotNan = ~isnan(indCmap);

% fill rgb-channels with appropriate values from colormap
r(indNotNan) = cmap(indCmap(indNotNan),1);
g(indNotNan) = cmap(indCmap(indNotNan),2);
b(indNotNan) = cmap(indCmap(indNotNan),3);

if nargout==1
    varargout{1} = cat(ndims(data)+1,r,g,b);
elseif nargout==3
    varargout{1} = r;
    varargout{2} = g;
    varargout{3} = b;
else
    error('Unexpected number of input arguments.');
end
end
