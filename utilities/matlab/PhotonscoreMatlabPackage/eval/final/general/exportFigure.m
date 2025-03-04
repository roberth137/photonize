function exportFigure(fname,resolution)
%EXPORTFIGURE
%
% exportFigure(fname)
% exportFigure(fname,resolution)

% defaults
format = 'png';
if ~exist('resolution', 'var')
    resolution = 96;
end

[~,~,ext] = fileparts(fname);
ext = ext(2:end);
if ~isempty(ext)
    format = ext;
else
    fname = [fname '.' format];
end

% style structure
style = struct;
style.Version = '1';
style.Format = format;
style.Preview = 'none';
style.Width = 'auto';
style.Height = 'auto';
style.Units = 'centimeters';
style.Color = 'rgb';
style.Background = 'w';
style.FixedFontSize = '10';
style.ScaledFontSize = 'auto';
style.FontMode = 'scaled';
style.FontSizeMin = '8';
style.FixedLineWidth = '1';
style.ScaledLineWidth = 'auto';
style.LineMode = 'none';
style.LineWidthMin = '0.5';
style.FontName = 'auto';
style.FontWeight = 'auto';
style.FontAngle = 'auto';
style.FontEncoding = 'latin1';
style.PSLevel = '2';
style.Renderer = 'painters';
style.Resolution = num2str(resolution);
style.LineStyleMap = 'none';
style.ApplyStyle = '0';
style.Bounds = 'loose';
style.LockAxes = 'on';
style.LockAxesTicks = 'off';
style.ShowUI = 'on';
style.SeparateText = 'off';

% export
hgexport(gcf, fname, style);

end
