function [hc] = AW_ContourAlphaData(img_CF, X_edges, Y_edges ,NumerOfLines, ShowLines, ShowText, ShowAreas)
% %  [hc] = AW_ContourAlphaData(img_CF, X_edges, Y_edges ,NumerOfLines, ShowLines, ShowText, ShowAreas)
if ShowText
       c_text = 'on';
else   c_text = 'off';
end
[~,hc] = contourf(X_edges, Y_edges, img_CF', NumerOfLines, 'ShowText',c_text);
drawnow;
%
if ~ShowLines
    set(hc.EdgePrims, 'Visible', 'off');
else
    [hc.EdgePrims.ColorType] = deal('truecoloralpha');
    for iFP = 1:numel(hc.EdgePrims)
        hc.EdgePrims(iFP).ColorData = hc.FacePrims(iFP).ColorData;
        hc.EdgePrims(iFP).ColorData(4) = uint8(255*0.8);
    end
    %     hc.EdgePrims(1).ColorData(4) = uint8(0);
end

if ~ShowAreas
    set(hc.FacePrims, 'Visible', 'off');
else
    [hc.FacePrims.ColorType] = deal('truecoloralpha');
    for iFP = 1:numel(hc.FacePrims)
        hc.FacePrims(iFP).ColorData(4) = uint8(255*0.4);
    end
    hc.FacePrims(1).ColorData(4) = uint8(0);
end
end