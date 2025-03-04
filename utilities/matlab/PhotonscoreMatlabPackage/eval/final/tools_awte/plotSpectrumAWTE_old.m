% [tauData] = plotSpectrum(tau_range, spectrum, B)
% plot [spectrum] over [tau_range]
%   with legend entries [B] (optional)
%   ndims(spectrum)==2:[length(tau_range), number of spectrum sets]
%   ndims(spectrum)==3:[length(tau_range), number of spectrum sets, number of iterations]
function [tauData] = plotSpectrumAWTE(tau_range, spectrum, B)

    % load style file
%    style = load('C:\Users\tengels\Documents\MATLAB\hgexport_style.mat');
%    style = style.style;
    
    % init plot colors
    textColor = lines(size(spectrum,2));
    specColor = jet(size(spectrum,3));
    
%     plot(ones(size(tau,1),1)*tau', [0 max(max(x(:,:,end)))], 'k')
    
    if (ndims(spectrum) == 2)
        it_ids = 1;
    elseif (ndims(spectrum) == 3)
        it_ids = 1:size(spectrum,3);
    end

%     hold off;
    axisLim = zeros(1, 2);
    legendStr = {};
    j = 0;
    for nn = 1:size(spectrum,2)
        for it = it_ids
            j = j + 1;
            
            x = squeeze(spectrum(:,nn,it));
            
            % spectrum
            semilogx(tau_range, x, 'Color', textColor(nn,:))
            if (nargin>2), legendStr(nn) = B(nn); end;
            ylim = max(x);
            if (ylim > axisLim(2)), axisLim(2) = ylim; end
            hold on;

            % find and plot peaks
            tauData = AnalysisAWTE.estimateTau(x, tau_range);
            peakColor = hsv(length(tauData));
            if isempty(tauData) || round(sum([tauData.area]))>round(sum(x)+1)
                warning('cant find borders in spectrum')
                break
            else
            for tt = 1:length(tauData)
                
                % color peaks
                plot(tau_range(tauData(tt).border(1):tauData(tt).border(2)), x(tauData(tt).border(1):tauData(tt).border(2)), 'Color', peakColor(tt,:), 'LineWidth', 2)
%                 plot(ones(2,1).*tau_range(tauData(tt).border(1)), [0 max(x)], '-.', 'Color', peakColor(tt,:))
%                 plot(ones(2,1).*tau_range(tauData(tt).border(2)), [0 max(x)], '-.', 'Color', peakColor(tt,:))
                
                % max
%                 plot(tauData(tt).byMax, tauData(tt).peakHeight, 'or')

                % barycenter
%                 plot(tauData(tt).byBarycenter, tauData(tt).peakHeight, '+r')
                
                % downarrow on barycenter
%                 text(tauData(tt).byMax, tauData(tt).peakHeight, [{ round(tauData(tt).byBarycenter)};...
%                          {'\downarrow'}],...
%                             'HorizontalAlignment', 'center', 'Color', textColor(nn,:))
                 txtStr = [ { [num2str(round(tauData(tt).byMax))] } ;...
                            { [num2str(round(tauData(tt).area*100/...
                               sum([tauData.area]),1)),' %'] } ];
                 for ts = 1:nn-1
                     txtStr = [txtStr ; {''}];
                 end
                 txtStr = [txtStr ; {'\downarrow'} ;];% {''}];
                 text(tauData(tt).byMax, tauData(tt).peakHeight, txtStr, ...
                            'HorizontalAlignment', 'center', ...
                            'VerticalAlignment', 'bottom', ...
                            'FontSize', 10, 'Color', textColor(nn,:))
                        
                % variance
                %             plot(taus(tt*7+2,nn,it)*ones(1,nn)+varMaus(tt+1,nn), max(max(x(:,nn,it))), '*',);
                %             plot(taus(tt*7+2,nn,it)*ones(1,nn)-varMaus(tt+1,nn), max(max(x(:,nn,it))), '*',);
                
                xlabel('\tau [ps]', 'FontSize', 16)
                ylabel('Estimated Contributions', 'FontSize', 16)
                set(gca, 'FontSize', 14)
            end
            end
        end
    end
    
    if (axisLim(2) > 0)
        axis([1 tau_range(end) 0 axisLim(2)*2])
    end
    
    if (nargin>2), legend(legendStr); end;
        
end
