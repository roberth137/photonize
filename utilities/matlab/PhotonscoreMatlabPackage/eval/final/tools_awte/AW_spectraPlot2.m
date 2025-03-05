function   [tau, Normalpha, COUNTS] = AW_spectraPlot2(spectrum, tau_range_exp, ch, iterations, B, ...
                                        do_print, ownFigure,  holdFigure, XLimit, YLimit)
                                    
%% [tau,Normalpha, COUNTS] = AW_spectraPlot2(spectrum, tau_range_exp, ch, iterations, B, ...
%%                                         do_print, ownFigure,  holdFigure, XLimit, YLimit)
%% changes: axis (XLim, YLim) 
%% addpath of Style 
% 
% style = load('E:\Users\Public\MATLAB\MEMLifetime\Server\Hefe\With_IRF_Shift_Compensation\hgexport_style.mat');
% style = style.style;

for kk=1:size(spectrum, 2)
%   figure;
spectrum(1,kk)=0;
abl_1=diff(spectrum(:,kk));
abl_2=diff(abl_1);
% plot(abl_1, 'r')
% hold on
% plot(abl_2, 'k')
% plot(spectrum(:,kk))

s = sign(abl_1);
s = [ s;0];%diff(s)
s = s.*(spectrum(:,kk) > 0.1);
s = s.*(-1000);
% stairs(s,'m');
if max(s)==0
warning ('max(spectrum)<0.1')
continue
end
%%
j=1
k=1
for i=1:size(spectrum,1)
    if [ s(i)==1000 && s(i-1)==0 ] || ...
       [ s(i)==-1000 && s(i-1)==0 ] ||...
       [ s(i)==-1000 && s(i-1)==1000 ]
        vektor1(j,kk)=i;
        j=j+1
    end
    if  s(i)==1000 && s(i+1)==0 && j>k ||...
        s(i)==-1000 && s(i+1)==0 && j>k ||...
        s(i)==1000 && s(i+1)==-1000 && j>k 
        vektor2(k,kk)=i  ;
        k=k+1;
    end
end
if do_print
    figure('Visible', 'off');
elseif holdFigure
elseif ownFigure
       get(gca);
else
    figure;
    
end
% h=get(gcf);
semilogx(tau_range_exp*ch,spectrum(1:end-1,kk), '-b')
hold on
 if holdFigure
     ylim([0 max(spectrum(:))+...
             max(spectrum(:))/5])
      
 else ylim([0 max(spectrum(:,kk))+...
             max(spectrum(:,kk))/5])  

 end
% Ccol=['.r';'.k'; '.g'; '.m'; '.c'];
Ccol=lines(size(vektor1,1));
for i=1:size(vektor1,1)
   if isnan(vektor1(i,kk))==0 && vektor1(i,kk)>0
      plot(tau_range_exp(vektor1(i, kk):vektor2(i, kk))*ch,...
           spectrum     (vektor1(i, kk):vektor2(i, kk), kk),...
            'Color', Ccol(i,:), 'LineWidth', 2)
       
       [A,AA]=max(spectrum(vektor1(i,kk):vektor2(i,kk),kk));
        MAX(i,kk)=vektor1(i,kk)+AA-1;
        MAX_tau(i,kk)=tau_range_exp(MAX(i,kk))*ch;
        COUNTS(i,kk)=sum(spectrum(vektor1(i,kk):vektor2(i,kk),kk));
        text(tau_range_exp(MAX(i,kk))*ch,A,[{num2str(round(COUNTS(i,kk)))};...
                         { round(MAX_tau(i,kk))};...
                         {'\downarrow'};...
                         {'          '};...
                         {'          '};...
                         {'          '}],...
                            'HorizontalAlignment', 'center',...
                            'FontSize',18)

%        if MAX_tau(i,kk)<border
%           FREI(i,kk)=COUNTS(i,kk);
%        else
%           GEBU(i,kk)=COUNTS(i,kk);
%        end
        set(gca, 'FontName', 'Arial')
        set(gca, 'FontName', 'Arial')
%         grid on
%         set(gca, 'GridLineStyle', ':');

   end
end

if  sum(abs(YLimit)) ~= 0
    set(gca, 'YLim', [YLimit(1) YLimit(2)])
else ylim([0 max(spectrum(:,kk))+...
        max(spectrum(:,kk))/5])
end

if XLimit==0
    set(gca, 'XLim', [tau_range_exp(1)*ch tau_range_exp(end)*ch])
else
    set(gca, 'XLim', [XLimit(1)*ch XLimit(2)*ch])
end

xlabel('\tau [ps]', 'FontSize', 16)
ylabel('Estimated Contributions', 'FontSize', 16)
set(gca, 'FontSize', 14, 'TickDir', 'out', 'Layer', 'top');

if holdFigure
else
    
    % legend(leg{})
    title(B{kk})
end
        if do_print
            exportFigure( ['Est_Lifetimes_',strrep(B{kk},' ', '_'),'_',num2str(iterations(kk)),'it_', '.png'],300);
            saveas(gcf, ['Est_Lifetimes_' ,strrep(B{kk},' ', '_'),'_',num2str(iterations(kk)),'it_','.fig'], ...
                 'fig');
        end

end
if max(s)==0
warning ('max(spectrum)<0.1')
tau=NaN;
Normalpha=NaN;
COUNTS = NaN;
else
tau=MAX_tau;
Normalpha=zeros(size(COUNTS));
for i=1:size(COUNTS,2)
Normalpha(:,i)=COUNTS(:,i)/sum(COUNTS(:,i));
end
end
end
