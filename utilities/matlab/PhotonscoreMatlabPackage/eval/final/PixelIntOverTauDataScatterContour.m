function [img_CF,mean_CF, hsp] = PixelIntOverTauDataScatterContour(img_FLIM, img_Counts, Tau_edges, Int_edges,int_Thresh, Max_Counts,  Max_Tau , Title_t)
%% [img_CF,mean_CF, hsp] = PixelIntOverTauDataScatterContour(img_FLIM, img_Counts, Tau_edges, Int_edges,int_Thresh, Max_Counts,  Max_Tau , Title_t)%%
Tau_edges_inc = Tau_edges(2)-Tau_edges(1);
Int_edges_inc = Int_edges(2)-Int_edges(1);
 
if numel (Tau_edges(1):Tau_edges_inc:Max_Tau*1000)>5
    Tau_Tick_inc = (Max_Tau*1000 - Tau_edges (1))/5;
else Tau_Tick_inc= Tau_edges_inc;
end

if numel (Int_edges(1):Int_edges_inc : Max_Counts)>5
    Int_Tick_inc = (Max_Counts - Int_edges (1))/5;
else Int_Tick_inc = Int_edges_inc;
end
% figure
    subplot(2,2,2)
    
        IntTau(img_Counts,img_FLIM/1000, [int_Thresh Max_Counts],[Tau_edges(1)/1000 Max_Tau], []);
        pos222 = get(gca, 'Position');
        set(gca, 'Position', [pos222(1:2) pos222(3) pos222(4)]);

        mask_int = img_Counts>int_Thresh;%;median(flat(tmpC(tmpC>0)));
        mean_CF  = [mean(flat(img_FLIM(mask_int))), mean(flat(img_Counts(mask_int)))];
    hsp=subplot(223);
    
       [img_CF] = histcounts2(img_FLIM(mask_int) ,img_Counts(mask_int),Tau_edges,Int_edges);
        Marker ='o' ; 
        MarkerSize=5;
        MarkerFaceAlpha=0.1;
        MarkerEdgeAlpha=.1;


        [img_Data_Contour] = AW_PlotAlphaData(flat(img_FLIM(mask_int)),...
                             flat(img_Counts(mask_int)), Marker, MarkerSize,...
                              MarkerFaceAlpha, MarkerEdgeAlpha);
        ylim([-1*Int_edges_inc/2+Int_edges(1) Max_Counts-Int_edges_inc/2]);
        xlim([-1*Tau_edges_inc/2+Tau_edges(1) Max_Tau*1000-Tau_edges_inc/2]);
        set(gca, 'YDir', 'normal')
        set(gca, 'YTick', [(-1*Int_edges_inc/2)+Int_edges(1):Int_Tick_inc:Max_Counts-Int_edges_inc/2], ...
                 'YTickLabel', [Int_edges(1):Int_Tick_inc:Max_Counts] )
             
        set(gca, 'XTick', [(-1*Tau_edges_inc/2)+Tau_edges(1):Tau_Tick_inc:Max_Tau*1000-Tau_edges_inc/2], ...
                 'XTickLabel', [Tau_edges(1) : Tau_Tick_inc:Max_Tau*1000] )     
        xlabel('\tau [ps]')
        ylabel('Intensity [photons]')
        hold on;
        plot(mean_CF(1),mean_CF(2),...
            'sk', 'MarkerFaceColor','r', 'MarkerSize', 10)
        mF=8;
        if sum(flat(medfilt2(img_CF, [mF mF])))==0
           mF=2;
        end   
        AW_ContourAlphaData(medfilt2(img_CF, [mF mF]), Tau_edges(1:end-1), Int_edges(1:end-1) ,4, 1, 0, 1);
        FigStyle;
        pos_imgCF = get(gca,'Position');
        set(gca, 'Position', [pos_imgCF(1:2) pos_imgCF(3)*1.3 pos_imgCF(4)*1.30])
        pos_imgCF = get(gca,'Position');
 
        % axis square
    
    subplot(2,2,1)
    
        bar(Tau_edges(1)   + Tau_edges_inc/2 : Tau_edges_inc:...
            Tau_edges(end) - Tau_edges_inc/2, sum(img_CF,2)) % Tau
        ylabel('# pixel')
%         hold on; plot(50:100:9950, sum(img_CF,2), 'r-')
%         xlabel('\tau [ps]')
        xlim([Tau_edges(1) Max_Tau*1000])
        pos_c= get(gca, 'Position');
        set(gca, 'XTick', [Tau_edges(1):Tau_Tick_inc:Max_Tau*1000], ...
                 'XTickLabel', [Tau_edges(1):Tau_Tick_inc:Max_Tau*1000] )
             
        FigStyle;
        set(gca, 'Position',[pos_c(1:2) pos_imgCF(3), pos_c(4)]);
        set(gca, 'XTickLabel', [])
       if Title_t ~0;
        title(['time = ' num2str(Title_t)])
       end
       
    subplot(2,2,4)
    
        barh(Int_edges(1)  + Int_edges_inc/2 : Int_edges_inc:...
            Int_edges(end) - Int_edges_inc/2 , sum(img_CF,1))
        ylim([Int_edges(1) Max_Counts])
        set(gca, 'YTick', [Int_edges(1):Int_Tick_inc:Max_Counts], ...
                 'YTickLabel', [Int_edges(1):Int_Tick_inc:Max_Counts] )
             
        xlabel('# pixel')
%         ylabel('Intensity [photons]')
        FigStyle
        pos_c= get(gca, 'Position');
        set(gca, 'Position',[pos_c(1)*1.05 pos_c(2) pos_c(3)*0.8 pos_imgCF(4)]);
        set(gca, 'YTickLabel', [])

end
