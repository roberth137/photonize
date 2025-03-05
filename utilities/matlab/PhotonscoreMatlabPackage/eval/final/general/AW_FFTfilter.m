function [FFT_Signal_filtered, Frequencies, Signal_Filtered, FFT_Signal] = ...
            AW_FFTfilter(Signal, Fs, freq_low, freq_high, dim, plotfigure, NFFT )
%% function [FFT_Signal, Frequencies, Signal_Filtered] = ...
%%           AW_FFTfilter(Signal, Fs, freq_low, freq_high, dim, plotfigure, NFFT )
% Fs=10*1000; % sampling Frequency
% freq_high=220; 
% freq_low=50;
% dim=1; matrix dimension of time
% TODO: Legend Gruppieren
%% checking input
    if sum(size(size(Signal)))>3
        error ('dimension too big: try 2D Matrix')
    end
    if sum(isfinite( [freq_low; freq_high]))==0
       warning ('no frequency limits: check input! - return full Spectra') 
    end
    if sum(isfinite( [freq_low; freq_high]))==2
        if freq_low>=freq_high
            error ('freq_low>=freq_high: check input')
        end

        if freq_low>=Fs/2 || freq_high>=Fs/2
            error ('freq_input>=SamplingFrequency/2: check input!')
        end

    end


%%
if dim==2
    if nargin==6
        NFFT=2^nextpow2(size(Signal,dim));% numel FFT
        MEA_Zeros=zeros(NFFT,size(Signal,1));
    end
    MEA_Zeros(1:size(Signal,dim),:)=Signal';
else
      if nargin==6
        NFFT=2^nextpow2(size(Signal,dim));% numel FFT
        MEA_Zeros=zeros(NFFT,size(Signal,2));
      end
    MEA_Zeros(1:size(Signal,dim),:)=Signal;
end

Frequencies=Fs*(0:NFFT-1)/NFFT; % frequency
FFT_Signal_filtered=fft(MEA_Zeros, NFFT);
FFT_Signal=fft(MEA_Zeros, NFFT);
FFT_max=max(abs(FFT_Signal_filtered(:)));

    if plotfigure
        figure;
        plot(Frequencies,abs(FFT_Signal)./FFT_max, 'b'); hold on;
        xlabel('Frequency [Hz]')
        ylabel('Normalised Power Spectra')
    end

if isfinite(freq_low) && freq_low<Fs/2;
    % %% Filter Lowpass
    if freq_low==0
        FFT_Signal_filtered=FFT_Signal;
        f_low=1;
    else
        f_low = find(Frequencies<=freq_low,1, 'last'); %% Unter 2 Hz
        FFT_Signal_filtered(1:f_low, :)=0+0*1i;
        FFT_Signal_filtered(end-(f_low-1):end, :)=0+0*1i;
        if plotfigure
            plot(Frequencies,abs(FFT_Signal_filtered)./FFT_max, 'r');
            legend('original', ['Highpass filtered f= ',num2str(Frequencies(f_low)),'Hz'])
        end
    end
end
if isfinite(freq_high) && freq_high<Fs/2;
    % %% Filter Highpass
    f_high = find(Frequencies<=freq_high,1, 'last'); %% Unter 2 Hz
    FFT_Signal_filtered(f_high:end-(f_high-1), :)=0+0*1i;
    if plotfigure
        plot(Frequencies,abs(FFT_Signal_filtered)./FFT_max, 'g');
        legend('original', ['Lowpass filtered f= ',num2str(Frequencies(f_high)),'Hz'])
    end
end
    % %% Filter Bandpass
    if sum(isfinite( [freq_low; freq_high]))==2  && freq_high<Fs/2  && freq_low<Fs/2;
        legend('original', ['Highpass filtered f= ',num2str(Frequencies(f_low)),'Hz'],...
            ['Bandpass filtered f= [', num2str(Frequencies(f_low)),'; ',...
            num2str(Frequencies(f_high)),'] Hz'])

    end
    
MEA_High=real(ifft(FFT_Signal_filtered,NFFT));

if NFFT>=size(Signal, dim)
    Signal_Filtered=MEA_High(1:size(Signal,dim),:);
else
    Signal_Filtered=MEA_High;
end

if dim==2
   Signal_Filtered=Signal_Filtered';
end
end