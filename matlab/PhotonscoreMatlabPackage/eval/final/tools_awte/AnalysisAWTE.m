classdef AnalysisAWTE
    properties (Access = protected, Constant)
        
    end

    methods (Static)
        % EM algorithm to estimate contributions of tau
        %   for given model [mo] and dataset of decays [d]
        %   dim(d):[channels with decay data, number of decays]
        %   dim(mo):[channels per decay, number of decays]
        %   with initial estimate [x0]
        %   a total of [numberOfIterations] iteration steps will be performed
        %   Every [savedSteps]'th step:
        %       spectrum is saved into [x],
        %       last step (regardless of [savedSteps]) is saved into x(end)
        %   out:
        %       [x]: estimates
        %       [x_sol]: estimated, when [chi2] reaches calue of 1 or lower
        %       [Li]: Likelihood function
        %       [chi2]: error chi²
        %       [lambda]: L/E, derived from klkkt-paper  (stopping rules)
        function [x, x_sol, Li, chi2, lambda] = EMcore(d, mo, mi, x0, numberOfIterations, savedSteps)

            Li = zeros(numberOfIterations, size(d,2));
            chi2 = zeros(numberOfIterations, size(d,2));
            L = zeros(numberOfIterations, size(d,2));
            E = zeros(numberOfIterations, size(d,2));
            % size(x,3): +1 for last iteration / best iterations step
            x = ones(size(mo,2), size(d,2), floor(numberOfIterations/savedSteps) + 1);
            x_sol = zeros(size(x0));
            x(:,:,1) = x0;
            a = x0;
%             it_sol = ones(size(d,2), 1);

            j = 0;
            for i = 1:numberOfIterations

                % EM
                if (i > 1) % i=1 is x0
                    if (~isempty(mi))
                        a = AnalysisAWTE.UnmixEMInf(mo, mi, d, a);
                    else
                        a = AnalysisAWTE.UnmixEM(mo, d, a);
                    end
                    
                    % save only some iteration steps for later use
                    if (mod(i,savedSteps) == 0)
                        j = j + 1;
                        j*savedSteps;
                        x(:,:,j) = a;
                    end
                    
                    % save last step
                    if (i == numberOfIterations)
                        x(:,:,end) = a;
                    end
                end

                % Likelihood
                [Li(i,:) chi2(i,:)] = AnalysisAWTE.Likelihood(mo*a, d);

                fchi2 = find(chi2(i,:) > 1);
                if (~isempty(fchi2))
                    x_sol(:,fchi2) = a(:,fchi2);
                end
                
                % klkkt stopping rule
%                 for kl = 1:size(d,2)
%                     [L(i,kl) E(i,kl)] = AnalysisAWTE.klkkt(mo, a(:,kl), d(:,kl));
% 
%                     % stoppingRule = L - lambda*E
%                     % lambda = L / E     (when stoppingRule->0)
% 
%                 end
            end
            % if iteration step of solution is bigger than numberOfIterations
%             it_sol(find(it_sol>numberOfIterations)) = numberOfIterations; 

            lambda = L ./ E;
        end
        
        % [x]: estimates
        % [y]: decay
        % [H]: model
        function [L, E] = klkkt(H, x, y)
            Hx = div(1, H * x);
            L = x .* (H' * (1 - y .* Hx));
            L = L' * L;
            E = ((H .* H)' * Hx)' * (x .* x);
            
            function x = div(a, b)
                x = a ./ b;
                x(isnan(x) | isinf(x)) = 0;
            end
        end
        
        % Step EM algorithm to resolve mixtures
        %   [m]: model matrix
        %   [n]: goal counts
        %   [x0]: initial guess
        function x = UnmixEM(m, n, x0)
            g = m * x0;
            g = n ./ g;
            g(isnan(g) | isinf(g)) = 0;
            x = x0 .* (m' * g);
        end
        
        % Step EM algorithm to resolve mixtures
        %   [m]: model matrix
        %   [mi]: infinity model pixel
        %   [n]: goal counts
        %   [x0]: initial guess
        function x = UnmixEMInf(m, mi, n, x0)
            g = m * x0;
            g = n ./ g; %% equals to 1 (only in inf pixel) Normalisation 
            g(isnan(g) | isinf(g)) = 0;
            x = x0 .* (m' * g + mi);
        end
        
        % compute likelihood function
        %   using fit(model*estimates) [g] and decay [n]
        %   out:
        %       [Li]: likelihood function
        %       [chi2]: error
        function [Li, chi2] = Likelihood(g, n)
            chi2 = zeros(1, size(n,2));
            
            s = n > 0 & g > 0;
            Li = sum(n(s) .* log(n(s) ./ g(s))) + sum(g - n);
            
            resid = (n-g) ./ sqrt(n);
            s = isfinite(resid);
            for i = 1:size(n,2)
                temps = (s(:,i));
                tempr = resid(:,i);
                temp = tempr(temps);
%                 chi2(1,i) = sum(temp.*temp)/length(temp);
                chi2(1,i) = sum(temp'*temp)/length(temp);
            end
        end

        % compute variance for mono exponential decay (Maus, 2001)
        function sigma2 = varTau_Maus(decay, tau, numberOfChannels, T)
            sigma2 = zeros(1,size(decay,2));

        %     % with for-loop
        %     for j = 1:size(decay,2)
        %         sigma2(:,j) = tau^4*numberOfChannels^2/(sum(decay(:,j),1)*T^2)*(1-exp(-T/tau))/...
        %             (exp(T/(numberOfChannels*tau))*(1-exp(-T/tau))/(exp(T/(numberOfChannels*tau))-1)^2-...
        %             numberOfChannels^2/(exp(T/tau)-1));
        %     end

            % without for-loop
            sigma2 = tau^4*numberOfChannels^2./(sum(decay,1).*(T^2))*(1-exp(-T/tau))./...
                (exp(T/(numberOfChannels*tau))*(1-exp(-T/tau))/(exp(T/(numberOfChannels*tau))-1)^2-...
                numberOfChannels^2/(exp(T/tau)-1));
        end
        
        % compute variance (Hall, 1981)
        function sigma2 = varTau_Hall(decay, tau, numberOfChannels, ch_width)
            sigma2 = zeros(1,size(decay,2));
            sigma2 = tau^4./(sum(decay,1).*(ch_width^2)*(exp(ch_width/tau)/(exp(ch_width/tau) - 1)^2 -...
                numberOfChannels^2*exp(numberOfChannels*ch_width/tau)/(exp(numberOfChannels*ch_width/tau) - 1)^2));
            % NaN for tau << numberOfChannels*ch_width
        end
        
        % estimates tau value from spectrum
        %   thus it looks for peaks in [data] and gets the tau value
        %   by calculating the barycenter of the peak in the interval determined by
        %   the minima aside the peak.
        %   If no real minima are found, an interval of [0 size(data)] is used
        %   [tau_range] contains all tau values from the model, which was used in
        %   the EM algorithm when [data] was calculated
        %   [tau_range] and [data] have to be row or column vectors of the same size
        %   out:
        %       [tau]<struct>: estimated tau values
        %       [.byMax]: tau derived from height of peak maimum
        %       [.byBarycenter]: tau derived from weighted peak maximum
        %       [.border]: peak width derived by minima (indices! of [data] entries)
        %       [.area]: area under peak in interval [border]
        %       [.skewness]:    -1: tail on left side of PDF is longer
        %                       +1: tail on right side of PDF is longer
        %                        0: PDF is symmetrical
        function [tau] = estimateTau(data, tau_range)

            % detect peaks by first and secound derivation
            p = AnalysisAWTE.detectPeaks(data);

            tau = [];

            if ~isempty(p)

                tau = struct();
                j=1;
                for i = 1:length(p)
                    if data(p(i).pos)<1
                        continue
                    end
                    % tau from max of peak
                    tau(j).byMax = tau_range(p(i).pos);

                    % barycenter (tau from weighted mean)
                    tauMin = p(i).border(1);
                    tauMax = p(i).border(2)-1;
                    tt2 = sum( data(tauMin:tauMax) .* tau_range(tauMin:tauMax) );
                    tau(j).byBarycenter = tt2 / sum( data(tauMin:tauMax) );
                    
                    % border of peak intervall as indices
                    tau(j).border = p(i).border;
                    
                    % peak height
                    tau(j).peakHeight = data(p(i).pos);
                    
                    % area under peak
                    tau(j).area = sum(data(tauMin:tauMax));

                    % skewness
                    tau(j).skewness = sign( (tauMax-p(i).pos) - (p(i).pos-tauMin) );
                    j=j+1;
                end
            end
        end
        
        % find peaks by maxima in vector [x]
        %   out:
        %       [peaks]<struct>, dim(peaks)=[numberOfPeaks]
        %       [.pos]: index of peak position
        %       [.border]: left-handed and right.handed minimum,
        %                  if no minimum is found [0] and [length(x)] are
        %                  used instead
        function [peaks] = detectPeaks(x)

            % find max, min and turning Points
            [maxi, mini, turni] = AnalysisAWTE.deriv(x);

            peaks = [];

            % find interval for each peak (minima on both sides)
            if ~isempty(maxi)
                if isempty(mini)
                    mini = [1 numel(x)];
                end
                if numel(mini)<numel(maxi)
                    if maxi(1)==1
                        warning('lowest lifetime found')
                    elseif maxi(1)>1 && mini(1)>maxi(1)
                        mini = [1; mini];
                    elseif maxi(end)==numel(x)
                        warning('highest lifetime found')
                    else
                        mini = [mini;numel(x)]
                    end
                end
                peaks = struct();

                % check, if there's a right-handed mining point
                % (esp. when there's only 1 maximum)

                for m = 1:size(maxi,1)
                    d = zeros(2, 1);

                    % no real minimum found
                    if (isempty(mini))
                        d = [1 size(x,1)];
                    else

                        % find two nearest minimum points beside maximum
                        [so, ind] = sort( abs(mini-maxi(m)) );
                             
                        if (mini(ind(1))-maxi(m))<0 &&...
                           (mini(ind(2))-maxi(m))<0 &&...
                            sum((mini-maxi(m))>0)>0
                              tmp =  mini-maxi(m);
                             [V1] = max(tmp(tmp<0));  % left one
                             ind(1)= find(tmp==V1);
                             [V2]  = min(tmp(tmp>0)); 
                             ind(2)= find(tmp==V2);% right one
                        end
                        if (size(ind) < 2)
                            ind(2) = ind(1);
                        end

                        % is minimum left-handed or right-handed?
                        s = sign(maxi(m)-mini(ind(1)));
                        if (s>0)
                            if (s == sign(maxi(m)-mini(ind(2))))
                                d = [mini(ind(1)) size(x,1)];
                            else
                                d = [mini(ind(1)) mini(ind(2))];
                            end
                        else
                            if (s == sign(maxi(m)-mini(ind(2))))
                                d = [1 mini(ind(1))];
                            else
                                d = [mini(ind(2)) mini(ind(1))];
                            end
                        end
                    end

                    peaks(m).pos = maxi(m);
                    peaks(m).border = d;
                end
            else
                % no peak found
            end
        end
        
        % find maxima, minima and turning points in vector [y]
        % [y] must be a row vector
        %   returns empty results, if nothing found   
        %   out:
        %       [max_id]: indices of maxima
        %       [min_id]: indices of minima
        %       [tp_id]: indices of turning points
        function [max_id, min_id, tp_id] = deriv(y)

            deriv1 = diff(y);
            deriv2 = diff(deriv1);

            % find maximum (deriv1==0)
            s1 = sign(deriv1);
            size1 = size(s1,1);

            % find turning points (deriv2==0)
            s2 = sign(deriv2);
            size2 = size(s2,1);

            % save maximum and turning points
            % shift +1, indices are shifted because of diff()-command

            max_id = find( s1([1:size1-1])>s1([2:size1]) ) + 1;
            min_id = find( s1([1:size1-1])<s1([2:size1]) ) + 1;
            tp_id = find( s2([1:size2-1])~=s2([2:size2]) ) + 1;

            % remove max or min at beginning and end
            max_id = max_id( find(max_id~=1 & max_id~=size1-1) );
            min_id = min_id( find(min_id~=1 & min_id~=size1-1) );
            tp_id = tp_id( find(tp_id~=1 & tp_id~=size2-1) );

            % test
%             figure; hold on;
%             t = 1:size(y,1);
%             plot(t, y/max(y));
%             plot(t(max_id), y(max_id)/max(y), 'or')
%             plot(t(min_id), y(min_id)/max(y), 'ok', 'MarkerSize', 8)
%             plot(t(tp_id), y(tp_id)/max(y), '*b', 'MarkerSize', 8)
%             legend('y', 'maximum', 'minimum', 'turning point')
        end
    end
end