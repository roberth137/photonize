classdef ModelsAWTE
    properties (Access = protected, Constant)
        
    end

    methods (Static)
	% create linear range vector
	%   with a total of [steps] entries
	%   and range [from to]
        function res = Range(from, to, steps)
            dt = (to - from) / (steps - 1);
            res = from:dt:to;
            res = res(:);
        end

	% create vector with logaritmic scale
	%   with a total of [steps] entries
	%   and range [from to]
        function res = LogRange(from, to, steps)
            dt = log(to/from) / (steps - 1);
            res = exp(log(from):dt:log(to));
            res = res(:);
        end
        
        % 1st order pure decay model (no IRF)
        %   returns model of decays for
        %   [numberOfChannels] channels
        %   with channel width [T]
        %   for all decay times [tau]
        %   out:
        %       [M]: Integral [(ch-1)*T..(ch)*T] for all channels
        %       [Minf]: Integral [(numberOfChannels)*T..(INF)*T]
        function [M Minf] = CreateModel(tau, numberOfChannels, T)
            tau = tau(:);
            n = length(tau);
            ch = numberOfChannels;
            M = zeros(ch, n);
            Minf = zeros(n, 1);
            j = 1:ch;

            for i = 1:n
                t = tau(i);
                factor = exp(T/t) - 1;
                M(:, i) = factor * exp(-j*T/t);
                Minf(i) = exp(-ch*T/t);
            end
        end
        
        % 1st order decay, conv with gaussian IRF
        %   returns decay for alll channels [k]
        %   with gaussian IRF(mu, sigma)
        %   for all decay times [tau]
        %   out:
        %       [d]: dim(d)=[length(k),length(tau)]
        function d = GDecayK(tau, mu, sigma, k)
            sq2 = sqrt(2);
            sigma2 = sigma * sigma;
            for tt = 1:size(tau,1)
                tau2 = tau(tt) * tau(tt);
                % This formula is auto generated (see ExpGaussConvolution)
                d(:,tt) = ((-1 + erf((1 + k - mu)/(sq2*sigma)) +...
                    erfc((k - mu)/(sq2*sigma)) +...
                    exp((sigma2 - 2*(1 + k - mu)*tau(tt))/(2*tau2)).*...
                    (-2 + erfc((-sigma2 + tau(tt) + k*tau(tt) - mu*tau(tt))/...
                    (sq2*sigma*tau(tt))) +...
                    exp(1/tau(tt))*erfc((sigma2 - k*tau(tt) + mu*tau(tt))/...
                    (sq2*sigma*tau(tt))))))/2;
                d(:,tt) = abs(d(:,tt));
            end
        end
    end
    
end

