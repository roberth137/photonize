function cumulative_photons = count_photons(photon_timestamps)
    % count_photons - Counts the number of photons arriving each millisecond
    %                 and calculates the cumulative number of photons
    %                 return to data.ms for writing the photons to frames
    %
    % Syntax:
    %   [photon_counts, cumulative_photons] = count_photons(photon_timestamps)
    %
    % Inputs:
    %   photon_timestamps - Array of photon arrival times in milliseconds.
    %
    % Outputs:
    %   photon_counts - A vector where each element represents the number of
    %                   photons arriving at each millisecond.
    %   cumulative_photons - A vector showing the cumulative total number of
    %                        photons up to each millisecond.

    % Check if the input is non-empty
    if isempty(photon_timestamps)
        error('Input photon_timestamps cannot be empty');
    end
    
    % Find the maximum timestamp (millisecond)
    max_time = max(photon_timestamps);
    
    % Count how many photons arrive at each millisecond using histcounts
    photon_counts = histcounts(photon_timestamps, 0:max_time+1);  % bin edges from 1 to max_time
    photon_counts = [0, photon_counts];
    % Calculate the cumulative sum of photons over time
    cumulative_photons = cumsum(photon_counts);
end