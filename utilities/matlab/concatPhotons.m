function concatenatedPhotons = concatPhotons(data, data1)

    % Check if the structs have the fields 'x' and 'ms'
    if ~isfield(data, 'x') || ~isfield(data, 'ms') || ~isfield(data, 'y') || ~isfield(data, 'dt')
        error('struct1 must contain fields ''x'' and ''ms''.');
    end
    if ~isfield(data1, 'x') || ~isfield(data1, 'ms') || ~isfield(data1, 'y') || ~isfield(data1, 'dt')
        error('struct2 must contain fields ''x'' and ''ms''.');
    end
    %disp(data1.ms(1:5))

    if ~isempty(data.ms)
        lastMsPhotons = data.ms(end);
        data1.ms = data1.ms(2:end) + lastMsPhotons;% Assuming 'ms' is an array
    end

    %disp(data1.ms(1:5))
    
    % Concatenate the structs
    concatenatedPhotons = struct();
    concatenatedPhotons.x = [data.x; data1.x];
    concatenatedPhotons.y = [data.y; data1.y];
    concatenatedPhotons.dt = [data.dt; data1.dt];
    concatenatedPhotons.ms = [data.ms; data1.ms];
end