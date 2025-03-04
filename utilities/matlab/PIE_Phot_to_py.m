%% Load data from .photons

clear all; clc;

% Set parameters to write TIFF

int_time = 200; % integration time in ms 
ms_offset = 20; % offset in ms 
offset_frames = 10;
pixels = 256;
binning = 4096/pixels;

first_dt = [0, 1680];
second_dt = [1680, 2500]; 

folderPath = uigetdir;

filePattern = fullfile(folderPath, '*.photons');
photonFiles = dir(filePattern);

data = struct();
data.x = [];
data.y = [];
data.dt = [];
data.ms = [];


% Read in and concat structs 
for i = 1:length(photonFiles)
    disp(['______READ IN FILE_____', num2str(i), '/', num2str(length(photonFiles))])
    fileName = photonFiles(i).name;
    filePath = fullfile(folderPath, fileName);
    if i == 1
        savePath = filePath;
    end
    [dataNew, dt_channel] = dotPhotons_toFullMS(filePath);
    
    data = concatPhotons(data, dataNew);

    disp(['length data.x: ', num2str(length(dataNew.x)), '   length data.ms: ', num2str(length(dataNew.ms))])

    clear dataNew
end

%% crop data to full frames: 
ind_frames = floor((length(data.ms)-1)/int_time)-1;
kept_ms = ((ind_frames+1)*int_time)-1; % full frames minus last ms
kept_phot = data.ms(kept_ms+1);
data.x = data.x(1:kept_phot);
data.y = data.y(1:kept_phot);
data.dt = data.dt(1:kept_phot);
data.ms = data.ms(1:kept_ms+1);

disp('____DONE____')
disp('read in and concatenated files, cropped to full frames -1ms')
disp('num full ms: || num full frames: ')
disp([num2str(length(data.ms)-1),'       || ' num2str(ind_frames)])
disp(['kept photons: ', num2str(kept_phot)])

%% Create ms index 
ms = ones(1,data.ms(end)).';
counter = 1;
disp(['length of ms before: ', num2str(length(ms))])
for i = 1:kept_ms
    num_photons = (data.ms(i+1)-data.ms(i));
    for j = 1:num_photons
        ms(counter) = i-1;
        counter = counter+1;
    end
end
data.ms = ms;
clear ms

%% Select ROI of photons

img_tmp = photonscore.hist_2d(data.x(1:10000000), 0, 4095, pixels, data.y(1:10000000));
%take only 10 mio photons to speed up

imagesc(img_tmp)
colormap gray
my_roi = drawrectangle('AspectRatio', 1, 'FixedAspectRatio', 1);
wait(my_roi)
my_pos = my_roi.Position;
close all

if my_pos(1) < 1
    my_pos(1) = 1;
end
if my_pos(2) < 1
    my_pos(2) = 1;
end

fov_in_pixels = floor(my_pos(3));
length_edge = fov_in_pixels -1;

x_edges = floor([my_pos(1), my_pos(1) + length_edge]);
y_edges = floor([my_pos(2), my_pos(2) + length_edge]);

img0 = img_tmp(y_edges(1):y_edges(2), x_edges(1):x_edges(2));


%% Crop photons to ROI and set beginning to 0

x_photons_bounds(1) = (x_edges(1)-1) * binning; %x_edges in tif coordinates: [1:256] for 16x binning
x_photons_bounds(2) = (x_edges(2) * binning) - 1;
y_photons_bounds(1) = (y_edges(1)-1) * binning; %y_edges in tif coordinates: [1:256] for 16x binning
y_photons_bounds(2) = (y_edges(2) * binning) - 1;


disp(' ')
disp('selected ROI:')
disp(['x edges: ', num2str(x_edges), ' |           pixel coordinates (0 - ', num2str(pixels), ')'])
disp(['x_photons_bounds: ', num2str(x_photons_bounds), '| lincam coordinates (0 - 4096)'])
disp(['y edges: ', num2str(y_edges), ' |           pixel coordinates (0 - ', num2str(pixels), ')'])
disp(['y_photons_bounds: ', num2str(y_photons_bounds), '| lincam coordinates (0 - 4096)'])

mask_roi = find(data.x >= y_photons_bounds(1) & data.x <= (y_photons_bounds(2)) ...
    & data.y >= x_photons_bounds(1) & data.y <= (x_photons_bounds(2)));
data.x = data.x(mask_roi);
data.y = data.y(mask_roi);
data.dt = data.dt(mask_roi);
data.ms = data.ms(mask_roi);
disp(['length data.x after cutting ROI: ', num2str(length(data.x))])
disp(' ')

data.x = data.x - y_photons_bounds(1);
data.y = data.y - x_photons_bounds(1);

%% calculate TIF file size and how many to write

max_pixels_file = 4e+09; % maximum 8 bit pixels per tif file - storage for metadata
total_pixels = ind_frames*offset_frames*length_edge^(2);
num_tif_files = ceil(total_pixels/max_pixels_file);

indiv_frames_per_file = floor(max_pixels_file/(length_edge^(2)*offset_frames));

disp(['total frames/frames per file: ', ...
    num2str(ind_frames*offset_frames), '/', ...
    num2str(indiv_frames_per_file*offset_frames)])
disp(['writing ', num2str(num_tif_files), ' .tif files.'])
%disp([(ind_frames*offset_frames), (indiv_frames_per_file*offset_frames)])
%disp('total files: ')
%disp(num_tif_files)

ind_frames_to_write = ind_frames;

fileBorders = ones(num_tif_files, 2);
for i = 1:num_tif_files

    start_k = 1 + (i-1) * (indiv_frames_per_file);  % Starting frame for the i-th file
    end_k = start_k + min((indiv_frames_per_file-1), ind_frames_to_write-1);  % Ending frame (either 300 or last frame)
    
    % Store the borders 
    fileBorders(i, :) = [start_k, end_k];

    ind_frames_to_write = ind_frames_to_write - indiv_frames_per_file;
    start_k = start_k+1;
end

%% Create filter masks:
mask_first = find(data.dt>= first_dt(1) & data.dt <= first_dt(2));
mask_second = find(data.dt >= second_dt(1) & data.dt <= second_dt(2));


%% Get first dt photons 
data1.x = data.x(mask_first);
data1.y = data.y(mask_first);
data1.dt = data.dt(mask_first);
data1.ms = data.ms(mask_first);

%% Get second dt photons
data2.x = data.x(mask_second);
data2.y = data.y(mask_second);
data2.dt = data.dt(mask_second);
data2.ms = data.ms(mask_second);


%clear data


%% Create new ms index for dt1 and dt2

cumulative_photons_dt1 = count_photons(data1.ms);
cumulative_photons_dt2 = count_photons(data2.ms);



%% Writing dt 1 tif and creating ms index
my_ms_dt1 = ones(ind_frames+1, offset_frames);

for i = 1:offset_frames
    frame_off = (i-1)*ms_offset;
    my_ms_dt1(:,i) = cumulative_photons_dt1(1+frame_off:int_time:end);
end

%indiv_frames_written = 0;
for i = 1:num_tif_files
    
    disp('starting to write dt1 tiff...')
    % set extension and metadata
    if i == 1
        extension = 'x_dt1.ome.tif';
    else
        extension = ['x_dt1_', num2str(i-1), '.ome.tif'];
    end
    my_tiff = Tiff([savePath(1:end-8), '_', num2str(int_time), ...
        'ms_', num2str(pixels), 'px_', num2str(offset_frames), extension], 'w');
    tagstruct.Compression = Tiff.Compression.None;
    tagstruct.ImageLength = size(img0,1);
    tagstruct.ImageWidth = size(img0,2);
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 8;
    tagstruct.SampleFormat = Tiff.SampleFormat.UInt;

    for k = fileBorders(i,1):fileBorders(i,2)
        
        %if ismember(k, fileBorders)
        %    disp(k)
        %end

        if mod(k,1000) == 0
            frame_counter =  ['Written dt1 full frames ', num2str(k), ' of ', num2str(ind_frames)];
            disp(frame_counter)
        end
        for m = 1:offset_frames
            %disp('m value is: ')
            %disp(m)
            j1 = my_ms_dt1(k,m);
            j2 = my_ms_dt1(k+1,m);
            
            min_image = 0;
            max_image = (fov_in_pixels*binning)-1;
            img_tmp = photonscore.hist_2d(data1.x(j1+1:j2), min_image, max_image, ...
                fov_in_pixels, data1.y(j1+1:j2));
            my_img = uint8(img_tmp);
        
            my_tiff.setTag(tagstruct)
            my_tiff.write(my_img);
            my_tiff.writeDirectory();
        end
    end
    my_tiff.close;
    disp(['Written .tiff file: ', num2str(i)])
end

clear data1

% set datatypes: single for x,y, uint16 for dt; uint32 for ms

data.x = single(data.x);
data.y = single(data.y);

data.x = (data.x/binning);
data.y = (data.y/binning);

data.dt = uint16(data.dt);
data.ms = uint32(data.ms);

%% Shift data.x and data.y according to ROI and binning
save_x1 = data.x; %switch x and y because of TIF
save_y1 = data.y; %switch x and y because of TIF
data.x = save_y1;
data.y = save_x1;

%Create hdf5 file with /photons dataset, overwrite existing file 
savePathInd       = strcat(savePath(1:end-8), '_', num2str(offset_frames), 'x_dt_all_index.hdf5');
DATASET        = '/photons';
dims          = length(data.x);
file = H5F.create (savePathInd, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% Assign the required data types (uint16 for first 3 columns, uint32 for last)
doubleType1=H5T.copy('H5T_NATIVE_FLOAT');
sz(1)     =H5T.get_size(doubleType1);
doubleType2=H5T.copy('H5T_NATIVE_FLOAT');
sz(2)     =H5T.get_size(doubleType2);
doubleType3=H5T.copy('H5T_NATIVE_USHORT');
sz(3)     =H5T.get_size(doubleType3);
doubleType4=H5T.copy('H5T_NATIVE_UINT');
sz(4)     =H5T.get_size(doubleType4);

% Compute the offsets to each field. The first offset is always zero.
offset(1)=0;
offset(2:4)=cumsum(sz(1:3));

% Create the compound datatype for memory.
memtype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert (memtype,'x',offset(1), doubleType1);
H5T.insert (memtype,'y',offset(2), doubleType2);
H5T.insert (memtype,'dt',offset(3), doubleType3);
H5T.insert (memtype,'ms',offset(4), doubleType4);

%Create the compound datatype for file with the required fields - 'x', 'y', 'dt',
%'ms'
filetype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert (filetype, 'x', offset(1),doubleType1);
H5T.insert (filetype, 'y', offset(2), doubleType2);
H5T.insert (filetype, 'dt',offset(3), doubleType3);
H5T.insert (filetype, 'ms',offset(4), doubleType4);

% Create dataspace.  Setting maximum size to [] sets the maximum
% size to be the current size.
space = H5S.create_simple (1,fliplr( dims), []);

% Create the dataset and write the compound data to it.
dset = H5D.create (file, DATASET, filetype, space, 'H5P_DEFAULT');
H5D.write (dset, memtype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', data);

% Close and release resources.
H5D.close (dset);
H5S.close (space);
H5T.close (filetype);
H5F.close (file);

message = ['dt1 photon_index created as hdf5. Containing ', num2str(dims), ' photons.'];
disp(message)




%% Writing dt 2 tif and creating ms index
my_ms_dt2 = ones(ind_frames+1, offset_frames);

for i = 1:offset_frames
    frame_off = (i-1)*ms_offset;
    my_ms_dt2(:,i) = cumulative_photons_dt2(1+frame_off:int_time:end);
end

%indiv_frames_written = 0;
for i = 1:num_tif_files

    disp('starting to write dt2 tif...')
    % set extension and metadata
    if i == 1
        extension = 'x_dt2.ome.tif';
    else
        extension = ['x_dt2_', num2str(i-1), '.ome.tif'];
    end
    my_tiff = Tiff([savePath(1:end-8), '_', num2str(int_time), ...
        'ms_', num2str(pixels), 'px_', num2str(offset_frames), extension], 'w');
    tagstruct.Compression = Tiff.Compression.None;
    tagstruct.ImageLength = size(img0,1);
    tagstruct.ImageWidth = size(img0,2);
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 8;
    tagstruct.SampleFormat = Tiff.SampleFormat.UInt;

    for k = fileBorders(i,1):fileBorders(i,2)
        
        %if ismember(k, fileBorders)
        %    disp(k)
        %end

        if mod(k,1000) == 0
            frame_counter =  ['Written dt2 full frames ', num2str(k), ' of ', num2str(ind_frames)];
            disp(frame_counter)
        end
        for m = 1:offset_frames
            %disp('m value is: ')
            %disp(m)
            j1 = my_ms_dt2(k,m);
            j2 = my_ms_dt2(k+1,m);
            
            min_image = 0;
            max_image = (fov_in_pixels*binning)-1;
            img_tmp = photonscore.hist_2d(data2.x(j1+1:j2), min_image, max_image, ...
                fov_in_pixels, data2.y(j1+1:j2));
            my_img = uint8(img_tmp);
        
            my_tiff.setTag(tagstruct)
            my_tiff.write(my_img);
            my_tiff.writeDirectory();
        end
    end
    my_tiff.close;
    disp(['Written .tiff file: ', num2str(i)])
end

% set datatypes: single for x,y, uint16 for dt; uint32 for ms

data2.x = single(data2.x);
data2.y = single(data2.y);
data2.x = (data2.x/binning);
data2.y = (data2.y/binning);
data2.dt = uint16(data2.dt);
data2.ms = uint32(data2.ms);

%% Shift data.x and data.y according to ROI and binning
save_x2 = data2.x; %switch x and y because of TIF
save_y2 = data2.y; %switch x and y because of TIF
data2.x = save_y2;
data2.y = save_x2;

%Create hdf5 file with /photons dataset, overwrite existing file 
savePathInd       = strcat(savePath(1:end-8), '_', num2str(offset_frames), 'x_dt2_index.hdf5');
DATASET        = '/photons';
dims          = length(data2.x);
file = H5F.create (savePathInd, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% Assign the required data types (uint16 for first 3 columns, uint32 for last)
doubleType1=H5T.copy('H5T_NATIVE_FLOAT');
sz(1)     =H5T.get_size(doubleType1);
doubleType2=H5T.copy('H5T_NATIVE_FLOAT');
sz(2)     =H5T.get_size(doubleType2);
doubleType3=H5T.copy('H5T_NATIVE_USHORT');
sz(3)     =H5T.get_size(doubleType3);
doubleType4=H5T.copy('H5T_NATIVE_UINT');
sz(4)     =H5T.get_size(doubleType4);

% Compute the offsets to each field. The first offset is always zero.
offset(1)=0;
offset(2:4)=cumsum(sz(1:3));

% Create the compound datatype for memory.
memtype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert (memtype,'x',offset(1), doubleType1);
H5T.insert (memtype,'y',offset(2), doubleType2);
H5T.insert (memtype,'dt',offset(3), doubleType3);
H5T.insert (memtype,'ms',offset(4), doubleType4);

%Create the compound datatype for file with the required fields - 'x', 'y', 'dt',
%'ms'
filetype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert (filetype, 'x', offset(1),doubleType1);
H5T.insert (filetype, 'y', offset(2), doubleType2);
H5T.insert (filetype, 'dt',offset(3), doubleType3);
H5T.insert (filetype, 'ms',offset(4), doubleType4);

% Create dataspace.  Setting maximum size to [] sets the maximum
% size to be the current size.
space = H5S.create_simple (1,fliplr( dims), []);

% Create the dataset and write the compound data to it.
dset = H5D.create (file, DATASET, filetype, space, 'H5P_DEFAULT');
H5D.write (dset, memtype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', data2);

% Close and release resources.
H5D.close (dset);
H5S.close (space);
H5T.close (filetype);
H5F.close (file);

message = ['dt2 photon_index created as hdf5. Containing ', num2str(dims), ' photons.'];
disp(message)