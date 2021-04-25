%%% Obtain Scaterring coefficients of the input images from a scattering network
%%% Used toolbox

data_path = 'CXR_testImages/';

tfiles=dir([data_path '*.mat']);

parfor k=1:numel(tfiles)
    if (mod(k, 50)) == 0
        k
    end
    process(data_path, tfiles, k); 
end

function process(data_path, tfiles, k)
%    save(['US_SC/' name],'S');
    load([data_path tfiles(k).name], 'img'); % for CXR
    filename = tfiles(k).name;
    name = filename(1:end-4);
    img = imresize(img,[512 512]);

    Wop = wavelet_factory_2d(size(img));

    St = scat(double(img), Wop);
    S = format_scat(St);
    save(['/sc_bcet/' name], 'S');
end
