%%% Obtain Scaterring coefficients of the input images from a scattering network
%%% Used toolbox

data_path = 'US_testImages/';

tfiles=dir([data_path '*.mat']);

parfor k=1:numel(tfiles)
    if (mod(k, 50)) == 0
        k
    end
    img = imread([data_path tfiles(k).name]);
    filename = tfiles(k).name;
    name = filename(1:end-4);
    img = imresize(img,[512 512]);

    Wop = wavelet_factory_2d(size(img));

    St = scat(double(img), Wop);
    S = format_scat(St);
end

function saver(name)
    save(['sc_bcet/' name], 'S');
end
