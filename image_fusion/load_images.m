%% Function to read set of images
% Input arguments
%   N    = No. of Images
%   path = folder name in which images are stored
%   type = format 0f the image
% Return Value
%   I = Stack of N color images (at double precision). Dimensions are (height x width x 3 x N).

function I = load_images(N,path,type)

for i = 1 : N
    filename = ['' path '/' num2str(i) '.' type];
    im = double(imread(filename)) / 255;
    I(:,:,:,i) = im;
end