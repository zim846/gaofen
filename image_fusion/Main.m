%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Implementation of "Visible and NIR Image Fusion using Weight Map Guided Laplacian-Gaussian Pyramid for Improving Scene Visibility"
% %
% % written by Ashish V. Vanmali, IIT Bombay, India
% % e-mail: ashishvanmali@iitb.ac.in, vanmaliashish@gmail.com
% %
% % Last Updated - June 2015
% %
% % This work is submitted to -
% % Sadhana - Academy Proceedings in Engineering Sciences 
% %
% % The input images are downloded from EPFL database
% % available at - http://ivrg.epfl.ch/supplementary_material/cvpr11/index.html
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;

RGBfiles=scanfile('E:\data_set\sea_pic');
NIRfiles=scanfile('E:\data_set\sea_nir');
for j=1:length(RGBfiles)
%% Select the inputs
file1 = char(RGBfiles(j));
file2 = char(NIRfiles(j));
I1_RGB = double(imread(file1)) / 255;
I2 = double(rgb2gray(imread(file2))) / 255;
tic %% Start Measuring time

[H S I1] = rgb2hsv(I1_RGB);

%% Weigt Control Parameter
alpha = [1 1 1];

%% Parameters
size1 = 5; size2 = 5;
sigma1 = 2; sigma2 = 2;
window1 = 5;        %% Local Contrast
window2 = 5;        %% Local Entropy
NHOOD1 = ones(window1);
NHOOD2 = ones(window2);

%% Weights for Visible image / NIR Image
%% Local Contrast
C1 = stdfilt(I1, NHOOD1);
C2 = stdfilt(I2, NHOOD1);
%% Local Entropy
J1 = entropyfilt(I1,NHOOD2)/8;
J2 = entropyfilt(I2,NHOOD2)/8;
%% Visibility  => Locally Normalizd Luminance
gaussian1 = fspecial( 'gaussian', size1, sigma1 ) ;
gaussian2 = fspecial( 'gaussian', size2, sigma2 ) ;

IM1 = imfilter( I1, gaussian1, 'replicate' ) ;
noise1 = I1 - IM1;
Vis1 = sqrt(imfilter( noise1.^2, gaussian2, 'replicate' ) ) ;

IM2 = imfilter( I2, gaussian1, 'replicate' ) ;
noise2 = I2 - IM2 ;
Vis2 = sqrt(imfilter( noise2.^2, gaussian2, 'replicate' ) ) ;

%% Final Weight
W1 = (C1.^alpha(1)).*(J1.^alpha(2)).*(Vis1.^alpha(3));
W2 = (C2.^alpha(1)).*(J2.^alpha(2)).*(Vis2.^alpha(3));
disp(size(W1));

%% Normalize weights: make sure that weights sum to one for each pixel
W(:,:,1) = W1;
disp(size(W));
W(:,:,2) = W2;
W = W + 1e-12; %avoids division by zero
W = W./repmat(sum(W,3),[1 1 2]) ;

%% Exposure Fusion(Here We use 'Exposure Fusion' code of Tom Mertens with some modifications)
I(:,:,1) = I1;
I(:,:,2) = I2;

r = size(I,1);
c = size(I,2);
N = 2;

% create empty pyramid
pyr = gaussian_pyramid(zeros(r,c,1));
nlev = length(pyr);

% multiresolution blending
for i = 1:N
    % construct pyramid from each input image
    pyrW = gaussian_pyramid(W(:,:,i));
    pyrI = laplacian_pyramid(I(:,:,i));
    
    % blend
    for l = 1:nlev
        w = repmat(pyrW{l},[1 1 1]);
        pyr{l} = pyr{l} + w.*pyrI{l};
    end
end

% reconstruct
Recon = reconstruct_laplacian_pyramid(pyr);

%% Put back fused component
I4(:,:,1) = H;
I4(:,:,2) = S;
I4(:,:,3) = Recon;
I5 = hsv2rgb(I4);

toc  %% display time without post processing

%% Inverse Tone Mapping
beta = 1.5;     %% beta preset to 1.5
I_out1(:,:,1) = (I5(:,:,1)./(Recon+1e-12)).^beta .* (I1+I2)/2;
I_out1(:,:,2) = (I5(:,:,2)./(Recon+1e-12)).^beta .* (I1+I2)/2;
I_out1(:,:,3) = (I5(:,:,3)./(Recon+1e-12)).^beta .* (I1+I2)/2;


%% Sharpen the reconstructed I
[Hn Sn Vn] = rgb2hsv((I_out1));
hsharp = [-1 -1 -1; -1 8 -1; -1 -1 -1] / 3;%construct a sharpening mask
Recon_sharp = imfilter(Vn,hsharp,'replicate');
V2 = Vn + 0.4*Recon_sharp ; %% lambda preset to 0.4
%% Put back fused component
I6(:,:,1) = Hn;
I6(:,:,2) = Sn;
I6(:,:,3) = V2;
I_out = hsv2rgb(I6);

toc  %% Display time taken with post processing

%% Display Weight Maps
%% figure;
%% subplot(221); imshow(I1,[]); title('VIS');
%% subplot(222); imshow(I2,[]); title('NIR');
%% subplot(223); imshow(W(:,:,1),[]); title('Weight Map VIS');
%% subplot(224); imshow(W(:,:,2),[]); title('Weight Map NIR');
%% Display Results
%% figure; imshow(I1_RGB,[]); title('Visible Image');
%% figure; imshow(I2,[]); title('NIR Image');
%% figure; imshow(I5,[]); title('Fused Image without Post Processing');
%% figure; imshow(I_out,[]); title('Fused Image with Post Processing');
str1 = 'E:\data_set\sea_com\';
str2 = num2str(j);
str3 = '_com.png';
SC = [str1,str2,str3];
imwrite(I_out, SC);
clear W;
clear I;
clear I4;
clear I_out1;
clear I6;
end
