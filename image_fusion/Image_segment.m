%% Segmentation of MAD Images to detect new objects
function I_out = MAD_segment(I)

%% Detect Entire Cell
edge_method ='canny';   %% For Forest marius
% edge_method ='sobel';   %% For arch garden puppet
[junk threshold] = edge(I, edge_method);
fudgeFactor = 1 ;
BWs = edge(mat2gray(I),edge_method, threshold * fudgeFactor);
% figure, imshow(BWs), title('binary gradient mask');

%% Dilate the Image
% se90 = strel('line', 7, 90);
% se0 = strel('line', 7, 0);
se90 = strel('square', 7);         % forest 3 Puppet 10/7 arch 7 Garden 7 marius 5 park 5
se0 = strel('square', 7);
BWsdil = imdilate(BWs, [se90 se0]);
% figure, imshow(BWsdil), title('dilated gradient mask');

%% Fill Interior Gaps
BWdfill = imfill(BWsdil, 'holes');
% figure, imshow(BWdfill); title('binary image with filled holes');

% %% Remove Connected Objects on Border
% BWnobord = imclearborder(BWdfill, 4);
% % figure, imshow(BWnobord), title('cleared border image');
BWnobord = BWdfill;

%% Smoothen the Object
seD = strel('diamond',5);
BWfinal = imerode(BWnobord,seD);
BWfinal = imerode(BWfinal,seD);
% figure, imshow(BWfinal), title('segmented image');

I_out = BWfinal;