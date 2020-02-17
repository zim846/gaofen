%% Function for RMS Contrast
function Iout = rms_contast(I,W)

[R C] = size(I);
xpad = (W-1)/2;
ypad = (W-1)/2;
% Pad image with border pixels
Ipad = padarray(I, [xpad ypad], 'replicate');
% Windowed RMS contrast
Iout = zeros(R,C);
for x = 1+xpad : R+xpad
    for y = 1+ypad : C+ypad
        Isub = Ipad(x-xpad:x+xpad, y-ypad:y+ypad);
        mu = mean(Isub(:));
        Iout(x-xpad,y-ypad) = sqrt(sum(sum((Isub-mu)^2))/(W));
%         Iout(x-xpad,y-ypad) = var(Isub(:));
    end
end