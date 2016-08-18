function prediction = VideoRGBPrediction(rgb_path, net, mean_pixel, num_class)
% Input:
%   rgb_path: the folder holding the rgb frames of a video
%   net: an instance of the Net class of Caffe
%   num_class: total number of class candidates
% Output:
%   prediction: a matrix of size [num_class, 10*25] showing the scores of each frame's 10 crops

% parameter
num_frame = 25;
batch_size = 50;

imglist = dir([rgb_path, '/*.jpg']);
duration = length(imglist);
step = floor((duration-1)/(num_frame-1));

rgb = zeros(256, 340, 3, num_frame,'single');
rgb_flip = zeros(256, 340, 3, num_frame,'single');

% selection
for i = 1:num_frame
    img = single(imread(sprintf('%s/image_%04d.jpg', rgb_path, (i-1)*step+1)));
    rgb(:,:,:,i) = img;
    rgb_flip(:,:,:,i) = img(:,end:-1:1,:);
end

% crop
rgb_1 = rgb(1:224,1:224,:,:);
rgb_2 = rgb(1:224,end-223:end,:,:);
rgb_3 = rgb(16:16+223,59:282,:,:);
rgb_4 = rgb(end-223:end,1:224,:,:);
rgb_5 = rgb(end-223:end,end-223:end,:,:);
rgb_f_1 = rgb_flip(1:224,1:224,:,:);
rgb_f_2 = rgb_flip(1:224,end-223:end,:,:);
rgb_f_3 = rgb_flip(16:16+223,59:282,:,:);
rgb_f_4 = rgb_flip(end-223:end,1:224,:,:);
rgb_f_5 = rgb_flip(end-223:end,end-223:end,:,:);

rgb = cat(4,rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5);

% substract mean and permute
IMAGE_MEAN = reshape(mean_pixel,[1,1,3]);
rgb = bsxfun(@minus,rgb(:,:,[3,2,1],:),IMAGE_MEAN);
rgb = permute(rgb,[2,1,3,4]);

% predict
prediction = zeros(num_class,size(rgb,4));
num_batches = ceil(size(rgb,4)/batch_size);
rgbs = zeros(224,224,3,batch_size,'single');
for bb = 1:num_batches
    range = 1 + batch_size*(bb-1): min(size(rgb,4),batch_size*bb);
    rgbs(:,:,:,mod(range-1,batch_size)+1) = rgb(:,:,:,range);
    net.blobs('data').set_data(rgbs);
    net.forward_prefilled();
    out_put = squeeze(net.blobs('fc').get_data());
    prediction(:,range) = out_put(:,mod(range-1,batch_size)+1);
end


end
