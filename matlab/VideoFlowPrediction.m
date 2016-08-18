function prediction = VideoFlowPrediction(flow_path, net, num_class)
% Input:
%   flow_path: the folder holding the flow frames of a video
%   net: an instance of the Net class of Caffe
%   num_class: total number of class candidates
% Output:
%   prediction: a matrix of size [num_class, 10*25] showing the scores of each frame's 10 crops

num = 25;
L = 5;

imglist = dir([flow_path,'/*flow_x*.jpg']);
duration = length(imglist);
step = floor((duration-L)/(num-1));
flow = zeros(256,340,L*2,num,'single');
flow_flip =  zeros(256,340,L*2,num,'single');

% selection
for i = 1:num
	for j = 1:L
		img_x = single(imread(sprintf('%s/flow_x_%04d.jpg',flow_path,(i-1)*step+j)));
		img_y = single(imread(sprintf('%s/flow_y_%04d.jpg',flow_path,(i-1)*step+j)));
		flow(:,:,(j-1)*2+1,i) = img_x;
		flow(:,:,(j-1)*2+2,i) = img_y;
		flow_flip(:,:,(j-1)*2+1,i) = 255-img_x(:,end:-1:1);
		flow_flip(:,:,(j-1)*2+2,i) = img_y(:,end:-1:1);
	end
end

% crop
flow_1 = flow(1:224,1:224,:,:);
flow_2 = flow(1:224,end-223:end,:,:);
flow_3 = flow(16:16+223,59:282,:,:);
flow_4 = flow(end-223:end,1:224,:,:);
flow_5 = flow(end-223:end,end-223:end,:,:);
flow_f_1 = flow_flip(1:224,1:224,:,:);
flow_f_2 = flow_flip(1:224,end-223:end,:,:);
flow_f_3 = flow_flip(16:16+223,59:282,:,:);
flow_f_4 = flow_flip(end-223:end,1:224,:,:);
flow_f_5 = flow_flip(end-223:end,end-223:end,:,:);

flow= cat(4,flow_1,flow_2,flow_3,flow_4,flow_5,flow_f_1,flow_f_2,flow_f_3,flow_f_4,flow_f_5);

% substract mean
flow(:) = flow(:) - 128;
flow = permute(flow,[2,1,3,4]);

% test
prediction = zeros(num_class,size(flow,4));
batch_size = 50;
num_batches = ceil(size(flow,4)/batch_size);
flows = zeros(224,224,2*L,batch_size,'single');

for bb = 1:num_batches
	range = 1 + batch_size*(bb-1): min(size(flow,4),batch_size*bb);
	flows(:,:,:,mod(range-1,batch_size)+1) = flow(:,:,:,range);
	out_put = net.forward({flows});
	out_put = squeeze(out_put{1});
	prediction(:,range) = out_put(:,mod(range-1,batch_size)+1);
end

end
