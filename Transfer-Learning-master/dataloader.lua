function load_vgg_net()


	prototxt_name = '/home/liuxinyao/models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt'
	binary_name = '/home/liuxinyao/models/vgg/VGG_ILSVRC_16_layers.caffemodel'

	if opt.gpuid >=0 then
		vgg_net = loadcaffe.load(prototxt_name, binary_name, 'cudnn')
	else
		vgg_net = loadcaffe.load(prototxt_name, binary_name, 'nn')
	end

	vgg_net:remove(vgg_net:size())  --40
	vgg_net:remove(vgg_net:size())  --39

	vgg_net:add(nn.Linear(4096, 10))   --39
	vgg_net:add(nn.LogSoftMax())  --40
	--vgg_net:add(nn.Tanh())  --41

	return vgg_net
end






function load_data ()
	-- if not paths.filep("cifar-10-torch.tar.gz") then
	--     os.execute('wget http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz')
	--     os.execute('tar xvf cifar-10-torch.tar.gz')
	-- end
	
	trainset = {
		data = torch.Tensor (50000, 32*32*3),
		labels = torch.Tensor (50000)
	}

	for i = 0, 4 do
		subset = torch.load ('cifar-10-batches-t7/data_batch_' .. (i + 1) .. '.t7', 'ascii')
		trainset.data [ { { i*10000 + 1, (i+1)*10000 } } ] = subset.data:t()
		trainset.labels [ { { i*10000 + 1, (i+1)*10000 } } ] = subset.labels
	end

	trainset.labels = trainset.labels + 1

	subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
	testset = {
		data = subset.data:t():double(),
		labels = subset.labels[1]:double()
	}
	testset.labels:add(1)

	print ('[loading data] time elapse: ' .. timer:time().real)


	print(trainset)
	


end

