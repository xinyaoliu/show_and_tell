require 'paths'
require 'loadcaffe'
require 'nn'
require 'cunn'
require 'image'
require 'optim'
require 'gnuplot'

paths.dofile('train_data.lua')
paths.dofile('preprocess_data.lua')
paths.dofile('dataloader.lua')

classes = {'airplane', 'automobile', 'bird', 'cat',
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck' }


cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-gpuid', 3, 'whic gpu to use, -1 for cpu')
cmd:option('-epochs', 60, 'number of epochs')
cmd:option ('-minibatch', 30, 'size of a minibatch')
cmd:option('-iteration', 49980, 'number of iterations at each training pass')
cmd:option ('-titer', 9990, 'number of iterations at each evaluation pass')
cmd:option ('-model', 'nil', 'name of the trained model to be loaded')
cmd:text()
opt = cmd:parse(arg)

timer = torch.Timer()

model = load_vgg_net()

if opt.gpuid >=0 then
	vgg_net:cuda()
	print(vgg_net)

else
	print(vgg_net)
end

print ('[loading vgg_net] time elapse: ' .. timer:time().real)

-- if opt.model then
-- 	-- load_model (opt.mod, vgg_net)
-- 	vgg_net = torch.load (opt.model)
-- 	print (vgg_net)
-- end

-- load datasets

load_data()

-- preprocess datasets
--preprocess_data (trainset.data, 'data/train/')
--preprocess_data (testset.data, 'data/test/')

train(vgg_net)




