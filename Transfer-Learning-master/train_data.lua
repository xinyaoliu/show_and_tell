function train(net)

	epochs = opt.epochs
	batch_size = opt.minibatch

	timer:reset()
	n_inputs = math.floor (opt.iteration / opt.minibatch)

	tr_losses = {}
	te_losses = {}
	tab = {}
	tr_acctab = {}
	te_acctab = {}
	epoch_tab = {}



	for i = 1, epochs do
		print ('------------------------------------------------------------------------')
		print ("	[[[ Epoch  " .. i .. ' / ' .. epochs ..' ]]]')
		print ('------------------------------------------------------------------------')

		_, train_loss, learn_rate, tr_acc = training (net, trainset.labels, 'train/processed/')
	

		test_loss, te_acc = testing (net, testset.labels, 'test/processed/')

		
		table.insert (epoch_tab, i)
		table.insert (tr_losses, train_loss[#train_loss])
		table.insert (te_losses, test_loss[#test_loss])
		table.insert (tr_acctab, tr_acc[#tr_acc])
		table.insert (te_acctab, te_acc[#te_acc])

		plot (epoch_tab, torch.Tensor(tr_losses), 'training_loss_' .. learn_rate, 'Loss')
		plot (epoch_tab, torch.Tensor(te_losses), 'test_loss', 'Loss')
		plot (epoch_tab, torch.Tensor(tr_acctab), 'training_accuracy', 'Accuracy')
		plot (epoch_tab, torch.Tensro(te_acctab), 'test_accuracy', 'Accuracy')

	end

	netsav = net:clone('weight', 'bias') 
	torch.save ('vgg_fine_tuned.t7', netsav)


end




function training (vgg_net, image_labels, fname)
	batch_size = opt.minibatch
	max_iter = opt.iteration
	epochs = opt.epochs
	conf_mat = torch.DoubleTensor (10, 10):zero()

	-- file = torch.DiskFile ('train_confusion.txt', 'w')

	print ('Start training......................................................')
	-- file:writeObject ('Start training...')
	print ('................................................................')

	if opt.gpuid >=0 then
		-- c_image_labels = image_labels:cuda()
		criterion = nn.ClassNLLCriterion():cuda()
	else
		criterion = nn.ClassNLLCriterion()
	end
	
	-- confusion = optim.ConfusionMatrix(10)

	params, grad_params = vgg_net:getParameters()

	print ("[getParameters] time elapse: " .. timer:time().real)

	time_vals = {}
	loss_vals = {}
	accs = {}
	output = torch.DoubleTensor (batch_size, 10)

	-- randomize inputs and targets
	indices = torch.randperm(50000)

	for i = 1, max_iter, batch_size do
		print ("	[[[ Batch number " .. math.ceil (i/batch_size) .. ' / ' .. opt.iteration/batch_size*epochs .. ' ]]]')

		inputs = torch.DoubleTensor (batch_size, 3, 224, 224)
		targets = torch.DoubleTensor (batch_size)

		for bat = i, math.min(max_iter, i+batch_size-1) do
			idx = indices[bat]

			img = image.load ('data/' .. fname .. idx .. '.png', 3, 'double')
			img = img:resize (3, 224, 224)

			if bat%batch_size ~= 0 then
				inputs[bat%batch_size]:copy (img)
				targets[bat%batch_size] = image_labels[idx]
			else
				inputs[batch_size]:copy(img)
				targets[batch_size] = image_labels[idx]
			end

		end

		c_inputs = inputs:cuda ()
		c_targets = targets:cuda ()

--		optim_state = { learning_rate = 0.00000000001, momentum = 0.9, weight_decay = 5e-4 }
		-- optim_state = { learning_rate = 0.000001 }
		-- optim_state = { learningRate = 0.0000000001, weightDecay = 0.0005 }
		weight_decay = 0.005
		learning_rate = 0.0005

		-- training
		grad_params:zero()

		-- evaluate function for the entire minibatch
		output = vgg_net:forward (c_inputs)
		loss = criterion:forward (output, c_targets)

		-- estimate df/dW
		dloss_dout = criterion:backward (output, c_targets)
		vgg_net:backward (c_inputs, dloss_dout)


		grad_params:clamp(-5, 5) --


		-- vanilla update the weights
		params:add(grad_params:mul(-learning_rate))
	
		-- measure the accuracy
		print ("[before measure_acc] time elapse: " .. timer:time().real)
		conf_mat, accuracy = measure_acc (conf_mat, output, targets)

	
			print ('params norm: ' .. params:norm())
			print ('grad_params norm: ' .. grad_params:norm())

			print ("Err : " .. loss)
			print ("Time: " .. timer:time().real)
			print ('Accuracy: ' .. accuracy)

			-- visualization
			table.insert (time_vals, timer:time().real)
			table.insert (accs, accuracy)
			
			bound = 50
			if loss < bound then
				table.insert (loss_vals, loss)
			else
				table.insert (loss_vals, bound)
			end


			print ('................................................................' .. i+batch_size-1)
			-- file:writeObject ("current loss: " .. loss)
		-- end
	end

	-- print (confusion)
	print ('Accuracy: ' .. accuracy)
	print ("Err : " .. loss)
	print ("Time: " .. timer:time().real)



	return time_vals, loss_vals, learning_rate, accs
end


function measure_acc (mat, output, targets)		-- mat : 10 x 10

	for i = 1, opt.minibatch do
		max = -math.huge
		for ind = 1, 10 do 	-- for each class
			if output[i][ind] > max then
				max_ind = ind
				max = output[i][ind]
			end
		end

		mat[targets[i]][max_ind] = mat[targets[i]][max_ind] + 1
	end

	correct = 0
	for i = 1, 10 do
		correct = correct + mat[i][i]
	end
	global_correct = correct / mat:sum() * 100

	print ('[measuring accuracy] time elapse: ' .. timer:time().real)

	return mat, global_correct

end


function plot (time, val, fname, ylabel)
	fpath = 'plots/'

	-- if torch.type (arb) == 'number' then
		-- fname = 'loss' .. learn_rate .. '.png'
		fname = fname .. '.png'

		gnuplot.pngfigure (fpath .. fname)

		gnuplot.plot (time, val, '-')
		-- gnuplot.xlabel ('time (s)')
		gnuplot.xlabel ('Epochs')
		gnuplot.ylabel (ylabel)

		gnuplot.plotflush ()
end