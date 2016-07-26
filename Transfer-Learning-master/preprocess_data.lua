function preprocess_data (image_set, fname)
	for i = 1, image_set:size(1) do
		-- scaling
		base_img = image_set:select(1, i)
		res_img = base_img:resize (3,32,32)
		mod_img = image.scale (res_img, 224, 224)

		-- normalizing
		for j = 1, 3 do
			img_mean = mod_img [{ {j}, {}, {} }]:mean()
			img_std = mod_img [{ {j}, {}, {} }]:std()
			mod_img [{ {j}, {}, {} }]:add(-img_mean)
			mod_img [{ {j}, {}, {} }]:div(img_std)
		end

		-- transform: RGB to BGR
		chan_r = mod_img [{ {1}, {}, {} }]
		chan_g = mod_img [{ {2}, {}, {} }]
		chan_b = mod_img [{ {3}, {}, {} }]

		mod_img = torch.cat (torch.cat (chan_b, chan_g, 1), chan_r, 1)

		-- save
		image.save ("data/" .. fname .. i .. ".png", mod_img)
	end
end