

require 'image'
local gm = require 'graphicsmagick'

image_test =  gm.Image()
image_test:load('./raindrop0201.jpg')
-- print(image_test)
-- image_test:save('file.jpg')
-- image_test:show()
image_test:save('trinh.jpg')


local radius = 0 
local sigma = 1.0 --torch.uniform(0.5,1.5)
local amount = 1.0--torch.uniform(0.1, 0.9)
local threshold = 0.05 --torch.uniform(0.0, 0.05)



local alo =  image_test:unsharpMask(radius, sigma, amount, threshold)
alo:save('trinh_02.jpg')



image_test:sharpen(1,5)
image_test:save('trinh_03.jpg')
