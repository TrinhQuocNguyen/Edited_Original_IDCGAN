-- usage: DATA_ROOT=/path/to/data/ name=expt1 which_direction=BtoA th test.lua
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'image'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')
torch.setdefaulttensortype('torch.FloatTensor')





opt = {
    DATA_ROOT = '',           -- path to images (should have subfolders 'train', 'val', etc)
    batchSize = 1,            -- # images in batch
    loadSize = 640,           -- scale images to this size
    fineSize = 640,           --  then crop to this size
    fineSize1 = 640,           --  then crop to this size
    flip=0,                   -- horizontal mirroring data augmentation
    display = 1,              -- display samples while training. 0 = false
    display_id = 200,         -- display window id.
    gpu = 1,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    how_many = 'all',         -- how many test images to run (set to all to run on every image found in the data/phase folder)
    which_direction = 'BtoA', -- AtoB or BtoA (BtoA : original <- noise)
    phase = 'val',            -- train, val, test ,etc
    preprocess = 'regular',   -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
    aspect_ratio = 1.0,       -- aspect ratio of result images
    name = '',                -- name of experiment, selects which model to run, should generally should be passed on command line
    input_nc = 3,             -- #  of input image channels
    output_nc = 3,            -- #  of output image channels
    serial_batches = 1,       -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,    -- iter into serial image list
    cudnn = 1,                -- set to 0 to not use cudnn (untested)
    checkpoints_dir = './model', -- loads models from here
    results_dir='./results/',          -- saves results here
    which_epoch = '900',            -- which epoch to test? set to 'latest' to use latest cached model
    display = false,            -- display samples while training. 0 = false
    display_id = 10,        -- display window id.
    display_plot = 'errL1',    -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
    display_freq = 100,          -- display the current results every display_freq iterations
    save_display_freq = 5000,    -- save the current display of results every save_display_freq_iterations
    load_image_separately = true,
    noise_folder_name = '/noise',
    original_folder_name = '/original',

}


-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1 -- test only works with 1 thread...

-- print(opt)

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- set seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.netG_name = opt.name .. '/' .. opt.which_epoch .. '_net_G'

-- local data_loader = paths.dofile('data/data.lua')
-- print('#threads...' .. opt.nThreads)
-- local data = data_loader.new(opt.nThreads, opt)
-- print("Dataset Size: ", data:size())

-- translation direction
-- local idx_A = nil
-- local idx_B = nil
-- local input_nc = opt.input_nc
-- local output_nc = opt.output_nc
-- if opt.which_direction=='AtoB' then
--   idx_A = {1, input_nc}
--   idx_B = {input_nc+1, input_nc+output_nc}
-- elseif opt.which_direction=='BtoA' then
--   idx_A = {input_nc+1, input_nc+output_nc}
--   idx_B = {1, input_nc}
-- else
--   error(string.format('bad direction %s',opt.which_direction))
-- end
----------------------------------------------------------------------------

-- local input = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
-- local target = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

print('checkpoints_dir', opt.checkpoints_dir)
local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)
--netG:evaluate()

print(netG)

-- add filepath_curr to filepaths
-- function TableConcat(t1,t2)
--     for i=1,#t2 do
--         t1[#t1+1] = t2[i]
--     end
--     return t1
-- end

----------------------------------------------------------------------------
-- if opt.how_many=='all' then
--     opt.how_many=data:size()
-- end
-- opt.how_many=math.min(opt.how_many, data:size())

-- local filepaths = {} -- paths to images tested on
-- -- require 'qtwidget'
-- -- local w = qtwidget.newwindow(640, 640)

---- add videWwriter
local cv = require 'cv'
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video input/output

-- local filepath = "./videos/2（雫なし）.MP4"
local filepath = "./videos/3_taken_noise.MP4"

local cap = cv.VideoCapture{filepath or 0}
-- local cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
    print("Failed to open the the file from "..filepath)
    os.exit(-1)
end
-- videoWriter
local sz = opt.fineSize  
local frameToSaveSize = {sz, sz}

local videoWriter = cv.VideoWriter{
    "sampleOutputForCool.Avi",
    cv.VideoWriter.fourcc{'D', 'I', 'V', 'X'}, -- or any other codec
    fps = 25,
    frameSize = frameToSaveSize
 }

if not videoWriter:isOpened() then
    print('Failed to initialize video writer. Possibly wrong codec or file name/path trouble')
    os.exit(-1)
end
-----
-- Create a new window
-- cv.namedWindow{"edges", cv.WINDOW_AUTOSIZE}
-- Read the first frame
local _, frame = cap:read{}

function torch_to_opencv(img)
    local img_opencv
    if img:nDimension() == 2 then
       img_opencv = img
    else
       img_opencv = img:permute(2,3,1):contiguous()
    end
    img_opencv:mul(255)
    -- uncomment this if the channel order matters for you,
    -- for example if you're going to use imwrite on the result:
    cv.cvtColor{img_opencv, img_opencv, cv.COLOR_BGR2RGB}
 
    return img_opencv
 end

 -- Init some

timer = torch.Timer()



while true do

    -- -- Resize it to 256 x 256
    -- local im = cv.resize{frame, {sz,sz}}:cuda()

    -- print(torch.type(im))
    -- print("resize")
    -- print(timer:time())
    -- timer:reset()

    -- --im = im:float()
    -- im.type(torch.FloatTensor)

    -- print(torch.type(im))
    -- print("cast to float")
    -- print(timer:time())
    -- timer:reset()

    -- im = im:div(255):mul(2):add(-1) -- 640x640x3
    
    -- print("1")
    -- print(timer:time())
    -- timer:reset()
  
    -- -- Resize again to CNN input size and swap dimensions
    -- -- to CxHxW from HxWxC
    -- -- Note that BGR channel order required by ImageNet is already OpenCV's default
    
    -- local input = im:transpose(1,3):transpose(2,3):clone() -- 3x640x640
    -- print("22")
    -- print(timer:time())
    -- timer:reset()
    
    -- -- TODO: LAM SAO CHO INPUT VANG CHACH TRUOC
    -- -- => customize data:getbatch()
    -- -- image.save('/home/ubuntu/trinh/Edited_Original_IDCGAN/ID-CGAN/IDCGAN/output/tao_lao.jpg',input)
    -- -- os.exit()
    -- if opt.gpu > 0 then
    --     input = input:cuda()
    -- end   
    
    
    -- input = torch.reshape(input, torch.LongStorage{1, 3, sz,sz})

    -- print("333")
    -- print(timer:time())
    -- timer:reset()
    
    -- local temp = netG:forward(input)
    -- print("**********")
    -- print(timer:time())
    -- timer:reset()

    -- START CODE HERE
    local resized_frame = cv.resize{frame, {sz,sz}}
    
    local temp = resized_frame:cuda():div(255):mul(2):add(-1):transpose(1,3):transpose(2,3)
    temp = netG:forward(torch.reshape(temp, torch.LongStorage{1, 3, sz,sz}))
    -- END CODE HERE
    local output = util.deprocess_batch(temp)
    print("4444")
    print(timer:time())
    timer:reset()
    
    --local output =netG:forward(input)
    -- local name_image_saved = '/home/ubuntu/trinh/Edited_Original_IDCGAN/ID-CGAN/IDCGAN/output/somepic_output.jpg'
    -- image.save(name_image_saved, output)
    print('======================================================')
    print('from after deprocess ')
    print(torch.type(output))
    
    
    output = output:float()
    -- output.type(torch.FloatTensor)
    -- nn.utils.recursiveType(toutputab, 'torch.FloatTensor')
    print("55555")
    print(timer:time())
    timer:reset()
    print('======================================================')
    print('after :float ')
    print(torch.type(output))
    
    -- paths.mkdir(paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase))
    -- local image_dir = paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase, 'outout')
    -- paths.mkdir(image_dir)
    -- paths.mkdir(paths.concat(image_dir,'input'))
    -- paths.mkdir(paths.concat(image_dir,'output'))
    -- paths.mkdir(paths.concat(image_dir,'target'))

    -- local name = paths.concat(image_dir,'output','raindrop.jpg')

    

    -- local img_show = torch.reshape(output, torch.LongStorage{3, sz, sz})
    -- image.save(name, img_show)
    -- os.exit()

    local img_show = torch.reshape(output, torch.LongStorage{3, sz, sz})
    img_show = img_show:transpose(3,2):transpose(3,1):clone()
    -- img_show = cv.resize{img_show, {256,256}}
    print("666666")
    print(timer:time())
    timer:reset()
    

    -- cv.cvtColor{img_show, img_show, cv.COLOR_BGR2RGB}
    -- img_show = torch_to_opencv(img_show)

    
    -- require 'qtwidget'
    -- local w = qtwidget.newwindow(640, 640)
    -- image.display{image = image.scale(output[i],output[i]:size(2),output[i]:size(3)/opt.aspect_ratio), win = w}    

    cv.imshow{"original video", resized_frame}    
    cv.imshow{"img_show", img_show}
    -- cv.waitKey{0}
    print("7777777")
    print(timer:time())
    timer:reset()

    -- finally, tell videoWriter to push frameToSave into the video
     --videoWriter:write{img_show}

    if cv.waitKey{30} >= 0 then break end

    -- Grab the next frame
    cap:read{frame}
end

--------- For loop -----------------

