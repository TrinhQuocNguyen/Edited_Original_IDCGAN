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

local params, gparams= netG:getParameters()
print('======================================================')
print('params: ')
print(params)
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

local filepath = "./videos/cat.MP4"
-- local filepath = "./videos/3_taken_noise.MP4"

local filepathvideos = {
    video_1 = "./videos/inpainting_rain.MP4",
    video_2 = "./videos/3_taken_noise_15f.MP4",
    video_3 = "./videos/2（雫なし）.MP4",
    video_4 = "./videos/3_taken_noise.MP4"
}

cv.namedWindow{"img_show_1", cv.WINDOW_AUTOSIZE}
cv.namedWindow{"img_show_2", cv.WINDOW_AUTOSIZE}
cv.namedWindow{"img_show_3", cv.WINDOW_AUTOSIZE}
cv.namedWindow{"img_show_4", cv.WINDOW_AUTOSIZE}

cv.moveWindow{"img_show_2", 0, 512}
cv.moveWindow{"img_show_3", 512, 0}
cv.moveWindow{"img_show_4", 512, 512}

local cap = cv.VideoCapture{filepath or 0}
local capvideos = {
    cap_1 = cv.VideoCapture{filepathvideos['video_1']},
    cap_2 = cv.VideoCapture{filepathvideos['video_2']},
    cap_3 = cv.VideoCapture{filepathvideos['video_3']},
    cap_4 = cv.VideoCapture{filepathvideos['video_4']}
}
-- local cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
    print("Failed to open the the file from "..filepath)
    os.exit(-1)
end
-- videoWriter
local sz = opt.fineSize  
local frameToSaveSize = {sz, sz}

-- local videoWriter = cv.VideoWriter{
--     "sampleOutputForCool.Avi",
--     cv.VideoWriter.fourcc{'D', 'I', 'V', 'X'}, -- or any other codec
--     fps = 25,
--     frameSize = frameToSaveSize
--  }

-- if not videoWriter:isOpened() then
--     print('Failed to initialize video writer. Possibly wrong codec or file name/path trouble')
--     os.exit(-1)
-- end
-----
-- Create a new window
-- cv.namedWindow{"edges", cv.WINDOW_AUTOSIZE}
-- Read the first frame
-- local _, frame = cap:read{}
local _1, frame_1 = capvideos['cap_1']:read{}
local _2, frame_2 = capvideos['cap_2']:read{}
local _3, frame_3 = capvideos['cap_3']:read{}
local _4, frame_4 = capvideos['cap_4']:read{}


timer = torch.Timer()

while true do
    timer:reset()
    
    -- START CODE HERE
    local temp_1 = cv.resize{frame_1, {sz,sz}}:cuda():div(255):mul(2):add(-1):transpose(1,3):transpose(2,3)
    temp_1 = netG:forward(torch.reshape(temp_1, torch.LongStorage{1, 3, sz,sz}))
    local output_1 = util.deprocess_batch(temp_1):float()
    local img_show_1 = torch.reshape(output_1, torch.LongStorage{3, sz, sz}):transpose(3,2):transpose(3,1):clone()
    
    
    local temp_2 = cv.resize{frame_2, {sz,sz}}:cuda():div(255):mul(2):add(-1):transpose(1,3):transpose(2,3)
    temp_2 = netG:forward(torch.reshape(temp_2, torch.LongStorage{1, 3, sz,sz}))
    local output_2 = util.deprocess_batch(temp_2):float()    
    local img_show_2 = torch.reshape(output_2, torch.LongStorage{3, sz, sz}):transpose(3,2):transpose(3,1):clone()
    
    local temp_3 = cv.resize{frame_3, {sz,sz}}:cuda():div(255):mul(2):add(-1):transpose(1,3):transpose(2,3)
    temp_3 = netG:forward(torch.reshape(temp_3, torch.LongStorage{1, 3, sz,sz}))
    local output_3 = util.deprocess_batch(temp_3):float()    
    local img_show_3 = torch.reshape(output_3, torch.LongStorage{3, sz, sz}):transpose(3,2):transpose(3,1):clone()
    
    local temp_4 = cv.resize{frame_4, {sz,sz}}:cuda():div(255):mul(2):add(-1):transpose(1,3):transpose(2,3)
    temp_4 = netG:forward(torch.reshape(temp_4, torch.LongStorage{1, 3, sz,sz}))
    local output_4 = util.deprocess_batch(temp_4):float()
    local img_show_4 = torch.reshape(output_4, torch.LongStorage{3, sz, sz}):transpose(3,2):transpose(3,1):clone()
    
    -- -- END CODE HERE
    -- print("4444")
    -- print(timer:time())
    -- timer:reset()
    -- print('======================================================')
    -- print('from after deprocess ')
    -- print("55555")
    -- print(timer:time())
    print('======================================================')
    cv.imshow{"img_show_1", img_show_1}
    cv.imshow{"img_show_2", img_show_2}
    cv.imshow{"img_show_3", img_show_3}
    cv.imshow{"img_show_4", img_show_4}

    if cv.waitKey{30} >= 0 then break end
    capvideos['cap_1']:read{frame_1}
    capvideos['cap_2']:read{frame_2}
    capvideos['cap_3']:read{frame_3}
    capvideos['cap_4']:read{frame_4}

    print(timer:time())
    timer:reset()
end


--------- For loop -----------------

