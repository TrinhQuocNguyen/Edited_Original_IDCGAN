-- a translated demo from here:
-- http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html

local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'math'

local filepath = "./videos/2（雫なし）.MP4"

local cap = cv.VideoCapture{filepath}
if not cap:isOpened() then
    print("Failed to open the default camera")
    os.exit(-1)
end

--cv.namedWindow{"edges", cv.WINDOW_AUTOSIZE}
local _, frame = cap:read{}
frame = cv.resize{frame, {640,640}}
-- make a tensor of same type, but a 2-dimensional one
--local edges = frame.new(frame:size()[1], frame:size()[2])

local blurred = frame:clone()
local lowContrastMask = frame:clone()
local shaprened = frame:clone()
while true do
    
    --cv.cvtColor{frame, edges, cv.COLOR_BGR2GRAY}

    cv.GaussianBlur{
        src = frame,
        ksize = {77,77},
        sigmaX = 12.5,
        sigmaY = 12.5,
        dst = blurred
    }

    -- cv.Canny{
    --     image = edges,
    --     threshold1 = 0,
    --     threshold2 = 30,
    --     apertureSize = 3,
    --     edges = edges
    -- }
    cv.imshow{"blurred", blurred}
    
    cv.imshow{"frame",frame }
    -- local blurred 
    -- cv.GaussianBlur{edges, ksize = {1,1}, 1, 1, dst = blurred}

    -- cv.imshow{"blurred", blurred}
    --lowContrastMask = math.abs(frame-blurred) < 5

    print (math.abs(1-2))

    -- lowContrastMask = math.abs(frame - blurred) < 5

    -- for i = 1, frame:size()[1] do
    --     for j = 1, frame:size()[2] do
    --         for c = 1, frame:size()[3] do
    --             --print(frame[i][j][c])
    --             if math.abs(frame[i][j][c] - blurred[i][j][c]) < 5 then
    --                 lowContrastMask[i][j][c] = 1
    --             else
    --                 lowContrastMask[i][j][c] = 0
    --             end
    --         end
    --     end
    -- end


    print(frame[640][640][3])
    
    print(frame:size()[1])
    print(frame:size()[2])
    print(frame:size()[3])

    for i = 1, frame:size()[1] do
        for j = 1, frame:size()[2] do
            for c = 1, frame:size()[3] do
                --print(frame[i][j][c])
                if (-5 < (frame[i][j][c] - blurred[i][j][c])) and ((frame[i][j][c] - blurred[i][j][c]) < 5) then
                    --lowContrastMask[i][j][c] = 1
                    shaprened[i][j][c]  = 2*frame[i][j][c] - blurred[i][j][c]
                    if shaprened[i][j][c]  < 0 then 
                        shaprened[i][j][c]  = 0
                    end
                    if shaprened[i][j][c]  > 255 then 
                        shaprened[i][j][c]  = 255
                    end
                -- else
                --     lowContrastMask[i][j][c] = 0
                end
            end
        end
    end

    --shaprened = frame*(1 + 1) + blurred* (-1)
    --shaprened = shaprened * lowContrastMask


    cv.imshow{"shaprened", shaprened}
    
    cv.waitKey{0}
    if cv.waitKey{1} >= 0 then break end

    --cap:read{frame}
end

