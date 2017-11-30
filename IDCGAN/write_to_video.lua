-- a translated demo from here:
-- http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html

local cv = require 'cv'
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video stream, VideoWriter
require 'cv.imgproc' -- Image processing (resize, crop, draw text, ...)

local filepath = "./videos/2（雫なし）.MP4"

local cap = cv.VideoCapture{filepath}
if not cap:isOpened() then
    print("Failed to open the default camera")
    os.exit(-1)
end
-- videoWiriter
local sz = 720  
local frameToSaveSize = {sz, sz}

local videoWriter = cv.VideoWriter{
    "sampleOutput.Avi",
    cv.VideoWriter.fourcc{'D', 'I', 'V', 'X'}, -- or any other codec
    fps = 25,
    frameSize = frameToSaveSize
 }

if not videoWriter:isOpened() then
    print('Failed to initialize video writer. Possibly wrong codec or file name/path trouble')
    os.exit(-1)
end


-- Create a new window
-- cv.namedWindow{"edges", cv.WINDOW_AUTOSIZE}
-- Read the first frame
local _, frame = cap:read{}

-- make a tensor of same type, but a 2-dimensional one
-- local edges = frame.new(frame:size()[1], frame:size()[2])

while true do
    local w = frame:size(2)
    local h = frame:size(1)
    -- Get central square crop
    local crop = cv.getRectSubPix{frame, patchSize={h,h}, center={w/2, h/2}}

    -- cv.cvtColor{frame, edges, cv.COLOR_BGR2GRAY}

    -- cv.GaussianBlur{
    --     edges,
    --     ksize = {7,7},
    --     sigmaX = 1.5,
    --     sigmaY = 1.5,
    --     dst = edges
    -- }

    -- cv.Canny{
    --     image = edges,
    --     threshold1 = 0,
    --     threshold2 = 30,
    --     apertureSize = 3,
    --     edges = edges
    -- }
       -- get next image; for example, read it from camera
    -- local frame = cv.resize{frame, {sz, sz}}

    local converted_frame = cv.resize{frame, {sz, sz}}
    for i=1,5 do
        cv.putText{
            converted_frame, 
            "Trinh "..i, 
            {10, 10 + i * 25},
            fontFace=cv.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color={200, 200, 50},
            thickness=2
        }
    end

    -- the next frame in the resulting video
    -- local frameToSave = torch.Tensor(frameToSaveSize[2], frameToSaveSize[1], 3)

    -- first, copy the original frame into the left half of frameToSave:
    -- frameToSave:narrow(2, 1, sz):copy(frame)

    -- second, copy the processed (for example, rendered in painter style)
    -- frame into the other half:
    -- frameToSave:narrow(2, sz+1, sz):copy(converted_frame)
    
    -- -- finally, tell videoWriter to push frameToSave into the video
    -- videoWriter:write{frameToSave}
    --    cv.imshow{"edges", edges}


    -- cv.imshow{"frame",frame }
    cv.imshow{"converted frame",converted_frame }

    
    -- cv.imshow{"frameToSave",frameToSave }
    -- finally, tell videoWriter to push frameToSave into the video
    videoWriter:write{converted_frame}

    if cv.waitKey{1} >= 0 then break end

    -- Grab the next frame
    cap:read{frame}
end

