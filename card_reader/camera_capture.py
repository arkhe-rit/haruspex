def video_capture_generator(video_capture):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        yield frame

def oak_capture_generator():
    import depthai as dai

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    xoutPreview = pipeline.create(dai.node.XLinkOut)

    xoutVideo.setStreamName("video")
    xoutPreview.setStreamName("preview")

    # Properties
    camRgb.setPreviewSize(3840, 2160)

    camRgb.setFps(12)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    camRgb.setInterleaved(True)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.initialControl.setSharpness(0)     # range: 0..4, default: 1
    camRgb.initialControl.setLumaDenoise(0)   # range: 0..4, default: 1
    camRgb.initialControl.setChromaDenoise(4) # range: 0..4, default: 1

    # Linking
    camRgb.video.link(xoutVideo.input)
    camRgb.preview.link(xoutPreview.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        video = device.getOutputQueue('video')
        preview = device.getOutputQueue('preview')

        while True:
            videoFrame = video.get()
            previewFrame = preview.get()

            # Get BGR frame from NV12 encoded video frame to show with opencv
            # yield videoFrame.getCvFrame()
            # Show 'preview' frame as is (already in correct format, no copy is made)
            yield previewFrame.getFrame()