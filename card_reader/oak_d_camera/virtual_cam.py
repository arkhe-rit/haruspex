import pyvirtualcam
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setPreviewSize(1920,1080)
# cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
# cam.setPreviewSize(3840,2160)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")

cam.preview.link(xout.input)

# Connect to device and start pipeline
with (dai.Device(pipeline) as device, 
      pyvirtualcam.Camera(width=1920, height=1080, fps=20) as uvc):
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    print("UVC running")
    while True:
        frame = qRgb.get().getFrame()
        uvc.send(frame)