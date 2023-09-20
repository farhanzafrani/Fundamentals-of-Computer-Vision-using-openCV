import cv2 as cv

# rescaling function
def rescale_image(frame,scale = 0.1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


# video reading
path = "image_processing_files\VID_20230830_131627.mp4"
capture = cv.VideoCapture(path)

while True:
    isTrue , frame = capture.read()
    rescale_frame = rescale_image(frame)
    cv.imshow('video',rescale_frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()