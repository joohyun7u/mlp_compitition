import cv2

class NoiseReconstruct(object):
    def __init__(self):
        "It recieves output of ToTensor()"
        return

    def __call__(self,sample):
        image = sample
        image = (image + 0.5) * 2
        return image
    
class RGBtoYcrcb(object):
    def __init__(self, ):
        return 

    def __call__(self,sample):
        image = sample
        image = cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)
        return image