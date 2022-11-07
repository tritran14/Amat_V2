class Poi:
    def __init__(self,x,y):
        self.x = int(x)
        self.y = int(y)
    def __iter__(self):
        yield 'x', int(self.x)
        yield 'y', int(self.y)

class DetectionArea:
    def __init__(self,left_top, right_bottom):
        self.left_top = left_top
        self.right_bottom = right_bottom
    def __iter__(self):
        yield 'left_top', dict(self.left_top)
        yield 'right_bottom', dict(self.right_bottom)

class FaceDetection:
    
    def set_mouth_area(self, mouth_area_list):
        self.mouth_area_list = mouth_area_list
    def has_mouth(self):
        return len(self.mouth_area_list) > 0
    
    def set_nose_area(self, nose_area_list):
        self.nose_area_list = nose_area_list
    def has_nose(self):
        return len(self.nose_area_list) > 0
    
    def set_eyes_area(self, eyes_area_list):
        self.eyes_area_list = eyes_area_list
    def has_eyes(self):
        return len(self.eyes_area_list) >= 2
    
    def valid_face(self):
        return self.has_eyes() and self.has_mouth() and self.has_nose()
    
    def __iter__(self):
        yield 'mouth_area_list', [dict(x) for x in self.mouth_area_list]
        yield 'nose_area_list', [dict(x) for x in self.nose_area_list]
        yield 'eyes_area_list', [dict(x) for x in self.eyes_area_list]
        yield 'valid_face', self.valid_face()

class FaceItem:
    glasses_label = ['Glasses', 'No glasses']
    mask_label = ['Mask', 'No mask']

    def set_mask(self, value):
        self.mask = value
    def set_glasses(self, value):
        self.glasses = value
    
    def has_mask(self):
        return self.mask
    def has_glasses(self):
        return self.glasses
    
    def has_glasses_str(self):
        idx = 0 if self.has_glasses() else 1
        return self.glasses_label[idx]
    def has_mask_str(self):
        idx = 0 if self.has_mask() else 1
        return self.mask_label[idx]

    def is_clean_face(self):
        return not self.has_mask() and not self.has_glasses()
    
    def __iter__(self):
        yield 'mask', bool(self.mask)
        yield 'glasses', bool(self.glasses)
        yield 'is_clean_face', self.is_clean_face()
    
class FaceStatus:
    def __init__(self, face_detection, face_item):
        self.face_detection = face_detection
        self.face_item = face_item

    def set_brightness_status(self, brightness_status_message):
        self.brightness_status_message = brightness_status_message
    
    def good_face(self):
        return self.face_detection.valid_face() and self.face_item.is_clean_face()
    
    def __iter__(self):
        yield 'face_detection', dict(self.face_detection)
        yield 'face_item', dict(self.face_item)
        yield 'brightness_status_message', self.brightness_status_message
        yield 'is_good_face', self.good_face()

class FaceRecognize:
    def __init__(self, id, name, accuracy, face_area):
        self.id = id
        self.name = name
        self.accuracy = accuracy
        self.face_area = face_area
    def __iter__(self):
        yield 'id', self.id
        yield 'name', self.name
        yield 'accuracy', self.accuracy
        yield 'face_area', dict(self.face_area)