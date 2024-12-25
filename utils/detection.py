from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(image):
    results = detector.detect_faces(image)
    faces = []
    boxes = []
    for result in results:
        bounding_box = result['box']
        x, y, w, h = bounding_box
        face = image[y:y+h, x:x+w]
        faces.append(face)
        boxes.append((x, y, x+w, y+h))
    return faces, boxes