import cv2
import os


class ViolaDetector:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    line_width = 3
    face_color = (0, 0, 255)
    eyes_color = (0, 255, 0)
    scale_factor = 1.3
    min_neighbors = 5

    def matcher(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, self.scale_factor, self.min_neighbors)
        for x, y, w, h in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), self.face_color, self.line_width)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), self.eyes_color, self.line_width)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    file_directory = 'images'
    images_count = 8
    images = [os.path.join(file_directory, f'image{i}.JPG') for i in range(1, images_count + 1)]

    matching_instance = ViolaDetector()

    for image in images:
        print(f'detecting {image}')
        matching_instance.matcher(image)
