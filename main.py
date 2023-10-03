import cv2

# Initialize face and smile cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_roi = gray[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)

        # Initialize variables to keep track of the smile aspect ratio
        max_area = 0
        smile_ar = 0

        for (sx, sy, sw, sh) in smiles:
            if sw * sh > max_area:
                max_area = sw * sh
                smile_ar = float(sw) / float(sh)

        # Classify the expression based on the aspect ratio
        if smile_ar > 0:
            if smile_ar >= 2.0:
                cv2.putText(frame, 'Smiling!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                print("Smile aspect ratio is", smile_ar)
            else:
                cv2.putText(frame, 'Neutral.', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print("Smile aspect ratio is", smile_ar)
        else:
            cv2.putText(frame, 'Frowning.', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print("Smile aspect ratio is", smile_ar)

    cv2.imshow('Real-time Face Expression Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
