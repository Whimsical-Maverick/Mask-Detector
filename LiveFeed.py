import tensorflow as tf
import cv2
labels_dict = {1: 'Danger', 0: 'Safe'}
model=tf.keras.models.load_model(r"D:\MaskDetection.h5")
faces=cv2.CascadeClassifier(r"D:\haarcascade_frontalface_default.xml")
video =cv2.VideoCapture(0)
while True:
  ret,frame=video.read()
  facedetect=faces.detectMultiScale(frame,1.3,3)
  for x,y,w,h in facedetect:
    face_img=frame[y:y+h,x:x+w]
    face_img=cv2.resize(face_img,(224,224))
    reshaped_face_img=tf.reshape(face_img,[1,224,224,3])
    result=model.predict(reshaped_face_img)
    label=result.argmax()
    if label==0:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
      cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
      cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 0, 255), -1)
      cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

  cv2.imshow('Video',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
video.release()
cv2.destroyAllWindows()