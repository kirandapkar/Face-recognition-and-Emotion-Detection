# Face-recognition-and-Emotion-Detection
├── Face-recognition-and-Emotion-Detection
  ├── dhoni.jpeg
  ├── emotion.jpeg
  ├── emotionRecognition.py
  ├── faceRecognition.py
  ├── haarcascade_files
  │   ├── haarcascade_eye.xml
  │   └── haarcascade_frontalface_default.xml
  ├── kiran.jpg
  ├── load_and_process.py
  ├── models
  │   ├── cnn.py
  │   └── _mini_XCEPTION.102-0.66.hdf5
  ├── modi.jpg
  ├── README.txt
  ├── Report.pdf
  ├── test.py
  ├── train_emotion_classifier.py
  ├── trainingImages2
  │   ├── 0 (contains images)
  │   └── 1 (contains images)
  ├── trainingImages3
  │   ├── 0 (contains images)
  │   ├── 1 (contains images)
  │   └── 2 (contains images)
  └── trainingImages4
      ├── 0 (contains images)
      ├── 1 (contains images)
      ├── 2 (contains images)
      └── 3 (contains images)
14 directories, 779 files

HOW TO RUN CODE:-
**********NOTE: <input_images> should contain only 1 face ***********
    For face recognition:
NOTE: Please refer to lines 16 and 36 in 'test.py' before running the code
'python3 test.py <input_image> <2/3/4>'
A file named 'trainingDatax.xml' (where x = 2/3/4) which can be used for subsequent runs.
OUTPUT will be an image with rectangle drawn around face and on top it, its label if labelled with confidence <=120
    For emotion recognition:
NOTE: Training has already been done. Just run the code in 'emotionRecognition.py' with an input
'python3 emotionRecognition.py <input_image>'
OUTPUT will be 2 windown one with rectangle drawn around face in input image, other contains probabilites of different emotions
