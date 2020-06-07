import face_recognition
import picamera
import pickle
import numpy as np
import playsound
from google_speech import Speech

# Get a reference to the Raspberry Pi camera.
camera = picamera.PiCamera()
camera.resolution = (320, 240)
output = np.empty((240, 320, 3), dtype=np.uint8)




#load known faces
print("Loading known face image(s)")
# Load face encodings
with open('dataset_faces.dat', 'rb') as f:
    all_face_encodings = pickle.load(f)

# Grab the list of names and the list of encodings
known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))





# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

while True:
    print("Capturing image.")
    # Grab a single frame of video from the RPi camera as a numpy array
    camera.capture(output, format="rgb")

    # Loop over each face found in the frame to see if it's someone we know.
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(output)
        print("Found {} faces in image.".format(len(face_locations)))

        face_encodings = face_recognition.face_encodings(output, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
            face_names.append(name)

    print(*face_name, sep = ", ")

    #play names of detected people
    lang = "hi"
    sox_effects = ("speed", "1.2")
    for name in face_names:
        speech = Speech(name, lang)
        speech.play(sox_effects)

    #toggle process_this_frame var to run FR on alternate frames
    process_this_frame = not process_this_frame

    #print("I see someone named {}!".format(name))
