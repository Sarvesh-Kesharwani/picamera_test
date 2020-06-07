"""
Usage:
face_recognize.py -d <train_dir>

Options:
-h, --help					 Show this help
-d, --train_dir =<train_dir> Directory with images for training
"""
import face_recognition
import docopt
import os
import pickle


def TakeSamples(dir):

    all_face_encodings = {}

    if dir[-1] != '/':
        dir += '/'
    train_dir = os.listdir(dir)

    face_index = 0
    for person in train_dir:
        pix = os.listdir(dir + person)
        face_index += 1
        for person_img in pix:  # runs only one time if there is only one image in one person folder

            face = face_recognition.load_image_file(dir + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)
            no_of_faces = len(face_bounding_boxes)

            if no_of_faces == 1:
                all_face_encodings[person] = face_recognition.face_encodings(face)[0]
            else:
                print(person + "_img contains multiple faces!")

    with open('dataset_faces.dat', 'wb') as f:
        pickle.dump(all_face_encodings, f)


def main():
    args = docopt.docopt(__doc__)
    train_dir = args["--train_dir"]
    TakeSamples(train_dir)


if __name__ == "__main__":
    main()
