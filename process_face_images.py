import argparse
import os
import uuid
from glob import glob

import cv2
import dlib
import imutils
import openface


def clear_dir():
    for f in glob('faces/*'):
        os.remove(f)


def find_faces(source_dir, predictor_file, align_faces=False):
    detector = dlib.get_frontal_face_detector()
    face_aligner = openface.AlignDlib(predictor_file)

    clear_dir()

    for filename in glob(source_dir):
        print("Processing file {0}".format(filename))

        image = cv2.imread(filename)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            print("Processing face {0}".format(i + 1))

            if align_faces:
                # Use openface to calculate and perform the face alignment
                aligned_face = face_aligner.align(534, image, rect,
                                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            else:
                aligned_face = image[rect.top():rect.bottom(), rect.left():rect.right()]

            try:
                resized_face = imutils.resize(aligned_face, width=250, inter=cv2.INTER_CUBIC)
                cv2.imwrite("faces/face-{0}.jpg".format(uuid.uuid4().hex), resized_face)
            except:
                print("Error processing file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face clustering data collector')
    parser.add_argument('--faces', help='Glob pattern for the face images', required=True)
    parser.add_argument('--predictor', help='Predictor .dat file to use', required=True)
    parser.add_argument('--align-faces', help='Align faces while maintaining ratios', action='store_true')
    parser.set_defaults(align_faces=False)
    args = parser.parse_args()

    find_faces(args.faces, args.predictor, args.align_faces)
