import argparse
import json
from glob import glob

from neural_feature_detector import NeuralFeatureDetector
from old_school_feature_detector import OldSchoolFeatureDetector


def do_it(faces_glob, predictor_file, output_filename, strategy):
    face_dataset = []

    if strategy == 'neural':
        detector = NeuralFeatureDetector()
    else:
        detector = OldSchoolFeatureDetector(predictor_file)

    for filename in glob(faces_glob):
        print("processing " + filename)
        for face_data in detector.process_file(filename):
            face_dataset.append(face_data)

    with open(output_filename, 'w') as outfile:
        json.dump(face_dataset, outfile, indent=2, sort_keys=False)

    for row in face_dataset:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face clustering data collector')
    parser.add_argument('--faces', help='Glob pattern for the face images', required=True)
    parser.add_argument('--predictor', help='Predictor .dat file to use', required=True)
    parser.add_argument('--output', help='Output file for JSON data', default='face_data.json')
    parser.add_argument('--strategy', help='[oldschool | neural]', default='neural')
    args = parser.parse_args()

    do_it(args.faces, args.predictor, args.output, args.strategy)
