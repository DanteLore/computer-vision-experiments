import face_recognition


class NeuralFeatureDetector:
    def process_file(self, filename):
        image = face_recognition.load_image_file(filename)
        encodings = face_recognition.face_encodings(image)

        print("Generated {0} encodings".format(len(encodings)))

        for encoding in encodings:
            # Return the feature data
            face_data = {
                "filename": "faces/" + filename.split('/')[-1],
                "features": encoding.tolist()
            }
            yield face_data
