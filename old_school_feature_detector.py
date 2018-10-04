import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance


class OldSchoolFeatureDetector:
    NOSE_TOP_IDX = 0
    NOSE_LEFT_IDX = 4
    NOSE_RIGHT_IDX = 8
    NOSE_BOTTOM_IDX = 6

    def __init__(self, predictor_file):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_file)

    # Not great as the face is the bottom half of the head only - need to normalise by head size, not feature extents
    def normalise_points(self, shape):
        (max_x, max_y) = np.amax(shape, axis=0)
        (min_x, min_y) = np.amin(shape, axis=0)
        width = max_x - min_x * 1.0
        height = max_y - min_y * 1.0

        return np.array([((x - min_x) / width, (y - min_y) / height) for (x, y) in shape])

    def poly_area(self, corners):
        n = len(corners)  # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2.0
        return area

    def measure(self, filename, shape):
        # Get the jaw features
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        jaw_points = shape[i:j]
        jaw_left = jaw_points[0]
        jaw_right = jaw_points[-1]
        jaw_width = distance.euclidean(jaw_left, jaw_right)
        jaw_middle = jaw_points[int(len(jaw_points) / 2)]
        jaw_top = jaw_left + jaw_right * (0.5, 0.5)
        jaw_height = distance.euclidean(jaw_top, jaw_middle)
        jaw_ratio = jaw_height / jaw_width
        jaw_area_size = self.poly_area(jaw_points)

        # Get the eye features
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        eye_points = shape[i:j]
        left_eye_location = np.mean(eye_points, axis=0)
        left_eye_size = self.poly_area(eye_points) / jaw_area_size
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        eye_points = shape[i:j]
        right_eye_location = np.mean(eye_points, axis=0)
        right_eye_size = self.poly_area(eye_points) / jaw_area_size
        eye_size_differential = np.abs(left_eye_size - right_eye_size)
        eye_size = (left_eye_size + right_eye_size) / 2
        eye_distance = distance.euclidean(left_eye_location, right_eye_location) / jaw_width

        # Get the eyebrow features
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        eyebrow_points = shape[i:j]
        left_eyebrow_location = np.mean(eyebrow_points, axis=0)
        left_eyebrow_width = distance.euclidean(eyebrow_points[0], eyebrow_points[-1])
        left_eyebrow_lift = distance.euclidean(left_eyebrow_location, left_eye_location)
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        eyebrow_points = shape[i:j]
        right_eyebrow_location = np.mean(eyebrow_points, axis=0)
        right_eyebrow_width = distance.euclidean(eyebrow_points[0], eyebrow_points[-1])
        right_eyebrow_lift = distance.euclidean(right_eyebrow_location, right_eye_location)

        eyebrow_width = (left_eyebrow_width + right_eyebrow_width) / (2 * jaw_width)
        eyebrow_lift = (left_eyebrow_lift + right_eyebrow_lift) / (2 * jaw_width)

        # Get the nose features
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        nose_points = shape[i:j]
        nose_top = nose_points[self.NOSE_TOP_IDX]
        nose_left = nose_points[self.NOSE_LEFT_IDX]
        nose_right = nose_points[self.NOSE_RIGHT_IDX]
        nose_bottom = nose_points[self.NOSE_BOTTOM_IDX]
        nose_width = distance.euclidean(nose_left, nose_right)
        nose_height = distance.euclidean(nose_top, nose_bottom)
        nose_ratio = nose_height / nose_width
        nose_size = nose_height / jaw_width

        # Return the feature data
        face_data = {
            "filename": "faces/" + filename.split('/')[-1],
            "features": [
                jaw_ratio,
                eye_distance,
                nose_ratio,
                nose_size,
                eye_size,
                eye_size_differential,
                eyebrow_width,
                eyebrow_lift
            ]
        }
        yield face_data

    def process_file(self, filename):
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(filename)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = self.detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = face_utils.shape_to_np(self.predictor(gray, rect))

            yield from self.measure(filename, shape)