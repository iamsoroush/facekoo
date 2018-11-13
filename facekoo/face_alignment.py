import numpy as np
import cv2


class FaceAligner:
    def __init__(self, predictor, landmarks_idxs, desired_left_eye=(0.35, 0.35),
                 desired_face_width=182, desired_face_height=None):
        """Init a FaceAligner instance.

        predictor: A facial landmark predictor model.

        landmarks_idxs: Positions for output landmarks.

        desired_left_eye: Desired output left eye position, given in a (x, y) tuple.
            Percentages between (0.2, 0.4), With 20% you’ll basically be getting a
            “zoomed in” view of the face, whereas with larger values the face will
            appear more “zoomed out.”

        desired_face_width: Output images width, in pixels.

        desired_face_height: Output images height, in pixels.
        """

        self.predictor = predictor
        self.landmarks_idxs = landmarks_idxs
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        if desired_face_height is None:
            self.desired_face_height = desired_face_width
        else:
            self.desired_face_height = desired_face_height

    def align(self, image, gray, rect):
        """Aligns the faces found in img.

        img: The RGB input image.
        gray: The gray scale input image.
        rect: The bounding box rectangle produced by dlib’s HOG face detector.
        """

        FACIAL_LANDMARKS_IDXS = self.landmarks_idxs
        # apply dlib’s facial landmark predictor
        shape = self.predictor(gray, rect)

        # convert the landmarks into (x, y)-coordinates in NumPy format.
        shape = self.shape_to_np(shape)

        # read the left_eye  and right_eye  regions from the FACIAL_LANDMARK_IDXS  dictionary
        (left_eye_start, left_eye_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_eye_start, right_eye_end) = FACIAL_LANDMARKS_IDXS["right_eye"]

        # get coordinates of left eye and right eye
        left_eye_pts = shape[left_eye_start: left_eye_end + 1]
        right_eye_pts = shape[right_eye_start: right_eye_end + 1]

        # compute the center of mass for each eye
        left_eye_center = left_eye_pts.mean(axis=0).astype("int")
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]

        # determine the scale of the new resulting image by taking the ratio of the distance between eyes
        # in the *current* image to the ratio of distance between eyes in the *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= self.desired_face_width
        scale = desired_dist / dist

        # compute center (x, y)-coordinates (i.e., the median point) between the two eyes in the input image
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        # apply the affine transformation
        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

    @staticmethod
    def shape_to_np(shape, dtype="int"):
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)  # Initialize the list of (x, y)-coordinates

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords
