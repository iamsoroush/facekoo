from os.path import normpath, abspath, dirname, realpath, join


class FaceKooConfig:

    ROCW_K = 50
    ROCW_METRIC = 'minkowski'
    ROCW_N_ITERATION = 2
    ROCW_ALGORITHM = 'auto'

    # Higher 'min_samples' or lower 'eps' indicate higher density necessary to form a cluster.
    DBSCAN_EPS = 0.4
    DBSCAN_MIN_SAMPLES = 2
    DBSCAN_METRIC = 'cosine'


    PROJECT_ROOT = normpath(abspath(dirname(dirname(realpath(__file__)))))
    MODELS_DIR = join(PROJECT_ROOT, 'models')

    # Download from here: https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view
    FACENET_MODEL_PATH = join(MODELS_DIR, '20180402-114759')
    FACENET_BATCH_SIZE = 32
    FACENET_IMAGE_SIZE = 160
    FACENET_DO_PREWHITEN = True
    FACENET_DO_CROP = True

    # Download dlib's face detector model and unzip it: http://dlib.net/files/mmod_human_face_detector.dat.bz2
    DLIB_FACE_DETECTOR_PATH = join(MODELS_DIR, 'mmod_human_face_detector.dat')

    # Download dlib's facial shape predictor and unzip it:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    DLIB_FACIAL_SHAPE_PREDICTOR_PATH = join(MODELS_DIR, 'shape_predictor_5_face_landmarks.dat')
    FACE_ALIGNER_OUTPUT_SIZE = 182
    FACE_ALIGNER_LEFT_EYE = (0.35, 0.35)

    VALID_IMAGES = ('.jpg', '.png')

    # The face-detector may crash when the input image is so large, when the detector is cnn-based,
    #   i.e. DLIB_FACE_DETECTOR_PATH is not None
    MAX_INPUT_HEIGHT = 1024

    # When processing a video, skip detection on frames that have correlation coefficient of higher than this value vs
    #   the most recent frame
    CORRELATION_THRESHOLD_BETWEEN_FRAMES = 0.985

    # Sample video every () seconds when doing clustering on video
    VIDEO_SAMPLING_PERIOD = 1.0
