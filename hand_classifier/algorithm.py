import operator
import os
import cv2
import numpy as np
from keras import backend as K
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model

from hand_classifier.utils import detector_utils


class Config(object):
    img_rows, img_cols = 200, 200
    height, width = 200, 200
    img_channels = 1
    batch_size = 32
    nb_classes = 20
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_ellipse2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    kernel_ellipse3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lastgesture = -1
    output = ["Scissors", "Scissors", "Others", "Others", "Rock", "Others", "Others", "Others", "Scissors", "Scissors",
              "Scissors", "Others", "Paper", "Paper", "Rock", "Scissors", "Others", "Others", "Others", "Others"]


class HandClassifier(object):
    def __init__(self, config=Config(), model_dir=os.path.join(os.path.dirname(__file__) + './weights/weights.hdf5')):
        self.config = config
        self.detection_graph, self.sess = detector_utils.load_inference_graph()
        input_sensor = Input(shape=(self.config.img_rows, self.config.img_cols, self.config.img_channels))
        x1 = Conv2D(self.config.nb_filters, (self.config.nb_conv, self.config.nb_conv), padding='same')(input_sensor)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(self.config.nb_filters, (self.config.nb_conv, self.config.nb_conv), padding='same')(x1)
        x1 = layers.add([x1, input_sensor])
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D(pool_size=(self.config.nb_pool, self.config.nb_pool))(x1)

        x2 = Conv2D(self.config.nb_filters, (self.config.nb_conv, self.config.nb_conv), padding='same')(x1)
        x2 = Activation('relu')(x2)
        x2 = Conv2D(self.config.nb_filters, (self.config.nb_conv, self.config.nb_conv), padding='same')(x2)
        x2 = layers.add([x2, x1])
        x2 = Activation('relu')(x2)
        x2 = MaxPooling2D(pool_size=(self.config.nb_pool, self.config.nb_pool))(x2)

        x = Dropout(0.5)(x2)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.config.nb_classes)(x)
        output = Activation('softmax')(x)

        self.model = Model(inputs=input_sensor, outputs=output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.model.summary()
        self.model.get_config()

        print("loading ", model_dir)
        self.model.load_weights(model_dir)
        self.layer = self.model.layers[-1]
        self.roi = 0
        # 0 others, 1 Rock, 2 Papers, 3 scis
        self.gesture_identifier = 0
        self.get_output = K.function([self.model.layers[0].input, K.learning_phase()], [self.layer.output, ])

    def get_identifier(self):
        return self.gesture_identifier

    def guess_gesture(self, img):
        image = np.array(img).flatten()
        image = image.reshape(self.config.img_rows, self.config.img_cols, self.config.img_channels)
        image = image.astype('float32')
        image = image / 255
        image = image.reshape(1, self.config.img_rows, self.config.img_cols, self.config.img_channels)
        prob_array = self.get_output([image, 0])[0]

        d = {}
        i = 0
        for items in self.config.output:
            d[items] = prob_array[0][i] * 100
            i += 1

        guess = max(d.items(), key=operator.itemgetter(1))[0]
        d[guess] = 100
        prob = d[guess]

        if prob > 99.0:
            return self.config.output.index(guess)
        else:
            return 1

    def detect(self, image):
        return detector_utils.detect_objects(image, self.detection_graph, self.sess)

    @staticmethod
    def draw_result(image, boxes, scores, score_thresh=0.2, im_width=640, im_height=480):
        return detector_utils.draw_box_on_image(1, score_thresh, scores, boxes, im_width, im_height, image)

    def image_preprocess(self, image):
        boxes, scores = self.detect(image)
        image = self.draw_result(image, boxes, scores)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.low_range, self.config.upper_range)
        erosion = cv2.erode(mask, self.config.kernel_ellipse, iterations=1)
        dilation = cv2.dilate(erosion, self.config.kernel_ellipse, iterations=1)
        gaussian_blur = cv2.GaussianBlur(dilation, (15, 15), 1)
        image = cv2.bitwise_and(image, image, mask=gaussian_blur)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rx, ry = image.shape
        if rx > 0 and ry > 0:
            image = cv2.resize(image, (self.config.width, self.config.height), interpolation=cv2.INTER_CUBIC)
        return image

    @staticmethod
    def gesture_postprocess(retgesture):
        if retgesture == 4 or retgesture == 14:
            return 1
        elif retgesture == 12 or retgesture == 13:
            return 2
        elif retgesture == 0 or retgesture == 1 or retgesture == 8 or retgesture == 9 or retgesture == 10 or retgesture == 15:
            return 3
        return 0

    def classify(self, image):
        image = self.image_preprocess(image)
        gesture = self.guess_gesture(image)
        result = self.gesture_postprocess(gesture)
        return image, result


if __name__ == '__main__':
    model = HandRecognizer()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    im_width, im_height = (cap.get(3), cap.get(4))
    font = cv2.FONT_HERSHEY_COMPLEX

    while True:
        try:
            ret, image_np = cap.read()
            if ret:
                retgesture = model.guess_gesture(model.image_preprocess(image_np))
                cv2.putText(image_np, model.config.output[retgesture], (15, 40), font, 0.75, (77, 255, 9), 2)
                detector_utils.draw_fps_on_image(None, image_np)
                cv2.imshow('RPS', image_np)
                cv2.moveWindow('RPS', 0, 0)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        except Exception as err:
            print(err)
            print("Did not detect hand, put hand within the camera's frame!")
            continue
