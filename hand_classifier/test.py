from utils import detector_utils as detector_utils
import cv2
import datetime
import argparse
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('tf')
import numpy as np
import operator

img_rows, img_cols = 200, 200
img_channels = 1
batch_size = 32
nb_classes = 20
nb_filters = 32
nb_pool = 2
nb_conv = 3

low_range = np.array([0, 50, 80])
upper_range = np.array([30, 200, 255])
kernel_square = np.ones((11,11),np.uint8)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_ellipse2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
kernel_ellipse3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
height = 200
width = 200

lastgesture = -1
gestname = ""
path = ""
mod = 0
fname = "weights/weights.hdf5"
output = ["Scissors","Scissors","Others","Others","Rock","Others","Others","Others","Scissors","Scissors",
          "Scissors","Others","Paper","Paper","Rock","Scissors","Others","Others","Others","Others"]

detection_graph, sess = detector_utils.load_inference_graph()
global roi
#0 others, 1 Rock, 2 Papers, 3 scis
global gesture_identifier
gesture_identifier = 0

def getIdentifier():
    return gesture_identifier

def guessGesture(model, img):
    global output, get_output

    image = np.array(img).flatten()
    image = image.reshape(img_rows,img_cols, img_channels)
    image = image.astype('float32')
    image = image / 255
    rimage = image.reshape(1, img_rows, img_cols, img_channels)
    prob_array = get_output([rimage, 0])[0]

    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1

    guess = max(d.items(), key=operator.itemgetter(1))[0]
    d[guess] = 100
    prob  = d[guess]

    if prob > 60.0:
        return output.index(guess)
    else:
        return 1

def skinMask(roi):
    global  mod , retgesture

    retgesture = guessGesture(mod, roi)


def loadCNN():
    global get_output

    input_sensor = Input(shape=(img_rows, img_cols, img_channels))
    x1 = Conv2D(nb_filters, (nb_conv, nb_conv), padding='same')(input_sensor)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(nb_filters, (nb_conv, nb_conv), padding='same')(x1)
    x1 = layers.add([x1, input_sensor])
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x1)

    x2 = Conv2D(nb_filters, (nb_conv, nb_conv), padding='same')(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(nb_filters, (nb_conv, nb_conv), padding='same')(x2)
    x2 = layers.add([x2, x1])
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x2)

    x = Dropout(0.5)(x2)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes)(x)
    output = Activation('softmax')(x)

    model = Model(inputs=input_sensor, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.summary()
    model.get_config()


    print("loading ", fname)
    model.load_weights(fname)
    layer = model.layers[-1]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])

    return model

if __name__ == '__main__':

    global retgesture

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.2, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    mod = loadCNN()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    num_hands_detect = 1
    font = cv2.FONT_HERSHEY_COMPLEX
    flag = False
    
    while True:
        ret, image_np = cap.read()
        image_np = cv2.flip(image_np, 3)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        boxes, scores = detector_utils.detect_objects(
            image_np, detection_graph, sess)

        roi = detector_utils.draw_box_on_image(
            num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)

        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, low_range, upper_range)
            erosion = cv2.erode(mask, kernel_ellipse, iterations=1)
            dilation = cv2.dilate(erosion, kernel_ellipse, iterations=1)
            gaussianBlur = cv2.GaussianBlur(dilation, (15, 15), 1)
            res = cv2.bitwise_and(roi, roi, mask=gaussianBlur)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            rx, ry = res.shape
            if rx > 0 and ry > 0:
                res = cv2.resize(res, (width, height), interpolation=cv2.INTER_CUBIC)

            if ret == True:
                if rx > 0 and ry > 0:
                    skinMask(res)
            flag = True
            #print(flag)
        except:
            print("Did not detect hand, put hand within the camera's frame!")
            continue
        #sys.exit(0)

        if (args.display > 0):
            if flag == True:
                cv2.putText(image_np,output[retgesture],(15,40),font,0.75, (77, 255, 9), 2)
            if (args.fps > 0):
                detector_utils.draw_fps_on_image(None, image_np)
            if flag == True:
                if retgesture == 4 or retgesture == 14:
                    gesture_identifier = 1
                elif retgesture == 12 or retgesture == 13:
                    gesture_identifier = 2
                elif retgesture == 0 or retgesture == 1 or retgesture == 8 or retgesture == 9 or retgesture == 10 or retgesture == 15:
                    gesture_identifier = 3
                else:
                    gesture_identifier = 0
            
            print(gesture_identifier)
            cv2.imshow('RPS', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            cv2.moveWindow('RPS',0,0)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

