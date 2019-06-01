from keras.models import load_model
import numpy as np
import copy
from utils import *
from projectParams import classes
import asyncio

from cnn12 import modelPath
from cnn12 import modelWeights
from cnn12 import imgDim

model = load_model(modelPath)
model.load_weights(modelWeights)
dataColor = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45
className = ''
sentence = ""
count = 0
showMask = 0
freq = 50


async def predictImg(roi):
    global count, className, sentence

    count = count + 1
    if count % freq == 0:
        img = cv2.resize(roi, (imgDim, imgDim))
        img = np.float32(img) / 255.
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        vec = model.predict(img)
        maxVal = np.amax(vec)

        if maxVal > 0.7:  # 70%
            pred = classes[np.argmax(vec[0])]
            pred = convertEnglishToHebrewLetter(pred)
            if pred == 'del':
                sentence = sentence[:-1]
            else:
                sentence = sentence + pred
            if pred == ' ':
                pred = 'space'
            print('prediction: ' + pred)
            print(finalizeHebrewString(sentence))
        else:
            className = ''


def main():
    global font, fx, fy, fh
    global dataColor, predict
    global className, count
    global showMask, window

    x0, y0, width = 400, 50, 224
    predict = 0
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    while True:
        # Get camera frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)  # mirror
        window = copy.deepcopy(frame)
        cv2.rectangle(window, (x0, y0), (x0 + width - 1, y0 + width - 1), dataColor, 12)

        if predict:
            dataColor = (0, 250, 0)
            cv2.putText(window, 'Prediction: ON', (fx, fy), font, 1.2, dataColor, 2, 1)
        else:
            dataColor = (0, 0, 250)
            cv2.putText(window, 'Prediction: OFF', (fx, fy), font, 1.2, dataColor, 2, 1)

        # get region of interest
        roi = frame[y0:y0 + width, x0:x0 + width]
        roi = binaryMask(roi)

        # apply processed roi in frame
        if showMask:
            img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            window[y0:y0 + width, x0:x0 + width] = img

        # take data or apply predictions on ROI
        if predict:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(predictImg(roi))

            # use below for demoing purposes
            # cv2.putText(window, 'Prediction: %s' % (pred), (x0,y0-25), font, 1.0, (255,0,0), 2, 1)
        # cv2.putText(window, 'Prediction: %s' % className, (fx, fy + 2 * fh), font, 1.0, (245, 210, 65), 2, 1)
        cv2.imshow('Original', window)

        # Keyboard inputs
        key = cv2.waitKeyEx(10)

        # use ESC key to close the program
        if key & 0xff == 27:
            break

        elif key & 0xff == 255:  # nothing pressed
            continue

        # adjust the position of window
        elif key == 2490368:  # up
            y0 = max((y0 - 5, 0))
        elif key == 2621440:  # down
            y0 = min((y0 + 5, window.shape[0] - width))
        elif key == 2424832:  # left
            x0 = max((x0 - 5, 0))
        elif key == 2555904:  # right
            x0 = min((x0 + 5, window.shape[1] - width))

        key = key & 0xff
        if key == ord('m'):  # mask
            showMask = not showMask
        if key == ord('p'):  # mask
            predict = not predict

    cam.release()


if __name__ == '__main__':
    main()
