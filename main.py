import cv2 as cv
import numpy as np


def canny(img):
    cvt_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(cvt_img, (3, 3), 0)
    canny_out = cv.Canny(blur, 50, 150)
    return canny_out


def roi(img):
    height = img.shape[0]
    dots = np.array([
        [(50, height), (450, 400), (670, 400), (920, height)]
    ])
    mask = np.zeros_like(img)
    cv.fillPoly(mask, dots, 255)
    filled_img = cv.bitwise_and(img, mask)
    return filled_img


def display_lines(img, lines):
    mask = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(mask, (x1, y1), (x2, y2), (255, 0, 0), 15)
    return mask


def averaged_slope_intercept_to_coordinates(img, lines):
    global left_line, right_line
    left_fit = []
    right_fit = []
    line_array = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if (left_fit != []):
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinate_parameters(img, left_fit_average)
        line_array.append(left_line)
    if (right_fit != []):
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinate_parameters(img, right_fit_average)
        line_array.append(right_line)
    return line_array


def make_coordinate_parameters(img, line):
    slope, intercept = line
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def main():
    capture = cv.VideoCapture('roadview.mp4')
    while (capture.isOpened()):
        _, frame = capture.read()
        canny_img = canny(frame)
        cropped_canny_img = roi(canny_img)
        lines = cv.HoughLinesP(cropped_canny_img, 1, np.pi / 180, 100,
                               np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = averaged_slope_intercept_to_coordinates(frame, lines)
        line_img = display_lines(frame, averaged_lines)
        weighted_img = cv.addWeighted(frame, 0.9, line_img, 1, 1)
        cv.imshow('Result (with Image and Lines)', weighted_img)
        if cv.waitKey(1) == ord('s'):
            break
    capture.release()
    cv.destroyAllWindows()

if __name__ == main():
    main()