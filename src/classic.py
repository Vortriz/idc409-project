import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import os
    import math

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    import marimo as mo


@app.cell(hide_code=True)
def _(cv, img_blurred):
    def old():
        corners = cv.cornerHarris(src=img_blurred, blockSize=3, ksize=5, k=1e-4)
        threshold = (np.min(corners) + np.max(corners)) / 2
        plt.imshow(corners > threshold, cmap="gray")

        groups = []

        for x, y in zip(*np.where(corners > threshold)):
            if len(groups) == 0:
                groups.append([(x, y)])

            distance_threshold = 10
            for group in groups:
                xt, yt = group[0]
                if (
                    np.linalg.norm(np.array((x, y)) - np.array((xt, yt)))
                    < distance_threshold
                ):
                    group.append((x, y))
                    break
            else:
                groups.append([(x, y)])

        num_corners = len(groups)
        num_corners

        # for group in groups :
        #     xs = []
        #     ys = []
        #     for x, y in group :
        #         xs.append(x)
        #         ys.append(y)
        #     print(np.mean(np.array(xs)), np.mean(np.array(ys)))

        # Apply thresholding in the gray image to create a binary image
        ret, thresh = cv.threshold(img_blurred, 150, 255, 0)

        # Find the contours using binary image
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        print("Number of contours in image:", len(contours))
        cnt = contours[0]

        # compute the area and perimeter
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)

    return


@app.function
def preprocessing(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # to reduce noise

    # threshold the image to convert to binary
    _, thresholded_image = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    return thresholded_image


@app.function
def find_contour(thresholded_image):
    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # assume the largest contour is our shape
    contour = max(contours, key=cv2.contourArea)

    # filter out tiny contours that are likely noise
    if cv2.contourArea(contour) < 100:
        return None

    return contour


@app.function
def extract_features_from_contour(contour):
    # Hu moments
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None  # avoid divide-by-zero

    hu_moments = cv2.HuMoments(M).flatten()
    # to make them more stable and comparable
    for i in range(7):
        hu_moments[i] = (
            -1
            * math.copysign(1.0, hu_moments[i])
            * math.log10(abs(hu_moments[i]) + 1e-7)
        )

    # corners
    perimeter = cv2.arcLength(contour, True)
    epsilon = (
        0.03 * perimeter
    )  # 3% of perimeter is a good starting point for approximation
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_corners = len(approx)

    # solidity
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    # circularity
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

    # final feature vector
    features = np.append([num_corners, solidity, aspect_ratio, circularity], hu_moments)

    return features


@app.function
def extract_features(image_path):
    thresholded_image = preprocessing(image_path)
    contour = find_contour(thresholded_image)
    if contour is None:
        return None
    features = extract_features_from_contour(contour)

    return features


@app.cell
def _():
    shapes = os.listdir("src/dataset/train/")
    print(shapes)
    return


@app.cell
def _():
    chosen_shape = "rectangle"
    print(extract_features(f"src/dataset/train/{chosen_shape}/{chosen_shape}_22.png"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
