import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import os

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    import marimo as mo


@app.cell
def _():
    from classic import preprocessing, find_contour, extract_features_from_contour, extract_features
    return (extract_features,)


@app.cell
def _():
    shape = 'pentagon'
    image = cv2.imread(mo.notebook_dir() / f'../src/dataset/{shape}/{shape}_23.png')
    plt.imshow(image)
    return (image,)


@app.cell
def _(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    return (gray,)


@app.cell
def _(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    plt.imshow(blur, cmap='gray')
    return (blur,)


@app.cell
def _(blur):
    threshold_value = np.max(blur)/2 + np.min(blur)/2
    _, thresholded_image = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(thresholded_image, cmap='gray')
    return (thresholded_image,)


@app.cell
def _(thresholded_image):
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    len(contours)
    return (contours,)


@app.cell
def _(contours, image):
    plt.imshow(cv2.drawContours(image, contours, -1, (0,255,0), 3))
    return


@app.cell
def _(contours):
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    M
    return M, contour


@app.cell
def _(contour, image):
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    plt.imshow(cv2.drawContours(image, [approx], -1, (0,255,0), 3))
    return


@app.cell
def _(M):
    hu_moments = cv2.HuMoments(M).flatten()
    hu_moments
    return (hu_moments,)


@app.cell
def _(hu_moments):
    hu_moments_normalized = -1 * np.copysign(1.0, hu_moments) * np.log10(abs(hu_moments) + 1e-7)
    hu_moments_normalized
    return


@app.cell
def _(extract_features):
    DATASET_DIR = mo.notebook_dir() / "dataset"
    LABELS = [
        "parallelogram",
        "triangle",
        "pentagon",
        "rectangle",
        "square",
        "circle",
        "trapezoid",
        "oval",
        "semicircle",
        "rhombus",
    ]

    samples = []
    samples_hu_moments = []
    images = []
    for label in LABELS:
        sample = os.path.join(DATASET_DIR, label, f'{label}_11.png')
        samples.append(sample)
        images.append(mo.image(sample))
        features = np.round(extract_features(sample)[-7:], 3)
        samples_hu_moments.append(features)

    df = pd.DataFrame(samples_hu_moments, index=LABELS, columns=[f'H[{i+1}]' for i in range(7)])
    df["images"] = images
    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
