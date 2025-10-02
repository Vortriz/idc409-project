import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import os

    import matplotlib.pyplot as plt
    import numpy as np


@app.cell
def _(cv, img_blurred):
    def processing_test():
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
