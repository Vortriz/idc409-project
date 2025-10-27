import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])

with app.setup:
    # Initialization code that runs before all other cells
    import os
    from collections import Counter

    import cv2
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelBinarizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
        RocCurveDisplay,
    )


    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Model""")
    return


@app.function
def preprocessing(image_path, threshold_value=None):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # to reduce noise

    # threshold the image to convert to binary
    if threshold_value is None:
        threshold_value = np.max(blur) / 2 + np.min(blur) / 2
    _, thresholded_image = cv2.threshold(
        blur, threshold_value, 255, cv2.THRESH_BINARY_INV
    )

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
    hu_moments = (
        -1 * np.copysign(1.0, hu_moments) * np.log10(abs(hu_moments) + 1e-7)
    )

    # corners
    perimeter = cv2.arcLength(contour, True)
    epsilon = (
        0.03 * perimeter
    )  # 3% of perimeter is a good starting point for approximation
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_corners = len(approx)

    # solidity
    area = M["m00"]  # cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    # circularity
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

    # final feature vector
    features = np.append(
        [num_corners, solidity, aspect_ratio, circularity], hu_moments
    )

    return features


@app.function
def extract_features(image_path):
    thresholded_image = preprocessing(image_path)
    contour = find_contour(thresholded_image)
    if contour is None:
        return None
    features = extract_features_from_contour(contour)

    return features


@app.function
@mo.cache
def load_data(dataset_path, labels):
    all_features = []
    all_labels = []

    feature_names = [
        "num_corners",
        "solidity",
        "aspect_ratio",
        "circularity",
        "hu1",
        "hu2",
        "hu3",
        "hu4",
        "hu5",
        "hu6",
        "hu7",
    ]

    for label in mo.status.progress_bar(
        labels, title=f"Loading dataset of {len(labels)} labels"
    ):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            print(f"Warning: Label directory not found: {label_path}")
            continue

        for filename in mo.status.progress_bar(
            os.listdir(label_path),
            title=f"Extracting features for label '{label}'",
            remove_on_exit=True,
        ):
            if filename.endswith(".png"):
                image_path = os.path.join(label_path, filename)
                features = extract_features(image_path)

                if features is not None and len(features) == len(
                    feature_names
                ):
                    all_features.append(features)
                    all_labels.append(label)

    df = pd.DataFrame(all_features, columns=feature_names)
    df["label"] = all_labels

    return df


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Evaluation""")
    return


@app.cell
def _():
    DATASET_DIR = mo.notebook_dir() / "dataset_classic"
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

    data = load_data(DATASET_DIR, LABELS)

    mo.stop(data.empty, mo.md("**Error**: No data was loaded. Exiting!"))
    mo.output.append(
        mo.md(f"Successfully loaded and processed **{len(data)}** samples.")
    )
    return LABELS, data


@app.cell
def _(data):
    X = data.drop("label", axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # to ensure all classes are represented in train/test splits
    )
    return X, X_test, X_train, y_test, y_train


@app.cell
def _(X):
    X
    return


@app.cell
def _(X_train, y_train):
    mo.output.append("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    mo.output.append("Training complete.")
    return (model,)


@app.cell
def _(X_test, model, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mo.output.append(mo.md(f"Model Accuracy: **{acc * 100:.2f}%**"))
    return (y_pred,)


@app.cell
def _(LABELS, y_pred, y_test):
    mo.output.append(mo.md("**Classification Report:**"))
    print(classification_report(y_test, y_pred, labels=LABELS))
    return


@app.cell
def _(model, y_pred, y_test):
    C = confusion_matrix(y_test, y_pred, labels=model.classes_)

    ax = sns.heatmap(
        C, annot=True, fmt=".0f", cmap=sns.color_palette("Blues", as_cmap=True)
    )
    ax.set_xticklabels(
        model.classes_, rotation=45, rotation_mode="anchor", ha="right"
    )
    ax.set_yticklabels(model.classes_, rotation=0)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    return


@app.cell
def _(X_test, model, y_test, y_train):
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_score = model.predict_proba(X_test)

    display = RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_score.ravel(),
        name="micro-average OvR",
        curve_kwargs=dict(color="darkorange"),
        plot_chance_level=True,
        despine=True,
    )
    display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Micro-averaged One-vs-Rest\nReceiver Operating Characteristic",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Multi object image""")
    return


@app.function
def find_all_contours(thresholded_image):
    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= 100
    ]

    return filtered_contours


@app.function
def extract_features_from_all_shapes(image_path, threshold_value=None):
    feature_names = [
        "num_corners",
        "solidity",
        "aspect_ratio",
        "circularity",
        "hu1",
        "hu2",
        "hu3",
        "hu4",
        "hu5",
        "hu6",
        "hu7",
    ]

    thresholded_image = preprocessing(image_path, threshold_value)
    contours = find_all_contours(thresholded_image)
    mo.output.append(f"Found {len(contours)} contours in the image.")

    if len(contours) == 0:
        return None

    all_contours_features = []
    for contour in contours:
        all_contours_features.append(extract_features_from_contour(contour))

    df = pd.DataFrame(all_contours_features, columns=feature_names)

    return df


@app.cell
def _():
    multi_shape_image = mo.notebook_dir() / "shapes_used.png"
    mo.image(multi_shape_image)
    return (multi_shape_image,)


@app.cell
def _(model, multi_shape_image):
    all_shapes_features = extract_features_from_all_shapes(
        multi_shape_image, threshold_value=250
    )
    predicted_shapes = model.predict(all_shapes_features)
    Counter(predicted_shapes)
    return


if __name__ == "__main__":
    app.run()
