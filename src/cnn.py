import marimo

__generated_with = "0.17.2"
app = marimo.App(
    width="full",
    app_title="CNN",
    auto_download=["html", "ipynb"],
)

with app.setup:
    # Initialization code that runs before all other cells
    import numpy as np
    from matplotlib import pyplot as plt

    import glob

    from PIL import Image
    import cv2

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

    from keras.models import Model
    from keras.utils import to_categorical, set_random_seed
    from keras.models import Sequential
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Using CNNs to Classify Shapes
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Load datset""")
    return


@app.cell
def _():
    path_to_root_dataset = mo.notebook_dir() / 'dataset_cnn'
    shape_of_image = (126, 126) # resize all images to this shape
    return path_to_root_dataset, shape_of_image


@app.cell
def _(path_to_root_dataset, shape_of_image):
    datasets = {'train': {'images': [], 'labels': []}, 'validation': {'images': [], 'labels': []}, 'test': {'images': [], 'labels': []}}
    shapes = ['semicircle', 'triangle', 'circle', 'square', 'rhombus', 'rectangle', 'parallelogram', 'oval', 'trapezoid', 'pentagon']
    for dataset in datasets:
        for shape in shapes:
            image_paths = glob.glob(f'{path_to_root_dataset}/{dataset}/{shape}/*.png')
            for image_path in image_paths:
                _image = Image.open(image_path).convert('L')
                _image = _image.resize(shape_of_image)
                _image = (np.array(_image) / 255).astype('float16')
                datasets[dataset]['images'].append(_image)
                datasets[dataset]['labels'].append(shapes.index(shape))
                if len(datasets[dataset]['labels']) % 5000 == 0:
                    break
        datasets[dataset]['images'] = np.array(datasets[dataset]['images']).reshape(-1, shape_of_image[0], shape_of_image[1], 1)
        datasets[dataset]['labels'] = to_categorical(np.array(datasets[dataset]['labels']), num_classes=len(shapes))  # limit to 5000 images per shape in training set to not run out of memory
    return datasets, shapes


@app.cell
def _(datasets):
    print(f'Training dataset -> \t Images: {datasets['train']['images'].shape}, Labels: {datasets['train']['labels'].shape}')
    print(f'Validation dataset -> \t Images: {datasets['validation']['images'].shape}, Labels: {datasets['validation']['labels'].shape}')
    print(f'Test dataset -> \t\t Images: {datasets['test']['images'].shape}, Labels: {datasets['test']['labels'].shape}')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Define the CNN model""")
    return


@app.cell
def _(shape_of_image, shapes):
    set_random_seed(1)

    model = Sequential()
    model.add(Input(shape=(shape_of_image[0], shape_of_image[1], 1)))

    model.add(Conv2D(8,   (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16,  (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32,  (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64,  (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dense(64,  activation='elu'))
    model.add(Dense(len(shapes), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Train the model""")
    return


@app.cell
def _(datasets, model):
    history = model.fit(datasets['train']['images'], 
                        datasets['train']['labels'], 
                        epochs=9, batch_size=25, 
                        validation_data=(datasets['validation']['images'], 
                                         datasets['validation']['labels']), 
                        verbose='auto',
                        shuffle=True)
    return (history,)


@app.cell
def _(history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.ylim(0.75, 1.0)
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.suptitle('Training and Validation Metrics', fontsize=15)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1, wspace=0.25)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Evaluate the model""")
    return


@app.cell
def _(datasets, model, shapes):
    result = model.predict(datasets['test']['images'], verbose='auto')
    predicted_classes = np.argmax(result, axis=1)
    true_classes = np.argmax(datasets['test']['labels'], axis=1)

    print(classification_report(true_classes, predicted_classes, target_names=shapes))

    cm = confusion_matrix(true_classes, predicted_classes)
    cm_plot = ConfusionMatrixDisplay(cm, display_labels=shapes)
    cm_plot.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    cm_plot.ax_.set_title('Confusion Matrix')
    cm_plot.figure_.set_size_inches(8, 6)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Visualize the filters""")
    return


@app.cell
def _(model):
    for layer in model.layers:
        if 'conv' in layer.name:
            filter = layer.get_weights()[0]
            print(layer.name, model.layers.index(layer), filter.shape)
    return


@app.cell
def _(model):
    _filters, _biases = model.layers[0].get_weights()
    print(_filters.shape)
    _fig, _axs = plt.subplots(1, _filters.shape[3])
    _fig.set_size_inches(8, 4)
    for _i, _ax in enumerate(_axs):
        _ax.axis('off')
        try:
            _ax.imshow(_filters[:, :, 0, _i], cmap='gray', interpolation='none')
        except IndexError:
            pass
        _ax.set_title(f'F{_i}')
    plt.tight_layout()
    plt.suptitle('Some of the First Layer Filters', fontsize=15)
    plt.show()
    return


@app.cell
def _(model):
    _filters, _biases = model.layers[2].get_weights()
    print(_filters.shape)
    _fig, _axs = plt.subplots(8, 8)
    _fig.set_size_inches(8, 10)
    for _i, _ax in enumerate(_axs):
        for _j, _sub_ax in enumerate(_ax):
            _sub_ax.axis('off')
            _sub_ax.imshow(_filters[:, :, _j, _i], cmap='gray', interpolation='none')
            _sub_ax.set_title(f'F{_i}C{_j}')
    plt.suptitle('Some of the Second Layer Filters', fontsize=15)
    plt.show()
    return


@app.cell
def _(model):
    _filters, _biases = model.layers[8].get_weights()
    print(_filters.shape)
    _fig, _axs = plt.subplots(8, 8)
    _fig.set_size_inches(8, 10)
    for _i, _ax in enumerate(_axs):
        for _j, _sub_ax in enumerate(_ax):
            _sub_ax.axis('off')
            _sub_ax.imshow(_filters[:, :, _j, _i], cmap='gray', interpolation='none')
            _sub_ax.set_title(f'F{_i}C{_j}')
    plt.suptitle('Some of the Last Layer Filters', fontsize=15)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Visualize the feature maps""")
    return


@app.cell
def _(datasets, model, shape_of_image, shapes):
    test_img = datasets['test']['images'][np.random.randint(0, datasets['test']['images'].shape[0])]
    test_img = test_img.reshape(1, shape_of_image[0], shape_of_image[1], 1)
    print(test_img.shape)
    plt.figure(figsize=(3, 3))
    plt.imshow(test_img[0], cmap='gray', interpolation='none')
    plt.axis('off')
    plt.title('Random Test Image')
    plt.show()
    print("Predicted Class:", shapes[np.argmax(model.predict(test_img))])
    return (test_img,)


@app.cell
def _(model, test_img):
    view_layers = [2, 4, 6, 8, 9]
    _fig, all_axs = plt.subplots(len(view_layers), 16)
    _fig.set_size_inches(16, 8)
    for plt_idx, layer_idx in enumerate(view_layers):
        print(layer_idx)
        temp_model = Model(inputs=model.inputs, outputs=model.layers[layer_idx].output)
        feature_maps = temp_model.predict(test_img, verbose='auto')
        print(feature_maps.shape)
        _axs = all_axs[plt_idx]
        for _i, _ax in enumerate(_axs):
            _ax.axis('off')
            try:
                _ax.imshow(feature_maps[0, :, :, _i], cmap='gray', interpolation='none')
            except IndexError:
                pass
            _ax.set_title(f'L{layer_idx}C{_i}')
    plt.suptitle('Feature Maps of Different Layers for the Test Image', fontsize=15)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Segment and classify shapes from a image containing multiple shapes""")
    return


@app.cell
def _(model, shape_of_image, shapes):
    _image = cv2.imread(mo.notebook_location() / 'public' / 'shapes_used.jpeg')
    gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.title('Image containing multiple shapes')
    plt.show()
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_size = 256
    padding = 10
    for _i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(_image.shape[1], x + w + padding)
        y_end = min(_image.shape[0], y + h + padding)
        segment = _image[y_start:y_end, x_start:x_end]
        isolated_shape_image = np.ones((segment.shape[0], segment.shape[1], 3), dtype=np.uint8) * 255
        mask = np.zeros_like(gray[y_start:y_end, x_start:x_end])
        contour_offset = contour - np.array([x_start, y_start])
        cv2.drawContours(mask, [contour_offset], -1, 255, thickness=cv2.FILLED)
        if segment.shape[0] == mask.shape[0] and segment.shape[1] == mask.shape[1]:
            isolated_shape_image[mask == 255] = segment[mask == 255]
        else:
            continue
        isolated_shape_image = cv2.cvtColor(isolated_shape_image, cv2.COLOR_BGR2GRAY)
        h, w = isolated_shape_image.shape[:2]
        new_image = np.full((target_size, target_size), 255, dtype=np.uint8)
        scale = float(target_size) / max(h, w)
        new_w = round(w * scale)
        new_h = round(h * scale)
        if scale < 1:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        resized_image = cv2.resize(isolated_shape_image, (new_w, new_h), interpolation=interpolation)
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        new_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
        _, new_image = cv2.threshold(new_image, 200, 255, cv2.THRESH_BINARY_INV)
        new_image = Image.fromarray(new_image).resize(shape_of_image)
        print(new_image.size)
        new_image = ((255 - np.array(new_image)) / 255.0).astype('float16')
        plt.title(f'Segmented Image, predicted to be: {shapes[np.argmax(model.predict(new_image.reshape(1, shape_of_image[0], shape_of_image[1], 1)))]}', fontsize=10)
        plt.imshow(new_image, cmap='gray')
        plt.show()
    return


if __name__ == "__main__":
    app.run()
