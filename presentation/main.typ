#import "@preview/touying:0.5.5": *
#import "theme.typ": *

#show: university-theme.with(
  config-colors(
    primary: rgb("#13496d"),
    secondary: rgb("#00708f"),
    tertiary: rgb("#00989d"),
    neutral-lightest: rgb("#ffffff"),
    neutral-darkest: rgb("#000000"),
  ),
  config-info(
    title: [Shape detection],
    subtitle: [by methods of Feature Extraction and CNN],
    author: [Sparsha Ray #h(5em) Rishi Vora],
    date: [8th November 2025],
    institution: [],
  ),
  aspect-ratio: "16-9"
)


#set heading(numbering: "1.1 ")
#show figure.caption: set text(15pt)

#title-slide(extra: [IDC409 Project])


// ---------------------------------------------------------
= Dataset
// ---------------------------------------------------------

== Samples
// This is to give an idea of what we are trying to classify
// How many images per class

#align(center)[
  #set figure(supplement: none)
#grid(
  columns: (100pt,100pt,100pt,100pt,100pt),
  rows: 150pt,
  figure(image("assets/shape_circle.png"), caption: [`circle`]),
  figure(image("assets/shape_oval.png"), caption: [`oval`]),
  figure(image("assets/shape_parallelogram.png"), caption: [`parallelogram`]),
  figure(image("assets/shape_pentagon.png"), caption: [`pentagon`]),
  figure(image("assets/shape_rectangle.png"), caption: [`rectangle`]),
  figure(image("assets/shape_rhombus.png"), caption: [`rhombus`]),
  figure(image("assets/shape_semicircle.png"), caption: [`semicircle`]),
  figure(image("assets/shape_square.png"), caption: [`square`]),
  figure(image("assets/shape_trapezoid.png"), caption: [`trapezoid`]),
  figure(image("assets/shape_triangle.png"), caption: [`triangle`]),
)]

== Stats

- 10 labels
- 12000 images for each label
- All images are `224x224` pixels.
- Each class contains shapes that are rotated, scaled and differently coloured.


// ---------------------------------------------------------
= Feature Extraction
// ---------------------------------------------------------

== Overview

- The core idea is to extract the feature vector manually and then feed it to a classifier.
- The features that we chose are:
  - Number of corners
  - Solidity
  - Aspect ratio
  - Circularity
  - Hu Moments
- For classification, we chose Random Forests Classifier.

We use *OpenCV* for all image manipulation.

== Preprocessing

- Read the image
- Make it gray-scale
- Blur it to reduce noise
- Threshold it to make it binary

#align(center)[
  #set figure(supplement: none)
  #grid(
    columns: 4,
    rows: 150pt,
    figure(image("assets/pr_og.png"), caption: [original]),
    figure(image("assets/pr_bw.png"), caption: [grayscaled]),
    figure(image("assets/pr_bl.png"), caption: [blurred]),
    figure(image("assets/pr_th.png"), caption: [binary thresholded]),
  )
]

== Finding contour

#align(center + horizon)[
  #set figure(supplement: none)
  #grid(
    columns: 2,
    rows: 150pt,
    column-gutter: 5em,
    figure(image("assets/pr_og.png"), caption: [original]),
    figure(image("assets/contour.png"), caption: [with contour]),
  )
]

#pagebreak()

== Feature extraction

1. Number of corners
  - great first filter
  
  - can categorize shapes into broad groups like:
    - `triangles` (`3` corners)
    - `squares` + `rectangles` + `parallelogram` + `rhombus` (`4` corners)
    - `semicircle` (`2` corners)
    - `circle` + `oval` (no corners)

#pagebreak()

2. Solidity

#grid(
  columns: 2,
  align: center + horizon,
  figure(
    image("assets/solidity.png", height: 60%)
  ),
  $ "Solidity" = "Convex hull area"/"Area" $
)

- All the shapes we chose have solidity 1.0 but it is a great filter for concave shapes like stars.

#pagebreak()

3. Circularity (Roundness)

#grid(
  columns: 2,
  align: center + horizon,
  figure(
    image("assets/roundness.png", height: 60%)
  ),
  $ "Circularity" = (4 pi * "Area")/"Perimeter"^2 $
)

- A perfect `circle` will have a score of exactly 1.0.

#pagebreak()

4. Aspect Ratio

- We draw the tightest upright box around the shape and divides its `width` by its `height`.

#pause

- `square` and `circle` will have an aspect ratio of 1.0.

- `rectangle` and `oval` will have an aspect ratio not equal to 1.0.

#pause

- But this can be easily fooled by rotation.

#pagebreak()

5. Hu Moments

- Set of 7 numbers that can be considered unique "mathematical fingerprint" for a shape's geometry.

- Derived from the field of Statistics

The central moments are defined as

$ mu_(i j) = sum_x sum_y (x - dash(x))^i (y - dash(y))^j thin I(x, y) $

where $(x,y)$ are pixel coordinates, $(dash(x), dash(y))$ is the centroid of the shape, and $I(x,y)$ are the intensity values.

Central moments are translation invariant.

#pagebreak()

Using these, we get normalized central moments
  
$ eta_(i j) = mu_(i j)/mu_00^((1 + (i + j)/2)) $

These are invariant to scaling and translation both.

Using these, M. K. Hu, derived a set of 7 moments that are invariant to translation, scaling, and rotation.

#pagebreak()

#text(size: 17pt)[
$
  I_1 & = eta_20 + eta_02 \
  I_2 & = (eta_20 - eta_02)^2 + 4 eta_11^2 \
  I_3 & = (eta_20 - 3 eta_12)^2 +(3 eta_11 - eta_02)^2 \
  I_4 & = (eta_20 - eta_12)^2 +(eta_21 + eta_13)^2 \
  I_5 & = (eta_20 - 3 eta_12)(eta_20 + eta_12) [(eta_20 + eta_12)^2 - 3(eta_21 + eta_13)^2 ] \
  & #h(1em) + (3 eta_21 - eta_13)(eta_21 + eta_13) [3(eta_20 + eta_12)^2 -(eta_21 + eta_13)^2 ] \
  I_6 & = (eta_20 - eta_12)(eta_20 + eta_12)^2 -(eta_21 + eta_33)^2 ] \
  & #h(1em) + 4 eta_11 (eta_20 + eta_12)(eta_21 + eta_13) \
  I_7 & = (8 eta_21 - eta_30)(eta_30 + eta_22) [(eta_30 + eta_22)^2 - 3(eta_21 + eta_33)^2 ] \
  & #h(1em) - (eta_30 - 3 eta_22)(eta_21 + eta_33) [3(eta_30 + eta_12)^2 -(eta_21 + eta_33)^2 ]
$]

#pagebreak()

#figure(
  image("assets/Hu_table.pdf")
)

== Training

- Now we have 11 features (corners, solidity, aspect ratio, circularity, and 7 Hu moments.)

- We train a Random Forests Classifier with
  - 100 estimators
  - 120k samples
  - 80% training, 20% testing samples


== Evaluation

#pause
- Model Accuracy: *99.11%*

#pause
#align(center, text(size: 18pt, [
```
               precision    recall  f1-score   support

parallelogram       1.00      1.00      1.00      2400
     triangle       1.00      1.00      1.00      2400
     pentagon       1.00      1.00      1.00      2400
    rectangle       1.00      1.00      1.00      2400
       square       0.97      0.94      0.95      2400
       circle       1.00      1.00      1.00      2400
    trapezoid       1.00      1.00      1.00      2400
         oval       1.00      1.00      1.00      2400
   semicircle       1.00      1.00      1.00      2400
      rhombus       0.94      0.97      0.96      2400

     accuracy                           0.99     24000
    macro avg       0.99      0.99      0.99     24000
 weighted avg       0.99      0.99      0.99     24000
```
]))

#pagebreak()

#grid(
  columns: 2,
  align: end,
  figure(image("assets/confusion_matrix_classic.png")),
  pause,
  figure(image("assets/roc_classic.png"))
)

#pagebreak()

We also made a function recognize all present shapes from an image, by extracting all contours.

#grid(
  rows: 2,
  align: center + horizon,
  column-gutter: 5em,
  figure(
    image("assets/shapes_used.png", height: 35%)
  ),
  image("assets/classic_multi.png")
)

It mostly correctly identified all the shapes.

// ---------------------------------------------------------
= Convolutional Neural Networks
// ---------------------------------------------------------

== What is convolution ?


#alternatives[
            ][#box(height: 400pt, image("assets/conv1.gif"))
            ][#box(height: 400pt, image("assets/conv2.gif"))
            ][#box(height: 400pt, image("assets/conv3.gif"))
            ][#box(height: 400pt, image("assets/conv4.gif"))
            ][#box(height: 400pt, image("assets/conv5.gif"))]


== What are filters ?

#grid(
  columns: (1fr, 1fr),
  rows: (1fr, 3fr),
  gutter: 3pt,

  grid.cell(
    colspan: 2,
    text[
         Consider the filter (or kernel) $mat(-3+3i, +10i, 3+3i; 
                                              -10,     0,    10;
                                              -3-3i, -10i, 3-3i;)$
        ]
  ),
  image("assets/Bikesgray.jpg", width:100%),
  image("assets/Bikesgray-scharr.png", width:100%),
)

== How does CNNs work ?

#pause
- In CNNs, a image is passed through a series of filters (which the learnt as the model is trained) ...#pause
- Which extract features like edges corners etc (called feature maps) ...#pause
- As the feature maps are down-sampled and passed down through more and more layers of filters, specific filters starts to learn more complex higher order characteristics (like curvature, spacing between corners, parallel edges etc)#pause
- When these features are extracted, and the feature maps consists of only a few pixels, their values are passed down to a fully connected feed forward neural network (often with only 2 or 3 layers) which does the final classification task.

== Architecture

#image("assets/cnn_arch.png") 
#text(size:9pt)[Image courtesy of Dwika Sudrajat's blog post]

#show table.cell: set text(size: 18pt)

#align(center)[
#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: center+horizon,
  table.header(
    [*Layer (type)*], [*Output Shape*], [*Parameters*],
  ),
  "conv2d_5 (Conv2D)"             , "(None, 124, 124, 8)"  ,"80",
  "max_pooling2d_5 (MaxPooling2D)", "(None, 62, 62, 8)"    ,"0",
  "conv2d_6 (Conv2D)"             , "(None, 60, 60, 16)"   ,"1,168",
  "max_pooling2d_6 (MaxPooling2D)", "(None, 30, 30, 16)"   ,"0",
  "conv2d_7 (Conv2D)"             , "(None, 28, 28, 32)"   ,"4,640",
  "max_pooling2d_7 (MaxPooling2D)", "(None, 14, 14, 32)"   ,"0",
  "conv2d_8 (Conv2D)"             , "(None, 12, 12, 64)"   ,"18,496",
  "max_pooling2d_8 (MaxPooling2D)", "(None, 6, 6, 64)"     ,"0",
  "conv2d_9 (Conv2D)"             , "(None, 4, 4, 128)"    ,"73,856",
  "max_pooling2d_9 (MaxPooling2D)", "(None, 2, 2, 128)"    ,"0",
  "flatten_1 (Flatten)"           , "(None, 512)"          ,"0",
  "dense_3 (Dense)"               , "(None, 128)"          ,"65,664",
  "dense_4 (Dense)"               , "(None, 64)"           ,"8,256",
  "dense_5 (Dense)"               , "(None, 10)"           ,"650",
)
]

== Training

Training is done with the `Adam` optimizer for 9 epochs only
#image("assets/training_stats.png")

== Results
#slide(composer: (1fr, 1.25fr))[
  #align(left+horizon)[#box(image("assets/confusion_matrix_cnn.png"))]
][#table(
  columns: (auto, auto, auto, auto),
  inset: 8.9pt,
  align: center+horizon,
  table.header(
    [*Class*], [*Precision*], [*Recall*], [*f1-score*]
  ),
   "semicircle",       "0.99",      "0.99",      "0.99",
     "triangle",       "0.98",      "1.00",      "0.99",
       "circle",       "1.00",      "0.99",      "0.99",
       "square",       "0.99",      "1.00",      "0.99",
      "rhombus",       "0.99",      "1.00",      "1.00",
    "rectangle",       "1.00",      "0.99",      "0.99",
"parallelogram",       "1.00",      "1.00",      "1.00",
         "oval",       "1.00",      "0.99",      "1.00",
    "trapezoid",       "1.00",      "0.99",      "0.99",
     "pentagon",       "0.99",      "0.99",      "0.99",
     
emph("Overall Accuracy"),  "",          "",      "0.99",
)]

== Interpretations

#grid(
  columns: (1fr, 15pt, 2fr),
  image("assets/filters.png"),
  "",
  align(horizon)[#box(image("assets/feature maps.png"))]

)


= Conclusion

== How does these two methods compare
- Both reaches similar accuracy, CNN is slower, Feature Extraction + Random Forests (RF) is much faster
- CNN learns the features by itself, FE+RF is given the features to classify
- FE+RF doesn't generalize well for complex shapes, CNN does



// == Bibliography
// #bibliography("references.bib")

#focus-slide[#align(center)[Thank you! \ \ #text(size: 30pt, "Questions?")]]