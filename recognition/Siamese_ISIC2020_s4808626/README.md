# Siamese Network for Melanoma Classification (ISIC 2020)

## Feasibility Review

### 1. User Need, Scope and Acceptance Criteria

**User need**

The intended user is a skin-cancer screening clinic, where staff photograph each patient's lesions with a dermatoscope and a specialist reviews the images. Patients bring many lesions (a median of 12 per patient in ISIC 2020 [3]), so specialists cannot review every image promptly. Review delays matter: melanoma can spread to other parts of the body if it is not found early [1], and five-year relative survival falls from 100% when it is localised to 34% once it has spread to distant organs [2].

The clinic needs a triage assistant that:

1. **Separates melanoma from benign lesions reliably**, so suspicious lesions reach a specialist sooner and clearly benign ones do not use up review time (tested by AC1).
2. **Rarely misses a melanoma.** A missed melanoma can be fatal, while a false alarm costs only one extra specialist review, so the tool must favour catching melanoma over avoiding false alarms (AC4).
3. **Knows when it is unsure.** Only 1.8% of images are melanoma and many benign lesions look like melanoma [3], so some cases will be genuinely hard. Deep networks are often confidently wrong [4], so the clinic needs a confidence score it can trust. Uncertain cases must go to a specialist instead of being decided automatically, an approach known as selective prediction or a reject option [5] (AC4).
4. **Holds up on new patients.** The tool will only ever see patients it was not trained on, so its reported performance must come from patients it has never seen (AC2).
5. **Fits into the existing workflow**, running on modest hardware and returning a result quickly enough not to slow down review (AC5).

**Scope**

In scope:
- Binary classification of a single dermoscopic image as melanoma or benign, using the ISIC 2020 training set (33,126 images, 2,056 patients, 584 melanomas) [3], resized to 256x256 pixels [6].
- A confidence score for every prediction, and a rule that routes low-confidence cases to human review [5].
- A Siamese network [7] compared against a baseline CNN classifier on identical data splits.
- Training and evaluation on a GPU node of UQ's Rangpur HPC cluster, with development and small test runs on a MacBook (M5 Pro, 48 GB).

Out of scope:
- Locating lesions within wider skin photos (images are assumed to be pre-cropped).
- Using patient metadata such as age, sex or body site, although the dataset provides it [3].
- Comparing lesions within the same patient (the "ugly duckling" approach the dataset was designed to support [3]).
- Making a final diagnosis. The tool only supports a specialist's decision.

**Research question**

> Does a Siamese network, which classifies a lesion by comparing its learned embedding to labelled melanoma and benign reference images, produce confidence scores that better separate its correct predictions from its incorrect ones than a standard CNN classifier with the same pretrained backbone, and does this let it safely decide more cases automatically?

**Acceptance criteria**

#### AC1. Classification performance
- **Target:** Balanced accuracy of at least 0.80 on the held-out test split.
- **Why:** Only 1.8% of images are melanoma [3], so a model that always predicts benign scores 98% plain accuracy. Balanced accuracy averages the recall of both classes, so it cannot be gamed this way.
- **How it is tested:** The decision threshold is chosen on the validation split only, then applied once to the test split.

#### AC2. Data hygiene and leakage prevention
- **Target:** Zero patients shared between the train, validation and test splits, and zero flagged duplicate images remaining.
- **Why:** If one patient appears in both training and testing, the model can recognise the person's skin rather than the lesion, inflating test scores. The dataset authors report 425 pixel-identical duplicate images [3], and the CSV used here flags 434 rows as duplicates.
- **How it is tested:** The splitting code asserts that the patient ID sets of the three splits do not overlap, and duplicates are removed before splitting.

#### AC3. Baseline comparison
- **Target:** The Siamese network and the baseline CNN classifier are evaluated on the identical test split under the controlled conditions above.
- **Why:** A complex model only earns its extra cost if it beats a simple one under the same conditions.
- **How it is tested:** Both models report per-class precision and recall, macro-F1, AUROC and PR-AUC in one comparison table.

#### AC4. Uncertainty and safe routing to specialists
- **Target:** A confidence threshold under which the cases decided automatically miss no more than 5% of melanomas. The percentage of cases routed to human review at that threshold is reported for both models.
- **Why:** This answers the research question directly. A useful model must be less confident when it is wrong, so that its mistakes land in the human review pile rather than being decided automatically [5].
- **How it is tested:** The threshold is set on the validation split and checked on the test split. Calibration is shown with a reliability diagram and ECE for both models [4].

#### AC5. Resource limits
- **Target:** Peak GPU memory under 16 GB during training on Rangpur, and single-image prediction under 100 ms.
- **Why:** A clinic tool must run on modest hardware and respond quickly enough to fit into a review workflow. A 16 GB limit means the model would also fit on a common mid-range GPU.
- **How it is tested:** Peak GPU memory is recorded during training, and prediction time is averaged over at least 50 test images after a warm-up run, on the same Rangpur GPU for both models. Parameter counts are reported for both models.

### 2. Model Choice and Course Concepts

**Chosen project and difficulty**

Siamese network for melanoma versus benign classification on the ISIC 2020 Kaggle data, listed as **Hard** difficulty in the course specification. A baseline CNN classifier is built alongside it so the two can be compared under identical conditions.

**Dataset configuration**

| Setting | Choice | Reason |
|---|---|---|
| Source | ISIC 2020 training set, 256x256 resized JPEGs [3], [6], copied to Rangpur | The official test set has no public labels, so all evaluation uses splits made from the labelled training set. The resized copy keeps lesion detail while keeping storage and loading time manageable |
| Cleaning | Remove the 434 rows flagged as duplicates | Stops identical images appearing in both training and testing [3] |
| Split | 70% train, 15% validation, 15% test, split by patient | Every image from a patient stays in one split, so the model is tested on people it has never seen |
| Stratification | Patients with at least one melanoma (428) and without (1,628) are split separately | Each split gets a similar share of melanoma, which matters when there are only 584 positive images [3] |
| Input | Random 224x224 crop from the 256x256 image, ImageNet colour normalisation | Matches the input the pretrained encoder was trained on [9], [11] |
| Augmentation | Random flips, rotations and mild colour changes on training images only | Dermoscopic images have no fixed "up" direction, and colour varies between clinics and cameras [3] |

**Model architecture**

*Baseline CNN classifier.* An ImageNet-pretrained ResNet-18 [9] with its final layer replaced by a single output that gives the probability of melanoma. It is trained with binary cross-entropy, with melanoma cases weighted more heavily to offset the 1.8% class share. This is the standard way to approach the problem, so it is the fair point of comparison.

*Siamese network.* Two copies of the same ResNet-18 encoder with shared weights [7]. Each image is mapped to a 128-number embedding, scaled to unit length. Training uses pairs of images and contrastive loss [8]: pairs from the same class are pulled together and pairs from different classes are pushed at least a set distance apart.

- **Pair generation strategy.** The anchor image is drawn as melanoma half the time and benign half the time, and its partner is the same class half the time. This gives equal numbers of the four pair types (melanoma-melanoma, melanoma-benign, benign-benign, benign-melanoma), so the rare class is seen far more often than its natural 1.8% share. New pairs are drawn every epoch, so the 584 melanomas form many different pairs and the model is less likely to memorise them.
- **Making a prediction.** A new image is embedded and compared to a bank of labelled training images. For each class, the average distance to its 10 nearest reference images is computed. Taking the nearest neighbours of each class separately stops the large benign majority from outvoting melanoma. The two distances are turned into a melanoma probability, with a scaling factor tuned on the validation split so the confidence is well calibrated [4].
- **Why this suits the problem.** Classifying by comparison to confirmed cases mirrors how image retrieval has been used to support dermoscopic diagnosis [10]. It also gives an interpretable signal: a lesion that sits close to both melanoma and benign references is visibly ambiguous, and the nearest reference images can be shown to a specialist as evidence.

*Why both use ResNet-18.* Using the same pretrained encoder in both models means any difference in results comes from the Siamese training and comparison method, not from a stronger backbone. ResNet-18 is also small enough to train quickly on a Rangpur GPU and to test on a laptop during development.

**Course concepts applied**

| Concept | Where it is used |
|---|---|
| Convolutional neural networks and residual connections | ResNet-18 encoder in both models [9] |
| Transfer learning | Starting from ImageNet-pretrained weights [11] rather than random weights, since 584 melanomas are too few to train a deep network from scratch |
| Metric learning | Siamese network with shared weights and contrastive loss [7], [8]. Triplet loss [12] was considered and is a possible later experiment |
| Data augmentation | Flips, rotations and colour changes to reduce overfitting on the small melanoma class |
| Class imbalance | Weighted loss for the baseline, balanced pair sampling for the Siamese network, and evaluation with balanced accuracy and PR-AUC |
| Evaluation and model selection | Threshold and calibration chosen on the validation split only, with the test split used once at the end |
| Calibration and uncertainty | ECE, reliability diagrams and a reject option for routing cases to specialists [4], [5] |
| High-performance computing | Training runs submitted as batch jobs on the Rangpur GPU cluster, with short interactive sessions for debugging |

### 3. Preliminary Evidence

#### 3.1 Data Audit

Run with pandas on the downloaded dataset:

| Check | Result | What it means |
|---|---|---|
| Images | 33,126 | Matches the official count [3] |
| Patients | 2,056, median 12 images each, max 115 | Must split by patient to avoid leakage |
| Melanoma images | 584 (1.8%) | Need balanced metrics and balanced training |
| Patients with a melanoma | 428 | Enough to spread across all three splits |
| Flagged duplicates | 434 (paper reports 425 [3]) | Remove before splitting |

#### 3.2 Training Time and Resource Estimate

Measured: a 3-epoch baseline run on the full training split (23,232 images, batch size 64) took about 2 minutes per epoch (116 to 124 seconds) on the MacBook (M5 Pro, Apple GPU). The average training loss fell each epoch, from 1.07 to 0.94 to 0.88, confirming the data pipeline and training loop work end to end.

Estimated: ResNet-18 has about 11 million parameters. At batch size 64, the baseline should need roughly 3 to 5 GB of GPU memory and the Siamese network about 5 to 7 GB, since it processes two images per pair. Both fit the 16 GB limit. Memory had not been measured at this stage (see Resource profiling in the Results section for the measured values).

### 4. Risks, Budget and Fallback

#### 4.1 Risks

| Risk | Plan |
|---|---|
| Only about 400 melanomas for training | Balanced pairs, augmentation and pretrained weights |
| Confident wrong predictions | Tune confidence on validation data and check reliability diagrams |
| Small test set (about 90 melanomas) gives noisy results | Report confidence intervals by resampling the test set |
| Training time on a laptop | Quick tests on a small random subset (`--limit`) before full runs, and keep the best epoch so runs are not repeated |

#### 4.2 Budget

About 20 minutes per 10-epoch run at the measured 2 minutes per epoch. Around 15 runs across both models comes to roughly 5 hours of training on the MacBook.

#### 4.3 Next Experiment

Add validation metrics (balanced accuracy, AUROC, PR-AUC and melanoma recall) to the training loop, then run 10 epochs of the baseline and keep the epoch with the best validation AUROC.

#### 4.4 Fallback

If the Siamese network stays below 0.80 balanced accuracy, first try triplet loss [12] or harder pairs (images that look alike but have different labels). If it still falls short, report the baseline as the working model and present the Siamese results as an honest negative finding.

### References

[1] Cancer Council Australia, "Melanoma." [Link](https://connect.cancer.org.au/cancer-information/types-of-cancer/melanoma)

[2] National Cancer Institute, "Cancer Stat Facts: Melanoma of the Skin," SEER Program. [Link](https://seer.cancer.gov/statfacts/html/melan.html)

[3] V. Rotemberg et al., "A patient-centric dataset of images and metadata for identifying melanomas using clinical context," *Scientific Data*, vol. 8, 34, 2021. [Link](https://arxiv.org/abs/2008.07360)

[4] C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in *Proc. ICML*, 2017. [Link](https://arxiv.org/abs/1706.04599)

[5] Y. Geifman and R. El-Yaniv, "Selective Classification for Deep Neural Networks," in *Proc. NeurIPS*, 2017. [Link](https://arxiv.org/abs/1705.08500)

[6] C. Deotte, "JPEG Melanoma 256x256," Kaggle dataset. [Link](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256)

[7] G. Koch, R. Zemel and R. Salakhutdinov, "Siamese Neural Networks for One-shot Image Recognition," in *ICML Deep Learning Workshop*, 2015. [Link](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

[8] R. Hadsell, S. Chopra and Y. LeCun, "Dimensionality Reduction by Learning an Invariant Mapping," in *Proc. CVPR*, 2006. [Link](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

[9] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. CVPR*, 2016. [Link](https://arxiv.org/abs/1512.03385)

[10] P. Tschandl et al., "Diagnostic accuracy of content-based dermatoscopic image retrieval with deep classification features," *British Journal of Dermatology*, vol. 181, no. 1, 2019. [Link](https://doi.org/10.1111/bjd.17189)

[11] J. Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database," in *Proc. CVPR*, 2009. [Link](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf)

[12] F. Schroff, D. Kalenichenko and J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering," in *Proc. CVPR*, 2015. [Link](https://arxiv.org/abs/1503.03832)

## Changes from the Feasibility Review

**Compute.** The feasibility review planned to train on UQ's Rangpur cluster, with a MacBook used for development and small tests. After testing, all training and evaluation were done on the MacBook (M5 Pro, 48 GB, Apple GPU) instead. A full baseline epoch over 23,232 training images took about 2 minutes, so a 10-epoch run finishes in roughly 20 minutes, and peak training memory was 2.46 GB for the baseline and 4.61 GB for the Siamese network. That was fast enough to iterate without cluster queue times or copying the dataset to Rangpur. The memory target was also lowered from 16 GB to 8 GB, so the models would fit on an entry-level GPU.

**Calibration.** The feasibility review planned to turn the Siamese distances into a probability with a single scaling factor. The final version uses Platt scaling for both models instead. It fits two numbers (a scale and an offset) on the validation split, so the baseline and Siamese probabilities are calibrated in the same way and can be compared fairly [4].

**Routing.** The feasibility review described a single threshold between automatic decisions and human review. The final version uses three groups: auto-clear, specialist review and urgent review. The urgent group (10% chance of melanoma or above) was added so that the most suspicious lesions are seen first, and so that both models have a working specialist review group.

**Effect on results.** Apple and NVIDIA GPUs do floating-point maths slightly differently, so rerunning on Rangpur with the same seed may give slightly different numbers. Both models were trained and tested on the same MacBook, so the comparison between them is unaffected.

## Overview

**The problem.** Melanoma is the most dangerous common skin cancer, and it is far more treatable when it is found early [1], [2]. Skin-cancer clinics photograph many lesions per patient with a dermatoscope, and a specialist has to review every image. Most of these lesions are harmless, so much of the specialist's time goes on images that did not need review. This project builds a triage tool for that workflow. For each dermoscopic image it estimates the chance of melanoma and sends the image to one of three groups: auto-clear, specialist review or urgent review. It uses the ISIC 2020 dataset [3], where only about 1.8% of images are melanoma.

**How it works.** The main model is a Siamese network [7]. A ResNet-18 encoder [9], pretrained on ImageNet [11], turns each image into 128 numbers called an embedding. The network is trained on pairs of images with contrastive loss [8], which pulls images of the same class together and pushes images of different classes apart. To predict a new image, the network finds the 10 most similar melanomas and the 10 most similar benign lesions among the training images, and compares the average distance to each group. A lesion that sits closer to the melanomas is more likely to be melanoma. This distance gap is turned into a probability using Platt scaling fitted on the validation split [4], and the probability decides which group the image goes to. A standard ResNet-18 classifier with the same pretrained encoder is trained as the baseline, and both models are compared on the same unseen patients.

![Example predictions](figures/example_predictions.png)

*Four random test images from `predict.py`. Each row shows the test image, both models' predictions and routing decisions, and the three most similar training melanomas and benign lesions found by the Siamese network. A smaller distance means a closer match.*

**Main findings.** Both models rank lesions about equally well on the test split (AUROC 0.909 for the Siamese network and 0.899 for the baseline, a difference that is not statistically significant). The Siamese network auto-clears more images (69.5% against 63.3%), but the baseline's confidence is better at telling its right answers from its wrong ones. Both models met the 5% missed-melanoma target on validation but not on test. The full results are in the [Results](#results) section.

## Setup and Usage

### Requirements

- Python 3.14 (tested with 3.14.5)
- The packages in `requirements.txt`, with exact versions:

| Package | Version |
|---|---|
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| numpy | 2.5.4 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pillow | 12.3.0 |

The code runs on an Apple GPU (MPS) when one is available and falls back to the CPU otherwise. All results in this README were produced on a MacBook Pro (M5 Pro, 48 GB). Training on the CPU works but is much slower.

### Installation

```bash
git clone https://github.com/dhruvanil55/PatternAnalysis-2026.git
cd PatternAnalysis-2026
git checkout topic-recognition
cd recognition/Siamese_ISIC2020_s4808626
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Download the resized ISIC 2020 images, "JPEG Melanoma 256x256" by Chris Deotte, from Kaggle [6]. Unzip it so the folder looks like this:

```
~/isic2020/
    train.csv
    train/
        ISIC_0015719.jpg
        ...
```

Only `train.csv` and the `train/` folder are used. The dataset and trained models are not committed to this repository.

### Running the pipeline

Run these commands in order from this folder. Each step saves the files the next step needs.

| Step | Command | What it does | Time on the MacBook |
|---|---|---|---|
| 1 | `python dataset.py --data-dir ~/isic2020 --preview` | Removes duplicates, splits patients into train, validation and test, checks for leakage, and saves the splits to `splits/` | under a minute |
| 2 | `python train.py --data-dir ~/isic2020 --model baseline` | Trains the baseline CNN for 10 epochs and saves the best epoch to `outputs/baseline_best.pt` | about 21 minutes |
| 3 | `python train.py --data-dir ~/isic2020 --model siamese` | Trains the Siamese network for 10 epochs and saves the best epoch to `outputs/siamese_best.pt` | about 23 minutes |
| 4 | `python train.py --data-dir ~/isic2020 --test` | Tests both saved models. It fits calibration and thresholds on validation, scores the test split once, and saves the results and test figures | a few minutes |
| 5 | `python predict.py --data-dir ~/isic2020` | Prints predictions for four random test images and saves `figures/example_predictions.png` | seconds |

Step 4 runs the testing code in `evaluate.py`, so `python evaluate.py --data-dir ~/isic2020` gives exactly the same results. Steps 3 and 4 can also be combined as `python train.py --data-dir ~/isic2020 --model siamese --test`, which trains the Siamese network and then tests both models.

Useful options:

- `--limit 512` on `train.py` (including with `--test`) or `evaluate.py` runs on a small random subset to check that everything works. These quick runs save their files with a `quicktest` name, so they never overwrite a full run.
- `python predict.py --data-dir ~/isic2020 --failures` picks the most informative mistakes on the test split and saves `figures/failure_cases.png`.
- `python predict.py --data-dir ~/isic2020 --images ISIC_0572205 ISIC_3937539` predicts specific images.
- `python dataset.py --data-dir ~/isic2020 --check-pairs` checks that the Siamese training pairs are balanced.

### Example output

Running `python predict.py --data-dir ~/isic2020` prints:

```
Image         Truth       Baseline  decision            Siamese  decision           nearest mel nearest ben
ISIC_0572205  melanoma        3.0%  specialist review     12.2%  urgent review            0.120       0.114
ISIC_1729223  melanoma        2.4%  specialist review      9.1%  specialist review        0.159       0.134
ISIC_3937539  benign         35.7%  urgent review         11.0%  urgent review            0.085       0.076
ISIC_8009079  benign          0.0%  auto-clear             0.2%  auto-clear               0.057       0.030

Saved figure to figures/example_predictions.png
```

The percentages are each model's calibrated chance of melanoma. The last two columns are the distances to the closest training melanoma and the closest benign training lesion in the Siamese network's embedding.

### Files

| File | Contents |
|---|---|
| `dataset.py` | Loading the metadata, removing duplicates, patient-level splitting, image transforms, and the single-image and pair datasets |
| `modules.py` | The ResNet-18 backbone, the baseline classifier, the Siamese network, contrastive loss, and the nearest-neighbour distance calculation |
| `train.py` | Training and validation loops for both models, saving the best epoch, plotting the training curves, and the `--test` option that tests both saved models |
| `evaluate.py` | Helper with the testing code used by `train.py --test`: calibration, thresholds, test metrics, confidence intervals, resource profiling and the test figures. It is a separate file because testing needs both trained models at once |
| `predict.py` | Example predictions and failure cases, printed and shown as figures |
| `utils.py` | Shared helpers for the device, random seeds, data loader settings and plot styling |

### Reproducibility

- **Fixed splits.** Patients are assigned to splits with a fixed seed (42), and the splits are saved as CSV files in `splits/`. Every later step reads these files, so both models always use exactly the same patients.
- **Fixed training randomness.** `train.py` fixes the Python and PyTorch random seeds (42), which controls the order of the training images, the augmentations, the Siamese training pairs and the starting weights of the new final layers. The validation pair loss always uses the same 2,000 pairs, and the bootstrap confidence intervals in `evaluate.py` use a fixed seed.
- **Checked by rerunning.** The baseline was trained twice from scratch on the same MacBook with the same seed, and every epoch gave identical losses and validation scores.
- **Different hardware.** Apple, NVIDIA and CPU calculations add numbers in slightly different orders, so running on other hardware or other package versions may give slightly different numbers. The exact package versions are pinned in `requirements.txt` to limit this.
- **Choosing the saved model.** Both models always train for the full 10 epochs. After each epoch the model is scored on the validation split, and it is saved only when the validation AUROC is the best so far. This keeps the best epoch, but it is not early stopping, because training does not end early.
- **Test split used once.** Calibration, the balanced threshold and the routing thresholds are all chosen on the validation split. The test split is only scored once, by `evaluate.py`.

## Pre-processing and Data Splits

### Removing duplicates

The ISIC 2020 training set has 33,126 images. The dataset authors report 425 pixel-identical duplicate images [3], and the CSV used here flags 434 rows as duplicates (marked with `tfrecord == -1`). These rows are removed before splitting, leaving 32,692 images. If duplicates were kept, the same image could end up in both training and testing, which would make the test results look better than they really are.

### Patient-level split

Each patient has a median of 12 images, and images from the same patient often share skin tone, lighting and camera. If one patient's images were spread across splits, the model could learn to recognise the patient rather than the lesion. To prevent this, every patient is assigned to exactly one split, and `dataset.py` checks that no patient appears in more than one.

Melanoma is rare, so a random split could leave one split with very few melanomas. Patients are therefore split in two groups: the 428 patients with at least one melanoma, and the 1,628 patients with none. Each group is shuffled with a fixed seed (42) and divided 70/15/15, so every split gets a similar share of melanoma.

| Split | Images | Patients | Melanomas | Melanoma share |
|---|---|---|---|---|
| Train | 23,232 | 1,440 | 413 | 1.78% |
| Validation | 4,513 | 308 | 83 | 1.84% |
| Test | 4,947 | 308 | 85 | 1.72% |

### Why 70/15/15

- **Training (70%)** gets most of the data, because the model has only 413 melanomas to learn from.
- **Validation (15%)** is used to choose the best training epoch, fit the calibration, and set the routing thresholds. It needs enough melanomas (83) for these choices to be reliable, and it must be separate from the test split so the test result stays honest.
- **Test (15%)** is used only once, at the end. With 85 melanomas, one missed melanoma changes the miss rate by about 1.2 percentage points, so a smaller test split would make the results too noisy. Confidence intervals are reported to show this uncertainty.

### Image pre-processing

The images are the 256x256 resized version of ISIC 2020 [6], which keeps the lesion detail while making loading much faster than the full-size originals.

| Step | Training images | Validation and test images | Reason |
|---|---|---|---|
| Flips | Random horizontal and vertical flips | None | Dermoscopic images have no fixed "up" direction |
| Rotation | Random rotation of up to 30 degrees | None | Same reason as flips |
| Crop | Random 224x224 crop | Centre 224x224 crop | 224x224 is the input size ResNet-18 was pretrained on [9], and random crops add variety |
| Colour | Small random changes to brightness, contrast and saturation (up to 20%) and hue (up to 2%) | None | Colour varies between clinics and cameras [3] |
| Normalisation | ImageNet mean and standard deviation | Same | Matches the colour statistics the pretrained encoder expects [11] |

Augmentation is only applied to training images, so validation and test results are the same every time they are run.

![Sample training images](figures/sample_training_images.png)

*Sample training images after loading, saved by `python dataset.py --data-dir ~/isic2020 --preview`.*

### Handling class imbalance

Only 1.78% of training images are melanoma, so a model trained without any correction would learn to predict benign almost every time.

- **Baseline CNN.** The loss gives each melanoma image 55.3 times the weight of a benign image, which is the ratio of benign to melanoma images in the training split. With this weight, the melanoma class carries the same total weight in the loss as the benign class, even though it has about 55 times fewer images.
- **Siamese network.** Training pairs are drawn with two coin flips. The first image is melanoma or benign with equal chance, and the second image is from the same class or the other class with equal chance. This gives the four pair types (melanoma-melanoma, melanoma-benign, benign-benign and benign-melanoma) in equal numbers, so half of all images the network sees in training are melanomas. New pairs are drawn every epoch, with 11,616 pairs per epoch.

## Method Details

### Baseline CNN

The ResNet-18 backbone turns an image $x$ into 512 features, and a linear layer turns these into one score $z$. The model is trained with weighted binary cross-entropy, where $y = 1$ for melanoma and $\sigma$ is the sigmoid function:

$$
\mathcal{L}_{\text{BCE}} = -\Big[\, w \, y \log \sigma(z) + (1 - y) \log\big(1 - \sigma(z)\big) \Big], \qquad w = \frac{N_{\text{benign}}}{N_{\text{melanoma}}} = 55.3
$$

### Siamese network

Both images in a pair go through the same encoder: the ResNet-18 backbone followed by a linear layer with 128 outputs, scaled to unit length. For a pair with embeddings $e_a$ and $e_b$, the distance is $d = \lVert e_a - e_b \rVert_2$, which lies between 0 and 2. The contrastive loss [8], with $s = 1$ for a same-class pair and $s = 0$ otherwise, is

$$
\mathcal{L}_{\text{contrastive}} = s \, d^2 + (1 - s) \max(0, \, m - d)^2, \qquad m = 1.0
$$

Same-class pairs are pulled together, and different-class pairs are pushed apart until they are at least $m$ apart.

To score a new image, its embedding is compared with the embeddings of all 23,232 training images. $D_{\text{mel}}(x)$ is the average distance to its $k = 10$ nearest training melanomas, and $D_{\text{ben}}(x)$ is the average distance to its 10 nearest benign training images. The score is the gap between them:

$$
s(x) = D_{\text{ben}}(x) - D_{\text{mel}}(x)
$$

A positive score means the image is closer to the melanomas than to the benign lesions.

### Calibration

Each model's score ($z$ for the baseline, $s(x)$ for the Siamese network) is turned into a calibrated probability with Platt scaling [4]:

$$
\hat{p} = \sigma(a \cdot \text{score} + b)
$$

$a$ and $b$ are fitted by logistic regression on the validation split only, and then applied unchanged to the test split. The fitted values were $a = 0.576, \, b = -3.500$ for the baseline and $a = 11.28, \, b = -1.892$ for the Siamese network.

### Thresholds

All three thresholds are chosen on the validation split:

- **Balanced threshold** $t_{\text{bal}}$: the probability that maximises melanoma recall plus benign recall (Youden's index on the ROC curve). Used for the classification metrics.
- **Auto-clear threshold** $t_{\text{clear}}$: the highest probability below which at most 5% of validation melanomas fall. With 83 validation melanomas, this allows 4 to be cleared, so the threshold is the 5th lowest melanoma probability.
- **Urgent threshold**: a fixed probability of 0.10, or $t_{\text{clear}}$ if that is higher.

### Settings

| Setting | Value |
|---|---|
| Backbone | ResNet-18 with torchvision's ImageNet weights (IMAGENET1K_V1) [9], [11] |
| Input | 224x224 crop of the 256x256 image |
| Embedding size | 128, scaled to unit length |
| Contrastive margin $m$ | 1.0 |
| Nearest references per class $k$ | 10 |
| Melanoma loss weight $w$ (baseline) | 55.3 |
| Optimiser | AdamW, learning rate 0.0001, weight decay 0.01 (the PyTorch default) |
| Learning rate schedule | None, the learning rate stays constant |
| Batch size | 64 images (baseline) or 64 pairs (Siamese network) |
| Epochs | 10, keeping the epoch with the best validation AUROC |
| Training pairs per epoch | 11,616 (half the number of training images) |
| Validation pair loss | The same 2,000 validation pairs every epoch |
| Platt scaling | Logistic regression on validation, with almost no regularisation ($C = 10^6$) |
| Auto-clear rule | At most 5% of validation melanomas cleared |
| Urgent review | Calibrated probability of 0.10 or above |
| Confidence intervals | 1,000 bootstrap resamples of test patients |
| Random seed | 42 |

The margin, $k$ and learning rate were set before training and were not tuned. Each model was trained once with one seed.

## Results

All numbers below come from `evaluate.py`. Both models were scored on the same held-out test split of 4,947 images from 308 patients, of which 85 are melanomas. Calibration and every threshold were chosen on the validation split only, and the test split was scored once.

### Training

![Baseline training curves](figures/baseline_curves.png)
![Siamese training curves](figures/siamese_curves.png)

Both models were trained for 10 epochs with AdamW (learning rate 0.0001, batch size 64). After each epoch the model was scored on the validation split, and the epoch with the highest validation AUROC was saved. The dotted line in each figure marks that epoch.

- **Baseline CNN.** The best epoch was epoch 8, with a validation AUROC of 0.888. Validation scores moved up and down from epoch to epoch but did not show a clear decline.
- **Siamese network.** The best epoch was epoch 4, with a validation AUROC of 0.901. After that, the training pair loss kept falling (0.110 to 0.037) while the validation pair loss rose (0.198 to 0.342). This means the network started memorising the training pairs instead of learning features that carry over to new patients. Saving the best epoch protects against this.

The Siamese network's validation balanced accuracy during training stayed low (0.51 to 0.60). This is because training uses a fixed cut-off of 50%, and before calibration the Siamese network's scores lean towards benign. Melanomas are rare, so a new image usually has more benign images than melanoma images close to it. Platt scaling in `evaluate.py` corrects this, which is why the test results below are much better.

### Test performance (AC1 and AC3)

Metrics in the table use each model's balanced threshold, the cut-off that best balances melanoma and benign recall on the validation split (1.06% for the baseline and 1.38% for the Siamese network). Values in brackets are 95% confidence intervals from 1,000 bootstrap resamples of test patients.

| Metric | Baseline CNN | Siamese network |
|---|---|---|
| AUROC | 0.899 [0.866, 0.925] | 0.909 [0.877, 0.936] |
| PR-AUC | 0.223 | 0.189 |
| Accuracy | 0.648 | 0.791 |
| Balanced accuracy | 0.786 [0.749, 0.818] | 0.824 [0.780, 0.863] |
| Melanoma recall | 0.929 (79 of 85) | 0.859 (73 of 85) |
| Melanoma precision | 0.044 | 0.067 |
| Benign recall | 0.643 | 0.790 |
| Benign precision | 0.998 | 0.997 |
| Macro-F1 | 0.433 | 0.503 |
| Benign lesions flagged as melanoma | 1,735 | 1,022 |
| Melanomas missed | 6 | 12 |

![ROC and precision-recall curves](figures/test_roc_pr.png)

- **The two models rank lesions about equally well.** The Siamese network's AUROC is 0.010 higher, but the 95% interval for the difference is [-0.008, 0.028]. It includes zero, so the difference is not statistically significant.
- **They make different trade-offs.** The baseline catches more melanomas (79 against 73) but raises 713 more false alarms. The Siamese network clears more benign lesions correctly, which gives it the higher balanced accuracy.
- **The baseline has the higher PR-AUC.** Its highest scores are more often real melanomas, so it does better when only the most suspicious lesions are considered.
- **Plain accuracy is not the main metric.** The task sheet gives a target of test accuracy around 0.8 for this project. The Siamese network reaches 0.791 and the baseline 0.648. However, 98.3% of test images are benign, so a model that called every image benign would score 0.983 while missing every melanoma. Plain accuracy mostly reflects how many benign lesions are cleared, which is why balanced accuracy, which weighs melanoma and benign recall equally, is the main metric (AC1).
- **Precision is low for both.** With only 85 melanomas among 4,947 test images, even a good model flags many benign lesions for every melanoma it finds. This is expected for a screening tool, where a false alarm costs one extra review.

### Calibration and confidence (research question)

![Reliability diagrams](figures/test_reliability.png)

| Measure | Baseline CNN | Siamese network |
|---|---|---|
| ECE before calibration | 0.181 | 0.407 |
| ECE after calibration | 0.003 | 0.005 |
| AUROC of confidence for correctness | 0.772 | 0.632 |

The two measures in this table answer different questions.

**Expected calibration error (ECE)** asks whether the probabilities can be taken at face value: of the images given about 10%, are about 10% really melanoma? The test probabilities are split into 10 equal-width bins (0 to 10%, 10 to 20%, and so on), and ECE is the average gap between the mean predicted probability and the actual melanoma rate in each bin, weighted by the number of images in the bin.

**AUROC of confidence for correctness** asks whether a model knows when it is wrong. Each test image is first classified as melanoma or benign using the balanced threshold, and marked correct or incorrect. The model's confidence is how far its calibrated score is from that threshold, measured in log-odds: $c = \lvert \operatorname{logit}(\hat{p}) - \operatorname{logit}(t_{\text{bal}}) \rvert$. A prediction just above or below the threshold has low confidence, and one far from it has high confidence. The AUROC measures how often a correct prediction has higher confidence than an incorrect one, where 0.5 is no better than chance.

A model can score well on one and poorly on the other. Good calibration means the probabilities are right on average, but it does not mean the model gives its wrong answers lower confidence. Routing depends on the second property, because mistakes only reach a doctor if the model is unsure about them.

- **Calibration was essential for both models.** Before calibration, both models' probabilities were misleading, the Siamese network much more so. After Platt scaling, both have a test ECE close to zero.
- **The low ECE values flatter the models.** Because 98% of images are benign, almost all probabilities fall in the lowest bin (0 to 10%), so ECE mostly measures how well that one bin matches. Small errors in the 0 to 10% range, where both auto-clear thresholds sit, have little effect on the total. The values are also estimates from one test sample of 85 melanomas, and they may not hold at another clinic.
- **The baseline's confidence is more useful.** The baseline scores 0.772 and the Siamese network 0.632. This goes against the hypothesis in the research question: the Siamese network's distance-based confidence does not track its mistakes better than a standard classifier. No confidence interval was calculated for this gap, so it should be confirmed on more data.

### Routing to specialists (AC4)

Each test image is sent to one of three groups using the calibrated probability: auto-clear (below the auto-clear threshold), urgent review (10% or above), or specialist review (everything in between). The auto-clear threshold was set on validation so that no more than 5% of melanomas would be cleared.

| Measure | Baseline CNN | Siamese network |
|---|---|---|
| Auto-clear threshold | 1.06% | 0.49% |
| Validation melanomas auto-cleared | 4 of 83 (4.8%) | 4 of 83 (4.8%) |
| Test images auto-cleared | 63.3% [58.6, 67.5] | 69.5% [65.5, 73.0] |
| Test images sent to specialist review | 33.4% | 23.4% |
| Test images sent to urgent review | 3.3% | 7.1% |
| Test melanomas auto-cleared | 6 of 85 (7.1%) [2.3, 12.9] | 5 of 85 (5.9%) [1.3, 11.5] |
| Share of urgent cases that were melanoma | 17.8% | 15.0% |

The baseline's auto-clear threshold equals its balanced threshold (both 1.06%). The two are calculated separately. The balanced threshold always falls on the score of a validation melanoma, because that is where melanoma recall changes, and here it happened to land on the same melanoma as the auto-clear rule (the 5th lowest of 83). This is why the baseline's balanced-threshold results and its auto-clear results both show 6 missed test melanomas. For the Siamese network the two thresholds differ (1.38% and 0.49%).

![Automation and missed melanoma trade-off](figures/test_automation.png)

*Risk-coverage curve for routing. Each point is a possible auto-clear threshold, showing the share of test images that would be cleared (coverage) against the share of test melanomas that would be missed (risk).*

- **The 5% target was met on validation but not on test.** Both models auto-cleared 4.8% of validation melanomas, inside the target. On the test split they cleared 7.1% and 5.9%. With only 83 to 85 melanomas in each split, one or two extra misses move the rate by more than a percentage point, and both confidence intervals include 5%. The threshold needs a safety margin, which is covered in the recommendation.
- **The Siamese network automates more of the workload.** It auto-cleared 69.5% of test images against 63.3% for the baseline, while missing one fewer melanoma. The confidence intervals overlap, so this advantage is not certain.
- **The urgent group is much richer in melanoma than the whole test set.** About 1 in 6 urgent cases was melanoma, compared with 1 in 58 overall.

### Resource profiling (AC5)

| Measure | Baseline CNN | Siamese network |
|---|---|---|
| Parameters | 11,177,025 | 11,242,176 |
| Peak GPU memory during training (batch of 64) | 2.46 GB | 4.61 GB |
| Prediction time per image | 4.1 ms | 5.0 ms |
| Training time per epoch | about 2.1 minutes | about 2.3 minutes |
| Extra storage needed at prediction time | none | 11.9 MB of reference embeddings |

All measurements were taken on a MacBook Pro (M5 Pro, 48 GB) using the Apple GPU. Peak memory is the memory the GPU driver held over five training steps. Prediction time is the average over 50 test images, one at a time, after 10 warm-up runs. For the Siamese network it includes comparing the new embedding with all 23,232 stored training embeddings.

- Both models have almost the same number of parameters, since they share the ResNet-18 backbone. The Siamese network only swaps the final layer for a 128-number embedding layer.
- The Siamese network uses about twice the training memory, because every training step processes two images per pair.
- Both models are far inside the AC5 limits of 8 GB of memory and 100 ms per image.

### Acceptance criteria summary

| Criterion | Target | Baseline CNN | Siamese network |
|---|---|---|---|
| AC1. Classification | Balanced accuracy of at least 0.80 on test | Not met (0.786) | Met (0.824), although the lower end of its interval is 0.780 |
| AC2. Data hygiene | No shared patients, no duplicates | Met | Met |
| AC3. Baseline comparison | Both models on the same test split | Met | Met |
| AC4. Safe routing | At most 5% of melanomas auto-cleared | Met on validation (4.8%), not on test (7.1%) | Met on validation (4.8%), not on test (5.9%) |
| AC5. Resources | Under 8 GB memory and 100 ms per image | Met (2.46 GB, 4.1 ms) | Met (4.61 GB, 5.0 ms) |

## Failure Analysis

![Failure cases](figures/failure_cases.png)

*Five test images chosen by `python predict.py --data-dir ~/isic2020 --failures`. Three are melanomas (rows 1, 2 and 5) and two are benign (rows 3 and 4).*

### How the cases were chosen

The `--failures` option searches the 4,947 test images for the most informative mistakes:

- Row 1: the melanoma the Siamese network was most confident was benign
- Row 2: the melanoma the baseline CNN was most confident was benign
- Row 3: the benign lesion the Siamese network was most confident was melanoma
- Row 4: the benign lesion the baseline CNN was most confident was melanoma
- Row 5: the melanoma where the two models gave the most different answers

### How to read the figure

Each row is one test image. The first column is the test image itself, with its true diagnosis above it and both models' predictions below it. The next three columns are the three training melanomas that the Siamese network considers most similar to it. The last three columns are the three most similar benign training lesions.

The number above each training image is its distance from the test image in the Siamese network's embedding. A smaller distance means the network sees the two images as more alike. Distances in this project usually fall between about 0.03 and 0.25.

### How the percentages were calculated

Each percentage is the model's estimated chance that the lesion is melanoma.

- **Baseline CNN.** The ResNet-18 classifier gives each image a single score. That score is converted into a probability using Platt scaling, a small correction fitted on the validation split so that, for example, images given 10% really are melanoma about 10% of the time.
- **Siamese network.** The network turns the test image into an embedding and finds its 10 nearest melanomas and 10 nearest benign lesions among the 23,232 training images. It works out the average distance to each group. If the image is closer on average to the melanomas, it is more likely to be melanoma. The difference between the two averages is then converted into a probability with Platt scaling, in the same way as the baseline. The figure only shows the closest three of each group, so it gives a good idea of the decision but not the full calculation.

### How each case was routed

Each probability is compared with thresholds chosen on the validation split:

| Decision | Baseline CNN | Siamese network |
|---|---|---|
| Auto-clear | below 1.06% | below 0.49% |
| Specialist review | 1.06% to under 10% | 0.49% to under 10% |
| Urgent review | 10% or above | 10% or above |

The auto-clear thresholds were set so that no more than 5% of validation melanomas would be cleared (AC4). They are very low because melanoma is rare, making up only about 1.8% of images in every split.

| Row | Image | True diagnosis | Baseline | Siamese | Outcome |
|---|---|---|---|---|---|
| 1 | ISIC_5365617 | Melanoma | 0.6%, auto-clear | 0.2%, auto-clear | Missed by both models |
| 2 | ISIC_3093026 | Melanoma | 0.1%, auto-clear | 0.2%, auto-clear | Missed by both models |
| 3 | ISIC_5128348 | Benign | 4.1%, specialist review | 14.2%, urgent review | False alarm |
| 4 | ISIC_6704518 | Benign | 55.1%, urgent review | 13.6%, urgent review | False alarm |
| 5 | ISIC_7800750 | Melanoma | 58.3%, urgent review | 12.8%, urgent review | Caught by both models |

### What each case shows

**Row 1 (melanoma, missed).** This melanoma is faint, light brown and has a fine net-like pattern, which is typical of a harmless mole. Its three nearest benign images are very close (0.053 to 0.059). Its nearest melanoma is further away (0.089), and the next two are much further (0.215 and 0.242). Because it sits among benign lesions in the Siamese embedding, the Siamese network gave it 0.2%. The baseline also gave it well under 1% (0.6%), which suggests it looks benign to that model too, although the Siamese neighbours cannot show why the baseline made its decision. Both models auto-cleared it. In real use this patient would not have been seen by a doctor.

**Row 2 (melanoma, missed).** This melanoma is orange-brown with a sharp edge, on pink skin with a scaly surround. Again the nearest benign images are much closer (about 0.05) than the nearest melanomas (0.066, 0.190 and 0.236). Its three nearest melanomas are the exact same three images as in row 1. This means very few training melanomas exist in this area of the embedding. When a melanoma does not look like the usual examples, there is nothing similar to match it to, so it is treated as benign.

**Row 3 (benign, false alarm).** This harmless mole is large and very dark, which are features often seen in melanoma. Its nearest images in both groups are also dark, heavily pigmented lesions at similar distances (0.066 to 0.087). Because the evidence is mixed, the Siamese network gave 14.2% and sent it for urgent review. The baseline gave 4.1%, which still sent it for specialist review. A doctor would likely want to check this lesion too, so this is an acceptable mistake for a triage tool.

**Row 4 (benign, false alarm).** The baseline gave this harmless lesion 55.1%, its highest score for any benign test image, and the Siamese network gave 13.6%. Both sent it for urgent review. The Siamese neighbours cannot explain the baseline's score, because the two are different models. What they do show is that all six training images the Siamese network finds most similar, melanoma and benign, share the same round black dermoscope frame, ruler markings and pale, washed-out colouring. This suggests that the Siamese embedding groups images partly by camera setup rather than by the lesion alone, a form of shortcut learning where a model picks up on features that have nothing to do with the diagnosis. One example cannot prove this. Testing it would need follow-up experiments that were not done here, such as checking whether other false alarms share these artefacts, or covering the frame and rulers and seeing whether the predictions change. For the baseline, a heatmap method such as Grad-CAM would show which parts of the image drove its score.

**Row 5 (melanoma, caught).** This melanoma fills most of the frame and has low contrast against the skin. The baseline gave 58.3% and the Siamese network gave 12.8%, the largest gap between the two models for any melanoma. All six distances are almost the same (0.067 to 0.079), so the Siamese network could not tell which group the image was closer to, and its probability stayed low. Even so, both results were above 10%, so both models sent it for urgent review. Ruler markings also appear in several of its neighbours.

### Patterns across the cases

1. **Unusual melanomas are the main risk.** With only 413 melanomas among the 23,232 training images, the models have not seen every way a melanoma can look. Rows 1 and 2 show what happens when a melanoma looks more like a typical mole. These are the cases behind the auto-clear tier missing 7.1% (baseline) and 5.9% (Siamese) of test melanomas, above the 5% target in AC4.
2. **Image artefacts may influence the results.** In rows 4 and 5, dermoscope frames, ruler markings and colouring appear to affect which images the Siamese network sees as similar. This is an observation from a few cases, not a tested finding, but it means a training image shown as "similar" may be similar for the wrong reason.
3. **Some lesions are hard to classify.** Large dark moles and low-contrast melanomas sit between the two groups (rows 3 and 5). Sending these to a doctor is the correct result for a triage tool.
4. **Routing caught most errors.** Rows 3, 4 and 5 were all sent for review, so a doctor would have seen them. Only the auto-clear tier can cause harm, and rows 1 and 2 are examples of that harm.

### Possible improvements

- Add more melanoma examples, for example from earlier ISIC challenge datasets, so that less common types are represented.
- Remove or cover ruler markings, ink and dermoscope frames before training, to reduce shortcut learning.
- Choose the auto-clear threshold with a safety margin on validation, for example allowing 3% of melanomas to be cleared instead of 5%, so that the test result is more likely to stay under the 5% target.

## Limitations

These limitations affect how far the results can be trusted. None of them were tested with extra experiments in this project.

| Limitation | What it means for the conclusions |
|---|---|
| **Few melanomas.** The test split has 85 melanomas and the validation split 83. | One extra missed melanoma changes a miss rate by about 1.2 percentage points, so the confidence intervals are wide. The difference between the models is not statistically significant for most measures. |
| **Thresholds chosen on a small validation split.** | The 5% miss target was met on validation but not on test. AC4 has therefore not been shown to hold on new patients. |
| **One dataset.** All images come from ISIC 2020. | Performance at other clinics, with other cameras or with other patient groups is unknown. |
| **Calibration checked on one test sample.** | The low ECE values do not guarantee reliable probabilities elsewhere, and ECE mostly reflects the lowest probability bin (see Results). |
| **Possible shortcut learning.** | The failure analysis suggests image artefacts influence the Siamese embedding, but this was not tested. Predictions and nearest-image explanations may partly reflect how a photo was taken. |
| **One training run per model, no tuning.** | Each model was trained once with one seed, and the margin, $k$ and learning rate were not tuned. Results could change with other seeds or settings. |
| **Extra cost of the Siamese network.** | It needs about twice the training memory and must store and update embeddings for every training image. |
| **Images only.** | Patient age, sex and body site were not used, although they could help. |
| **Clinical safety.** | Both models missed more than 5% of test melanomas in the auto-clear group. Neither is safe to clear patients without review. |

## Recommendation

**Summary for the project manager.** There are three separate decisions, and they have different answers:

| Decision | Recommendation |
|---|---|
| **Model selection.** Which model should be developed further? | The baseline CNN |
| **Operational policy.** Which images, if any, can be handled automatically? | None yet. During a trial, the tool should only sort the review queue, with urgent cases first, while every image is still reviewed. A stricter auto-clear threshold should be tested separately. |
| **Deployment readiness.** Is the tool ready for clinical use? | No. A shadow trial and testing on new patients are needed first. |

### Model selection

The baseline CNN is recommended over the Siamese network.

- **Both models rank lesions equally well.** The AUROC difference is 0.010, with a 95% interval of -0.008 to 0.028, so neither model is clearly better at telling melanoma from benign.
- **The baseline's confidence is more trustworthy.** Routing only works if a model is less confident when it is wrong, so that its mistakes go to a doctor. The baseline scores 0.772 on this measure and the Siamese network 0.632. This answers the research question: the Siamese network does not produce better confidence scores than a standard classifier.
- **The baseline is cheaper and simpler.** The Siamese network needs about twice the training memory (4.61 GB against 2.46 GB), is slightly slower per image, and must keep embeddings for all 23,232 training images. These have to be rebuilt every time the model is retrained.
- **The Siamese network's advantages are not certain.** It auto-cleared more images (69.5% against 63.3%), missed one fewer melanoma on test, and was the only model to meet AC1. However, the confidence intervals for both models overlap, so these gains may not hold on new patients.

The Siamese network's nearest training images are still useful for explaining a prediction to a specialist. They should be shown with care, because the failure analysis found that some matches are based on rulers and dermoscope frames rather than the lesion.

### Operational policy

| Group | Rule | Who sees it |
|---|---|---|
| Auto-clear | Below a threshold set on validation so that at most 3% of melanomas are cleared | No one, once the trial has shown it is safe |
| Specialist review | Between the auto-clear threshold and 10% | A specialist, in the normal queue |
| Urgent review | 10% chance of melanoma or above | A specialist, first |

During the shadow trial, images in the auto-clear group are still reviewed by a specialist. Auto-clearing only starts if the trial shows the threshold is safe.

The 5% limit used in this project was met on validation but not on test (7.1% for the baseline). A 3% limit on validation leaves a safety margin for this difference. The 3% figure was chosen after seeing the test result, so it is a proposal for the next experiment, not a tested solution. Its effect needs to be measured by changing `MAX_MISSED_MELANOMA` in `evaluate.py` and checking it on new data. A stricter threshold will auto-clear fewer images.

### Costs and benefits

For every 1,000 images, using the test results of the baseline with the current 5% rule:

| Outcome | Number of images |
|---|---|
| Melanomas among the 1,000 images | about 17 |
| Auto-cleared, with no specialist review | 633 |
| Sent to specialist review | 334 |
| Sent to urgent review | 33 |
| Melanomas wrongly auto-cleared | about 1.2 |

- **Benefit.** Specialists would review about 37% of images instead of all of them. The urgent group is also far richer in melanoma than the full set (about 1 in 6 compared with 1 in 58), so the most likely melanomas are seen first.
- **Cost.** About one melanoma in every 1,000 images would be cleared without review. The upper end of the confidence interval is 12.9% of melanomas, about 2.2 per 1,000 images. A missed melanoma can be fatal, so this rate is too high to accept without the trial and the stricter threshold.

### Deployment readiness

The tool is not ready for clinical use. Before it is:

1. **Run a shadow trial.** Run the tool alongside specialists without clearing anything automatically, and compare its decisions with theirs.
2. **Use more melanoma examples.** Adding images from earlier ISIC challenge datasets would help with unusual melanomas like rows 1 and 2 in the failure analysis, and would make the threshold more stable.
3. **Remove image artefacts.** Covering rulers, ink and dermoscope frames before training would test, and may reduce, the possible shortcut learning seen in the failure analysis.
4. **Recheck calibration at each new clinic.** Different cameras and lighting can shift the probabilities, so calibration and thresholds should be checked on each clinic's own images before use.


## Related Implementations and Documentation

| Source | What it shows | Related part of this project |
|---|---|---|
| [PyTorch examples: Siamese network](https://github.com/pytorch/examples/tree/main/siamese_network) | The official PyTorch example of a Siamese network, with one shared encoder applied to both images of a pair | `SiameseNetwork` in `modules.py` and the pair training loop in `train.py` |
| [adambielski/siamese-triplet](https://github.com/adambielski/siamese-triplet) | PyTorch Siamese and triplet networks with an embedding network, contrastive loss and pair selection | `EmbeddingNet` and `ContrastiveLoss` in `modules.py`, balanced pairs in `dataset.py`, triplet loss as the planned fallback |
| [SIIM-ISIC 2020 1st place solution](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution) | The winning Kaggle solution for this dataset, trained on the resized ISIC 2020 images | Data handling and augmentation choices in `dataset.py` |
| [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | Fine-tuning an ImageNet-pretrained ResNet-18 on a new task | `make_backbone` and `BaselineClassifier` in `modules.py` |
| [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | Platt (sigmoid) scaling for turning model scores into calibrated probabilities | `fit_calibration` in `evaluate.py` |
