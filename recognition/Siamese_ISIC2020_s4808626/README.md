# Siamese Network for Melanoma Classification (ISIC 2020)

## Feasibility Review

### 1. User Need, Scope and Acceptance Criteria

**User need**

The intended user is a skin-cancer screening clinic, where staff photograph each patient's lesions with a dermatoscope and a specialist reviews the images. Patients bring many lesions (a median of 12 per patient in ISIC 2020 [3]), so specialists cannot review every image promptly. Review delays matter: melanoma can spread to other parts of the body if it is not found early [1], and five-year relative survival falls from 100% when it is localised to 34% once it has spread to distant organs [2].

The clinic needs a triage assistant that:

1. **Separates melanoma from benign lesions reliably**, so suspicious lesions reach a specialist sooner and clearly benign ones do not use up review time (tested by AC1).
2. **Rarely misses a melanoma.** A missed melanoma can be fatal, while a false alarm costs only one extra specialist review [2], so the tool must favour catching melanoma over avoiding false alarms (AC4).
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
- **Why:** Only 1.8% of images are melanoma [3], so a model that always predicts benign scores 98% plain accuracy. Balanced accuracy averages the recall of both classes, so it cannot be gamed this way [10].
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
- **Why this suits the problem.** Classifying by comparison to confirmed cases mirrors how image retrieval has been used to support dermoscopic diagnosis [12]. It also gives an interpretable signal: a lesion that sits close to both melanoma and benign references is visibly ambiguous, and the nearest reference images can be shown to a specialist as evidence.

*Why both use ResNet-18.* Using the same pretrained encoder in both models means any difference in results comes from the Siamese training and comparison method, not from a stronger backbone. ResNet-18 is also small enough to train quickly on a Rangpur GPU and to test on a laptop during development.

**Course concepts applied**

| Concept | Where it is used |
|---|---|
| Convolutional neural networks and residual connections | ResNet-18 encoder in both models [9] |
| Transfer learning | Starting from ImageNet-pretrained weights [11] rather than random weights, since 584 melanomas are too few to train a deep network from scratch |
| Metric learning | Siamese network with shared weights and contrastive loss [7], [8]. Triplet loss [13] was considered and is a possible later experiment |
| Data augmentation | Flips, rotations and colour changes to reduce overfitting on the small melanoma class |
| Class imbalance | Weighted loss for the baseline, balanced pair sampling for the Siamese network, and evaluation with balanced accuracy and PR-AUC [10] |
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

Estimated: ResNet-18 has about 11 million parameters. At batch size 64, the baseline should need roughly 3 to 5 GB of GPU memory and the Siamese network about 5 to 7 GB, since it processes two images per pair. Both fit the 8 GB limit. Memory is still to be measured.

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

**Compute.** The feasibility review planned to train on UQ's Rangpur cluster, with a MacBook used for development and small tests. After testing, all training and evaluation were done on the MacBook (M5 Pro, 48 GB, Apple GPU) instead. A full baseline epoch over 23,232 training images took about 2 minutes, so a 10-epoch run finishes in roughly 20 minutes, and peak memory stayed under [to be measured] GB. That was fast enough to iterate without cluster queue times or copying the dataset to Rangpur. The memory target was also lowered from 16 GB to 8 GB, so the models would fit on an entry-level GPU.

**Effect on results.** Apple and NVIDIA GPUs do floating-point maths slightly differently, so rerunning on Rangpur with the same seed may give slightly different numbers. Both models were trained and tested on the same MacBook, so the comparison between them is unaffected.

## Related Implementations and Documentation
| Source | What it shows | Related part of this project |
|---|---|---|
| [PyTorch examples: Siamese network](https://github.com/pytorch/examples/tree/main/siamese_network) | The official PyTorch example of a Siamese network, with one shared encoder applied to both images of a pair | `SiameseNetwork` in `modules.py` and the pair training loop in `train.py` |
| [adambielski/siamese-triplet](https://github.com/adambielski/siamese-triplet) | PyTorch Siamese and triplet networks with an embedding network, contrastive loss and pair selection | `EmbeddingNet` and `ContrastiveLoss` in `modules.py`, balanced pairs in `dataset.py`, triplet loss as the planned fallback |
| [SIIM-ISIC 2020 1st place solution](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution) | The winning Kaggle solution for this dataset, trained on the resized ISIC 2020 images | Data handling and augmentation choices in `dataset.py` |
| [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | Fine-tuning an ImageNet-pretrained ResNet-18 on a new task | `make_backbone` and `BaselineClassifier` in `modules.py` |
| [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | Platt (sigmoid) scaling for turning model scores into calibrated probabilities | `fit_calibration` in `evaluate.py` |
