# Batik Motif Classification: YOLOv11 vs MobileNetV3

30 Indonesian batik motifs, two models compared with the same dataset and training split to see performance differences.

It started as a group final project for a deep learning course at President University in late 2025, which I led. The question was narrower than it sounds. Not whether a CNN can classify batik, it can, but whether an architecture built for detection beats one built for phones when the job is telling apart patterns that were designed to look related.

| | YOLOv11-Medium | MobileNetV3-Large |
|---|---|---|
| Test accuracy | **83.1%** | 78.9% |
| Weighted F1 | **0.829** | 0.785 |
| Macro F1 | **0.734** | 0.695 |
| Training time | **6.6 min** | 12.6 min |

237 held-out images and 30 motifs running in a Tesla T4 using Google Colab.

Macro F1 is the number to read. Accuracy flatters whichever model happens to be good at the three biggest classes, and this dataset is lopsided enough for that to matter. Macro F1 weights every motif the same.

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/6e9e5fe0-31a5-4172-953c-d76c966dc4de" />

MobileNetV3 ends at 98% on training against 79.5% on validation. It memorised. Ultralytics doesn't log training accuracy for classification runs, so there's no equivalent gap to check on the YOLOv11 side.

<img width="2313" height="989" alt="image" src="https://github.com/user-attachments/assets/4bc04582-096d-4f05-9a8d-1de009cf7f4f" />

Row-normalised, so each row is recall for that motif rather than a reward for having the most test images.

## What doesn't work

83% hides a split result. Ten motifs score 0.94 F1 or better and four of those are perfect. Sidomukti, Sogan and Tambal score zero, and Priangan and Keraton are barely above it.

The three at zero are Central Javanese, built from the same vocabulary of shapes and fill densities, and each has around 30 training images. Two problems that make each other worse. More epochs fixes neither.

## The data

[[muhammadsalmanalfaridzi/Batik-Indonesia](https://huggingface.co/datasets/muhammadsalmanalfaridzi/Batik-Indonesia)](https://huggingface.co/datasets/muhammadsalmanalfaridzi/Batik-Indonesia) on Hugging Face, plus extra images pulled in to fill out the thinner classes, then curated in Roboflow. Anything under 30 images was dropped, because below that the test split gets too thin for a per-class number to mean anything. What's left is 30 classes and 2,330 images, split 70/20/10.

<img width="1990" height="790" alt="image" src="https://github.com/user-attachments/assets/ea8e761d-7f21-4e9a-9764-271250968bc2" />

## The setup

Both models got 224×224 input, batch 32, 30 epochs with early stopping at patience 10, and augmentation matched as closely as the two frameworks allow. YOLOv11 fine-tunes fully; MobileNetV3 has its last 30 backbone layers unfrozen with a GAP → Dropout → Dense head on top.

Keras has no random erasing built in, so it's written by hand in the notebook to match YOLO's `erasing=0.2`. Otherwise one model gets regularisation the other doesn't and the comparison stops meaning anything. Vertical flip is off for both — batik has a top and a bottom, and an upside-down motif isn't an image that exists.

## Running it

`notebooks/batik_yolov11_vs_mobilenetv3.ipynb` in Colab on a T4, top to bottom, about 20 minutes. The dataset pulls from Roboflow at runtime, so put your key in Colab Secrets as `ROBOFLOW_API_KEY`.

The paper in `Batik_Classification_Report.pdf` is from the original December 2025 run and its numbers sit similarly although still different from the ones above. For more details, read the report.
