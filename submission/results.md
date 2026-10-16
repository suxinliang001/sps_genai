# Assignment 2: Implementation and Results

## Completed requirements

- Implemented the exact specified PyTorch CNN in `app/cnn.py`.
- Trained using the complete CIFAR-10 training set and evaluated on its held-out test set.
- Added multipart image upload and ten-class inference at `POST /predict`.
- Preserved the earlier bigram and spaCy endpoints.
- Included trained weights, locked dependencies, Docker setup, tests, and all six theory answers.

## Training and evaluation

| Setting | Submitted run |
|---|---|
| Training examples | 50,000 |
| Held-out test examples | 10,000 |
| Epochs | 10 |
| Batch size | 128 |
| Optimizer | Adam, learning rate 0.001 |
| Loss | CrossEntropyLoss |
| Random seed | 42 |
| Device | Apple MPS |
| Trainable parameters | 825,398 |
| Final training accuracy | 87.31% |
| Held-out test accuracy | **65.58%** |

The checkpoint is the final epoch, not a checkpoint chosen using test accuracy. No test images were used for training. Both splits use RGB images resized to 64 x 64 and normalized by channel mean/std 0.5. The raw metrics and complete epoch history are in `models/metrics.json`.

## Interpretation

Accuracy is substantially above the 10% balanced random-guess baseline. The gap between training accuracy (87.31%) and test accuracy (65.58%) indicates overfitting. This run deliberately preserves the assignment architecture and uses no data augmentation or additional layers. A separate validation split, training-only augmentation, and regularization could be explored in future experiments without tuning against the test set.

| Class | Test accuracy |
|---|---|
| airplane | 67.5% |
| automobile | 80.5% |
| bird | 53.0% |
| cat | 36.4% |
| deer | 69.0% |
| dog | 57.6% |
| frog | 70.7% |
| horse | 73.9% |
| ship | 80.1% |
| truck | 67.1% |

Cat recognition is the weakest class in this evaluation. Confidence scores should not be interpreted as calibrated probabilities of correctness, especially on images unlike CIFAR-10.

## Example inference

- Input: `submission/example.png`, the first held-out CIFAR-10 test image.
- Ground truth: cat.
- API prediction: cat (class 3).
- Softmax confidence: 96.37%.
- Full response: `submission/example_response.json`.

```bash
curl -X POST http://127.0.0.1:8001/predict -F 'file=@submission/example.png'
```

## Verification

- All six automated tests passed locally.
- Docker image built successfully with locked CPU-only dependencies.
- A live multipart request to the Docker API returned the correct cat prediction; the full response is saved in `submission/docker_example_response.json`.
- The running Docker OpenAPI schema includes the new endpoint and all original endpoints.
- Exact intermediate shapes and total parameter count verified.
- RGB, grayscale, and RGBA inputs accepted and converted consistently.
- Invalid, missing, and oversize uploads checked.
- Earlier health, text generation, and embedding endpoints checked.
- Theory answer PDF rendered and visually inspected (two pages).

## Dataset provenance

The original CIFAR-10 server timed out during preparation. The unchanged archive was downloaded from the MIT Open Learning AI+D mirror (https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python), then verified against torchvision's expected MD5: `c58f30108f718f92721af3b95e74349a`. Torchvision also verified the extracted dataset files. Training code uses the normal torchvision CIFAR10 loader; the dataset is excluded from the repository.
