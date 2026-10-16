import spacy
import io
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.cnn import CIFAR10CNN, CLASSES, image_transform

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "cifar10_cnn.pt"
classifier = None
if MODEL_PATH.exists():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classifier = CIFAR10CNN()
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
preprocess = image_transform()

# Load the spaCy model once at startup (medium model has real word vectors)
nlp = spacy.load("en_core_web_md")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
    It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    doc = nlp(request.word)
    if not doc.has_vector:
        raise HTTPException(status_code=404, detail="No vector found for this word")
    return {
        "word": request.word,
        "dimension": len(doc.vector),
        "embedding": doc.vector.tolist(),
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Classify an uploaded image into one of the ten CIFAR-10 classes."""
    if classifier is None:
        raise HTTPException(503, "Trained model missing. Run python -m scripts.train and restart the API.")
    content = file.file.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image must be at most 10 MiB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 20_000_000:
                raise HTTPException(413, "Image must contain at most 20 million pixels")
            inputs = preprocess(image.convert("RGB")).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Upload a valid image, such as PNG or JPEG")
    with torch.inference_mode():
        probabilities = torch.softmax(classifier(inputs), dim=1)[0]
    index = int(probabilities.argmax())
    return {"class_id": index, "class_name": CLASSES[index],
            "confidence": float(probabilities[index]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)}}
