import os
import io
import timm 
import torch.nn as nn
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import torch
import cv2
from torchvision import transforms
from ultralytics import YOLO


app = FastAPI()

TEXT_DET_MODEL_PATH = "runs/detect/train5/weights/best.pt"
det_model = YOLO(TEXT_DET_MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize((100, 420)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz-"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(sorted(CHARS))}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

def decode(encoded_sequences, idx_to_char, blank_char="-"):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None  

        for token in seq:
            if token != 0:  
                char = idx_to_char[token.item()]
                
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char  

        decoded_sequences.append("".join(decoded_label))

    return decoded_sequences

class CRNN(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)


        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)  
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  

        return x

HIDDEN_SIZE = 256
N_LAYERS = 3
DROPOUT_PROB = 0.2
UNFREEZE_LAYERS = 3

reg_model = CRNN(vocab_size=37, hidden_size=256, n_layers=3, dropout=0.2, unfreeze_layers=3)
#reg_model.load_state_dict(torch.load("ocr_crnn.pt", map_location=torch.device("cpu")))

reg_model.load_state_dict(torch.load("ocr_crnn.pt"))
reg_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reg_model.to(device)

# Update /ocr endpoint
@app.post("/ocr")
async def process_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    image_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name

    result = det_model(temp_file_path, verbose=False)[0]
    bboxes = result.boxes.xyxy.tolist()
    img = Image.open(temp_file_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    predictions = []

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        crop = img.crop((x1, y1, x2, y2))
        input_tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = reg_model(input_tensor).cpu()
        pred = decode(logits.permute(1, 0, 2).argmax(2), IDX_TO_CHAR)[0]
        predictions.append(pred)


        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, pred, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    os.remove(temp_file_path)


    _, buffer = cv2.imencode(".png", img_cv)
    img_bytes = buffer.tobytes()


    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/png",
        headers={"X-Text": "|".join(predictions)}
    )

@app.get("/")
def home():
    return {"message": "OCR API is running"}
