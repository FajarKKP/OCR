import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device used: ", device)

model_name = "microsoft/trocr-base-printed"

processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

# Sample image (you can replace with your own later)
url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

with torch.no_grad():
    generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Predicted text:", generated_text)
