import pathlib
import sys
import torchvision.transforms.functional as TF

# Fix cross-platform WindowsPath loading on Linux
if sys.platform != 'win32':
    pathlib.WindowsPath = pathlib.PosixPath

import fastai.vision.augment
fastai.vision.augment.tvpad = TF.pad

from fastai.vision.all import *
import gradio as gr

learn = load_learner('eye_disease_model.pkl')

def predict_eye_disease(img):
    if img is None:
        return None
    pred, pred_idx, probs = learn.predict(img)
    return {c: float(p) for c, p in zip(learn.dls.vocab, probs)}

interface = gr.Interface(
    fn=predict_eye_disease,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="Eye Disease Detection - By Ali Akbar Khan",
    description="Upload an eye image to detect eye diseases like Cataract, Glaucoma, Diabetic Retinopathy, etc."
)

if __name__ == "__main__":
    interface.launch()
