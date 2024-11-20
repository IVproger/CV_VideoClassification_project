import gradio as gr
import os
import random
import cv2
import torch
import numpy as np


from peft import PeftModel, PeftConfig
from transformers import (
    pipeline,
    AutoModelForVideoClassification,
    AutoImageProcessor,
    VideoMAEImageProcessor,
)

USE_LORA_MODEL = True

BASE_MODEL = "MCG-NJU/videomae-base"
LORA_MODEL = "model"

model = AutoModelForVideoClassification.from_pretrained(
    BASE_MODEL, ignore_mismatched_sizes=True
)
if USE_LORA_MODEL:
    config = PeftConfig.from_pretrained(LORA_MODEL, inference_mode=True)
    model = PeftModel.from_pretrained(
        model,
        LORA_MODEL,
        config=config,
        is_trainable=False,
        ignore_mismatched_sizes=True,
    )
image_processor = VideoMAEImageProcessor.from_pretrained(LORA_MODEL)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline(
    "video-classification", model=model, image_processor=image_processor, device=device
)

# Load predefined videos
PREDEFINED_VIDEOS_DIR = "videos"
predefined_videos = {
    video: os.path.join(PREDEFINED_VIDEOS_DIR, video)
    for video in os.listdir(PREDEFINED_VIDEOS_DIR)
    if video.endswith((".mp4", ".avi", ".mkv"))
}
# null value for this dropdown
NO_PREDEFINED_VIDEO_SELECTED = "No predefined video selected"
predefined_videos[NO_PREDEFINED_VIDEO_SELECTED] = ""

# model
PREDEFINED_MODELS_DIR = "models"
predefined_models = {
    model_dir: os.path.join(PREDEFINED_MODELS_DIR, model_dir)
    for model_dir in os.listdir(PREDEFINED_MODELS_DIR)
}
print(predefined_models)


def on_predefined_video_select(predefined_video: str):
    path = predefined_videos.get(predefined_video)
    # if its "" we don't need to propagate path to video
    if path:
        return path


# https://huggingface.co/docs/transformers/en/tasks/video_classification#inference
def on_inference_button_click(video: str):
    if video:
        res = pipe(video)

        print(pipe.model.config.id2label)

        label = pipe.model.config.id2label[res[0]["label"]]
        score = res[0]["score"]

        return f"{label} {score * 100:.2f}%"

    # inputs = {
    #     "pixel_values": video_tensor.unsqueeze(0),
    #     # "labels": torch.tensor(
    #     #     [sample_test_video["label"]]
    #     # ),  # this can be skipped if you don't have labels available.
    # }

    # inputs = {k: v.to(device) for k, v in inputs.items()}
    # model = model.to(device)

    # # forward pass
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     print(outputs)

    #     logits = outputs.logits

    return video


def on_video_upload(video: str):
    # return video on upload and reset predefined video
    return video, gr.update(
        elem_id="predefined_video", value=NO_PREDEFINED_VIDEO_SELECTED
    )


with gr.Blocks() as demo:
    gr.Markdown("## Video Classification App")

    with gr.Row(equal_height=True, min_height=500):

        with gr.Column():
            predefined_video = gr.Dropdown(
                choices=list(predefined_videos.keys()),
                label="Select Predefined Video",
                value=NO_PREDEFINED_VIDEO_SELECTED,
                elem_id="predefined_video",
            )
            chosen_model = gr.Dropdown(
                choices=["MCG-NJU/videomae-base", "MCG-NJU/videomae-large"],
                label="Select Model",
                value="MCG-NJU/videomae-base",
            )
            output_label = gr.Textbox(label="Predicted label: ", interactive=False)
            inference_button = gr.Button("Classify")

        with gr.Column(min_width=1000):
            video = gr.Video(
                label="Selected Video",
                interactive=True,
                # streaming=True,
                sources="upload",
                height=600,
            )

    # apply on select, not on change
    predefined_video.select(
        fn=on_predefined_video_select,
        inputs=[predefined_video],
        outputs=[video],
    )

    inference_button.click(
        fn=on_inference_button_click,
        inputs=[video],
        outputs=[output_label],
    )

    # when video upload occures inserts the video to the video player
    video.upload(
        fn=on_video_upload,
        inputs=[video],
        outputs=[video, predefined_video],
    )


demo.launch(debug=True)
