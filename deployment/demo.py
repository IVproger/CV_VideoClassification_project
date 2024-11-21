import os
import torch
import numpy as np
import gradio as gr

from peft import PeftModel, PeftConfig
from transformers import (
    pipeline,
    AutoModelForVideoClassification,
    AutoImageProcessor,
    VideoMAEImageProcessor,
)

# Load predefined videos
PREDEFINED_VIDEOS_DIR = "videos"
PREDEFINED_VIDEOS = {
    video: os.path.join(PREDEFINED_VIDEOS_DIR, video)
    for video in os.listdir(PREDEFINED_VIDEOS_DIR)
    if video.endswith((".mp4", ".avi", ".mkv"))
}
# null value for this dropdown
NO_PREDEFINED_VIDEO_SELECTED = "No predefined video selected"
PREDEFINED_VIDEOS[NO_PREDEFINED_VIDEO_SELECTED] = ""

# model
PREDEFINED_MODELS_DIR = "models"
PREDEFINED_MODELS = {
    model_dir: os.path.join(PREDEFINED_MODELS_DIR, model_dir)
    for model_dir in os.listdir(PREDEFINED_MODELS_DIR)
}
DEFAULT_MODEL = list(PREDEFINED_MODELS.keys())[1]


USE_LORA_MODEL = False
BASE_MODEL = "MCG-NJU/videomae-base"
LORA_MODEL = "model"

# model = AutoModelForVideoClassification.from_pretrained(
#     BASE_MODEL, ignore_mismatched_sizes=True
# )
# if USE_LORA_MODEL:
#     config = PeftConfig.from_pretrained(LORA_MODEL, inference_mode=True)
#     model = PeftModel.from_pretrained(
#         model,
#         LORA_MODEL,
#         config=config,
#         is_trainable=False,
#         ignore_mismatched_sizes=True,
#     )
# image_processor = VideoMAEImageProcessor.from_pretrained(LORA_MODEL)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline(
    "video-classification",
    model=PREDEFINED_MODELS[DEFAULT_MODEL],  # paths to local model setup
    image_processor=PREDEFINED_MODELS[DEFAULT_MODEL],
    device=device,
)


def on_predefined_video_select(predefined_video: str):
    path = PREDEFINED_VIDEOS.get(predefined_video)
    # if its "" we don't need to propagate path to video
    if path:
        return path


def on_predefined_model_select(predefined_model: str):
    path = PREDEFINED_MODELS.get(predefined_model)

    # if its "" we don't need to propagate path to video
    if path:
        # reload global pipeline variable
        global pipe
        pipe = pipeline(
            "video-classification",
            model=path,
            image_processor=path,
            device=device,
        )

        return predefined_model


def on_inference_button_click(video: str):
    # https://huggingface.co/docs/transformers/en/tasks/video_classification#inference
    if video:
        res = pipe(video)

        if res is not None and len(res) > 0:
            # get most probable label and score
            label = res[0]["label"]
            score = res[0]["score"]
            return f"{label} {score * 100:.2f}%"
        else:
            return "No result"


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
                choices=list(PREDEFINED_VIDEOS.keys()),
                label="Select Predefined Video",
                value=NO_PREDEFINED_VIDEO_SELECTED,
                elem_id="predefined_video",
            )
            predefined_model = gr.Dropdown(
                choices=list(PREDEFINED_MODELS.keys()),
                label="Select Predefined Model",
                value=DEFAULT_MODEL,
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

    predefined_model.select(
        fn=on_predefined_model_select,
        inputs=[predefined_model],
        outputs=[predefined_model],
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


demo.launch(share=False, debug=True)
