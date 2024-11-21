# Deployment

This folder contains scripts and configurations for deploying the trained models. It includes instructions and files necessary to deploy the models to various environments, such as cloud services or local servers.

## Prerequisites

It is expected to have `videos` and `models` folders along `demo.py` to be able choosing default videos or changing inference model.

**Structure:**

```
deployment/
- demo.py 
- models/
- - model_name/
- - - ... (model state dict)
- videos/
- - example.(avi/mp4)
```

## Run

> From deployment folder

**In Production**

```bash
python demo.app
```

**In Development**

```bash
gradio demo.app
```