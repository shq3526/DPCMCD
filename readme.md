# Dual-Path Compression for Real-Time Multimodal Clickbait Detection: Quantization and Distillation

This project explores two primary techniques for optimizing a multi-modal classification model: **Knowledge Distillation** and **Quantization-Aware Training (QAT)**.

## 1. Prerequisites

Before you begin, ensure you have the following installed:
* **Python**: Version 3.10 is recommended.
* **GPU**: An NVIDIA GPU is highly recommended for training.

## 2. Setup & Installation

Follow these steps to set up your project environment.

#### Step 1: Create and Activate a Virtual Environment
Using a virtual environment is recommended to avoid conflicts with other projects.

* **Create the environment** (this will create a `.venv` folder):
    * On **Linux / macOS**:
        ```bash
        python3 -m venv .venv
        ```
    * On **Windows**:
        ```bash
        python -m venv .venv
        ```
* **Activate the environment**:
    * On **Linux / macOS**:
        ```bash
        source .venv/bin/activate
        ```
    * On **Windows (PowerShell)**:
        ```bash
        .venv\Scripts\Activate.ps1
        ```

#### Step 2: Install PyTorch and PyTorch Geometric (PyG)
* **PyTorch**: Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to generate the correct installation command for your specific OS and CUDA version.
* **PyG**: Follow the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install PyTorch Geometric.

#### Step 3: Install Project Dependencies
Install all other required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
````

## 3\. Download Pre-trained Models

This project relies on several pre-trained models. If they are not already included, you must download them.

1.  **Create a `model` directory** in the project root:

    ```bash
    mkdir model
    ```

2.  **Download the models**:

    ```bash
    # Navigate into the model directory
    cd model

    # Install Git LFS (required for large model files)
    git lfs install

    # Clone model repositories
    git clone [https://huggingface.co/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
    git clone [https://huggingface.co/openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)

    # The scripts also require the LTP model. Clone it into the project root.
    cd ..
    git clone [https://huggingface.co/LTP/small](https://huggingface.co/LTP/small) LTP/small
    ```

## 4\. Running the Project

You can follow one of the two main technical routes below.

-----

### Route 1: Model Distillation

This route involves training smaller "student" models to mimic the behavior of larger "teacher" models, resulting in lightweight models that can be finetuned.

#### Workflow A: Independent Distillation with Preprocessing (`nosyc` series)

1.  **Preprocess Graph Data**: Convert titles into graph structures.

    ```bash
    python preprocess.py
    ```

      * **Output**: Graph files (`data_*.pt`) in the `./processed_data/` directory.

2.  **Distill Models Independently**: Create lightweight versions of GAT, vision, and text models.

    ```bash
    python distill_nosyc.py
    ```

      * **Output**: Models saved to `./lightweight_gat_model.pth`, `./lightweight_content_model_distilled/`, and `./lightweight_vision_model_distilled/`.

3.  **Finetune the Final Model**: Train the distilled models jointly on the downstream task.

    ```bash
    python finetune_nosyc_distilled_model.py
    ```

      * **Output**: The final finetuned model at `./final_lightweight_model.pth`.

#### Workflow B: Synergistic Distillation with On-the-fly Processing (`syc` series)

1.  **Distill Models**:

    ```bash
    python distill_syc.py
    ```

      * **Output**: All model components saved into the `./syc_new_lightweight_models/` directory.

2.  **Finetune the Final Model**: This step processes graphs on-the-fly during training.

    > **Note:** The output directory of `distill_syc.py` (`syc_new_lightweight_models`) does not match the input directory expected by `finetune_syc_distilled_model.py` (`syc_lightweight_models`). You must rename the directory or modify the path in one of the scripts before running.

    ```bash
    python finetune_syc_distilled_model.py
    ```

      * **Output**: The final finetuned model at `./final_distilled_multimodal_model.pth`.

-----

### Route 2: Model Quantization

This route uses Quantization-Aware Training (QAT) to reduce the model's precision from FP32 to INT8, significantly accelerating inference speed with minimal accuracy loss.

#### Run Training (FP32 or QAT)

The primary script for this route allows you to train a model in standard full precision (FP32) or with Quantization-Aware Training (QAT). You can likely control this behavior via configuration files (e.g., `./config/config.txt`).

```bash
python main_benchmark_trainer_FP32_or_QAT_with_ablation.py
```

#### Run Ablation Study

An additional script is provided to perform ablation studies on the teacher model, likely to analyze the impact of different components.

```bash
python ablation_study_teacher_model.py
```

```
```