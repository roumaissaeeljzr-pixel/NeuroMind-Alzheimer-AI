# Alzheimer Multimodal Assistant

Modular decision-support project for Alzheimer / dementia related analysis.

The project contains two independent modules:

- `cnn_module/`: 3D brain MRI analysis with a CNN.
- `nlp_rag_module/`: speech transcript analysis with a hybrid NLP model, FAISS RAG, and a local Ollama LLM.

Important: CNN and NLP predictions are not fused into one final diagnosis. The MRI dataset and the speech transcript dataset are not paired at patient level.

## Project Structure

```text
alzheimer_multimodal_assistant/
├── app.py
├── cnn_module/
│   ├── data/
│   │   ├── ADNI_SKULL_STRIPPED/
│   │   └── dataset_preprocessed.csv
│   ├── models/
│   │   └── best_resnet3d_model.pth
│   ├── outputs/
│   └── src/
│       ├── cnn_model.py
│       ├── cnn_predictor.py
│       ├── train_cnn.py
│       ├── create_csv_v2.py
│       ├── brain_extractor.py
│       └── extract_zips_fast.py
├── nlp_rag_module/
│   ├── data/
│   ├── model/
│   ├── rag_documents/
│   ├── vector_store/
│   └── src/
│       ├── app_patient_chat.py
│       ├── build_rag_index.py
│       ├── llm_generator.py
│       ├── main_pipeline.py
│       ├── model_architecture.py
│       ├── predict_nlp_model.py
│       ├── rag_explainer.py
│       ├── report_runner.py
│       └── run_from_dataset.py
├── outputs/
├── requirements.txt
└── README.md
```

## Main Interface

Run the modular Streamlit interface:

```bash
streamlit run app.py
```

Tabs:

- Brain MRI Analysis: upload a `.nii` or `.nii.gz` image, run CNN prediction (`CN` or `AD`), show confidence, generate a RAG + LLM explanation, and optionally generate a 3D Grad-CAM visualization.
- Speech Transcript Analysis: enter transcript + 9 features, run NLP prediction, use RAG chat, and generate a downloadable patient report.

## CNN Module

The CNN module analyzes 3D MRI files from ADNI.

Expected files:

- `cnn_module/data/dataset_preprocessed.csv`
- `cnn_module/data/ADNI_SKULL_STRIPPED/`
- `cnn_module/models/best_resnet3d_model.pth`

Command-line inference:

```bash
python cnn_module/src/cnn_predictor.py --image path/to/image.nii.gz
```

CNN outputs:

- predicted label: `CN` or `AD`
- confidence
- probability CN
- probability AD

Grad-CAM:

- `cnn_module/src/gradcam_3d.py` generates a central axial slice with a 3D Grad-CAM heatmap overlay.
- Grad-CAM images are saved in `outputs/gradcam/`.
- This visualization is only an explainability tool and is not a medical diagnosis.

## NLP + RAG + LLM Module

The NLP module analyzes:

- an English patient transcript
- 9 clinical / linguistic features
- a hybrid RoBERTa classifier
- FAISS medical-document retrieval
- a local Ollama LLM

Standalone NLP interface:

```bash
streamlit run nlp_rag_module/src/app_patient_chat.py
```

## Ollama

Before using RAG + LLM explanations, start the local model:

```bash
ollama run gemma3:1b
```

## Installation

```bash
pip install -r requirements.txt
```

## Medical Note

This project does not provide a final medical diagnosis. CNN and NLP outputs are decision-support results and must be interpreted by a qualified healthcare professional.
