## HUMAN 2.0 – Architecture Overview

### 1. Top-Level Layers

- **Interface & Serving**
  - `templates/human2_interface.html` ↔ `src/interface/web_ui.py` ↔ `src/serving/app.py` (FastAPI)
  - CLI entrypoints: `src/main.py`, `src/run_human.py`, `src/run_interface.py`, `src/run_emotion_recognition.py`, `src/run_improvement.py`

- **Orchestration Core**
  - `HUMAN2System` in `src/main.py` wires together:
    - `core.human.HUMAN`
    - `components.MultimodalEmotionProcessor`
    - `components.EmotionalMemory`
    - `components.EmotionalLearningSystem`
    - `consciousness.SelfAwarenessSystem`

- **Cognitive & Consciousness Layer**
  - `src/core/human.py` – central state & global workspace
  - `src/consciousness/self_awareness.py`
  - `src/consciousness/reflection.py`
  - `src/consciousness/curiosity.py`
  - `src/consciousness/physiology.py`
  - `src/core/knowledge_representation.py`
  - `src/core/pattern_recognition.py`
  - `src/core/resource_monitor.py`
  - `src/core/version_control.py`

### 2. Emotion & Multimodal Layer

- **Multimodal Processor**
  - `src/components/multimodal_emotion_processor.py` combines:
    - Text → `src/models/emotion_recognition.py` (ERADEM) and `src/models/hierarchical_eradem.py`
    - Audio → `src/models/audio_emotion_model.py` (trained via `src/train_robust_audio_model.py`)
    - Visual → `src/components/deepface_visual_emotion.py` + `src/models/visual_emotion_model.py`

- **Emotion Systems**
  - `src/components/emotional_memory.py`
  - `src/components/emotional_learning.py`
  - Real-time:
    - `src/realtime_audio_emotion_improved.py`
    - `src/components/realtime_emotion_monitor.py`

### 3. Self-Coding / Self-Improvement Layer

- **Coordinator**
  - `src/components/self_coding_ai.py` orchestrates code understanding and modification using:
    - `src/components/code_analyzer.py`
    - `src/components/code_env.py`
    - `src/components/code_actions.py`
    - `src/components/code_metrics.py`
    - `src/components/self_improvement.py`
    - `src/components/self_modification.py`
    - `src/components/self_analysis.py`
    - `src/components/continuous_learning.py`
    - `src/components/external_learning.py`
    - `src/components/web_learning.py`
    - `src/components/firecrawl_knowledge.py`
    - `src/components/github_integration.py`
    - `src/embedder/code_embedder.py`
    - `src/components/llm_plugin.py`

- **Optimization Support**
  - `src/optimization/pipeline.py`
  - `src/training/train.py`

### 4. Data & Collection

- **Data Preparation**
  - `src/data/prepare_dataset.py`
  - `src/data/prepare_emotion_datasets.py`
  - `src/data/augment_emotion_data.py`

- **Collection Pipelines**
  - `src/data_collection/collect_data.py`
  - `src/data_collection/doc_collector.py`
  - `src/data_collection/github_collector.py`

- **Model Artifacts (root `models/` directory)**
  - Trained text, audio, and visual models (various `.pt` / `.pth` / `.json` / tokenizer files)
  - Training curves, confusion matrices, and config JSONs

### 5. Interfaces & Serving

- **CLI & Processes**
  - `src/main.py` – primary HUMAN 2.0 entry point with interactive CLI
  - `src/run_human.py`, `src/run_interface.py`, `src/run_emotion_recognition.py`, `src/run_improvement.py`

- **Web / API**
  - `src/interface/user_interface.py`
  - `src/interface/web_ui.py`
  - `src/serving/app.py` (FastAPI application)
  - `templates/human2_interface.html` – main HTML UI template

### 6. Tests

- `src/test/` and `tests/` contain:
  - ERADEM and model tests
  - Multimodal and emotion integration tests
  - Emotional memory/learning and performance tests
  - Self-improvement and GitHub-learning tests
  - Version-control and pattern-recognition tests

### 7. End-to-End Data & Control Flow (Text-Centric)

```text
User (CLI / Web UI / API)
   ↓
HUMAN2System.process_input({...})
   ↓
MultimodalEmotionProcessor.process_multimodal_input(text, audio?, visual?)
   ↓
 Text → ERADEM (emotion_recognition.py / hierarchical_eradem.py)
 Audio → audio_emotion_model.py
 Visual → DeepFaceVisualEmotionProcessor
   ↓
Fused EmotionResult (dominant emotion, intensity, confidence)
   ↓
EmotionalMemory.add_experience(...)
EmotionalLearningSystem.update(...)
SelfAwarenessSystem.add_experience(...)
HUMAN.process_input(...)
   ↓
HUMAN2System._generate_response(...)
   ↓
Response + emotion metadata → Interface / API
```

