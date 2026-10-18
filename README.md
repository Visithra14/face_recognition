# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-driven real-time computer vision system that detects, tracks, and recognizes human faces in video streams. The system automatically registers new unique identities upon first detection, re-identifies them across frames, strictly logs exactly one entry and exit event per encounter, and maintains an accurate unique visitor counter.

---

## 📑 Video Demonstration

> **Loom / YouTube Explanatory & Demo Video:**  
> 🔗 [Insert Your Loom / YouTube Demo Link Here]  
> *(Note: A recorded video link demonstrating the code in action is required for evaluation).*

---

## 📚 Technical Documentation & AI Planning

Detailed architectural blueprints and engineering specifications created during the AI planning phase:

* 📋 [AI Planning & Problem Decomposition](docs/planning.md): Complete phase breakdown, data models, edge case mitigation, and test strategies.
* 🏗️ [System Architecture & Sequence Flow](docs/architecture.md): Full end-to-end component diagrams, thread isolation, and event state machines.
* ⚡ [Hardware Compute & Resource Estimates](docs/compute_estimate.md): Throughput (FPS), GPU VRAM, CPU consumption, and optimization profiles.
* 🎯 [Feature Specifications & Quality Gates](docs/features.md): Bounding box margins, Laplacian blur filtering, cosine re-identification, and atomic persistence.

---

## 🏗️ Architecture & Pipeline Flow

```mermaid
flowchart TD
    subgraph Stream Ingestion
        A[Input Source: Video File or Live RTSP] --> B[VideoSource Thread Decoupler]
    end

    subgraph Optimization Layer
        B --> C{Frame Skip Interval? config.json}
        C -- "Detection Frame" --> D[YOLOv8-Face Detection + Margin Expansion]
        C -- "Skipped Frame" --> E[ByteTrack Kalman Motion Prediction]
    end

    subgraph Quality Filtering
        D --> QF{Laplacian Variance >= blur_threshold & Size >= min_face_size?}
        QF -- "Passed" --> E
        QF -- "Rejected (Blur/Too Small)" --> E
    end

    subgraph Tracking & Multi-Target Association
        E --> F[Track-to-Face Binding & Age Pruning]
    end

    subgraph Facial Recognition & Feature Extraction
        F --> G[InsightFace ArcFace 512-d L2 Embedding]
        G --> H{Cosine Similarity >= similarity_threshold?}
        H -- "Match" --> I[Re-identify Existing Visitor ID]
        H -- "No Match" --> J[Auto-Register New Face ID: VISITOR_XXX]
    end

    subgraph Lifecycle State Machine
        J --> K[OUTSIDE -> TENTATIVE -> INSIDE: Trigger 1 ENTRY Event]
        I --> L[Continuous Active Tracking & Periodic Re-verification]
        L --> M{Track Inactive > exit_patience_frames or Stream Termination?}
        M -- "Confirmed Exit" --> N[INSIDE -> EXIT_PENDING -> OUTSIDE: Trigger 1 EXIT Event]
    end

    subgraph Atomic Storage & Audit Logging
        K --> O1[Local FS: logs/entries/YYYY-MM-DD/ atomically via .tmp]
        N --> O2[Local FS: logs/exits/YYYY-MM-DD/ atomically via .tmp]
        K & N --> P[(SQLite Database in WAL Mode: faces & events)]
        K & N --> Q[Structured System Log: logs/events.log]
    end

    subgraph Analytics & Web Dashboard
        P --> R[Unique Visitor Counter CLI: python -m app.count]
        P --> S[Real-Time Visual Web Dashboard & OpenCV HUD Overlay]
    end
```

---

## 🚀 Key Features

* **Real-Time YOLOv8-Face Detection**: Dedicated face detector (`models/yolov8n-face.pt`) rather than generic person detector, with margin expansion (12%) for full facial context.
* **ByteTrack Multi-Object Tracking**: High-performance multi-target tracking robust to brief occlusions and trajectory crossings.
* **InsightFace ArcFace 512-d Embeddings**: Deep facial feature representation normalized to unit hypersphere with in-memory NumPy matrix cosine similarity matching.
* **Strict Single Entry / Exit Guarantee**: 4-state lifecycle state machine (`OUTSIDE -> TENTATIVE -> INSIDE -> EXIT_PENDING -> OUTSIDE`) ensuring strictly **one Entry** and **one Exit** event per physical encounter.
* **Zero Double Counting**: Re-identifications update active visitor sessions without inflating unique visitor metrics.
* **Configurable Detection Interval (`config.json`)**: Configurable frame skip (e.g., detect every 3rd or 4th frame, track with Kalman prediction on intermediate frames) for optimal FPS and compute efficiency.
* **Dual Ingestion Engine**: Seamless support for local video files (`.mp4`, `.avi`) and live RTSP camera streams with auto-reconnection and exponential backoff.
* **Full Structured Persistence**:
  * Mandatory rotating audit log (`logs/events.log`) tracking critical lifecycle milestones.
  * Structured local filesystem storage: `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/` written via atomic temp-rename (`.tmp_{name}` -> `{name}`).
  * ACID-compliant SQLite database in WAL (Write-Ahead Logging) mode storing visitor identities, timestamps, total visits, and event history.
* **Interactive Web Dashboard**: Flask-based real-time UI displaying live visitor counts, face snapshots, and live event audit streams.

---

## ⚙️ Configuration (`config.json`)

The application is fully configurable via `config.json`:

```json
{
  "source": {
    "type": "file",
    "path": "data/video_sample1.mp4",
    "rtsp_url": ""
  },
  "detection_skip_frames": 3,
  "detector": {
    "model_path": "models/yolov8n-face.pt",
    "conf_threshold": 0.5,
    "input_size": 640
  },
  "recognition": {
    "model_name": "buffalo_sc",
    "similarity_threshold": 0.45,
    "reverify_every_n_frames": 30
  },
  "tracking": {
    "algorithm": "bytetrack",
    "min_confirm_frames": 5,
    "exit_patience_frames": 30
  },
  "quality": {
    "min_face_size": 40,
    "blur_threshold": 50.0
  },
  "storage": {
    "db_path": "data/faces.db",
    "log_dir": "logs"
  },
  "device": "auto",
  "display": {
    "show_overlay": true,
    "save_output_video": true
  }
}
```

---

## 💻 Compute Load & Hardware Benchmarks

Estimated resource consumption on an **NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)** and 8-Core CPU:

| Metric | CPU-Only (`skip=0`) | GPU Full (`skip=0`) | GPU Optimized (`skip=3-5`) |
| :--- | :--- | :--- | :--- |
| **FPS Throughput** | 8 – 12 FPS | 50 – 65 FPS | **85 – 115+ FPS** |
| **CPU Utilization** | 90% – 100% | 20% – 30% | **12% – 18%** |
| **GPU Utilization** | 0% | 35% – 45% | **12% – 18%** |
| **VRAM Consumption** | ~200 MB | ~1.2 GB | **~1.2 GB** |
| **Latency per Frame** | ~90 ms | ~18 ms | **~8 ms avg** |

### Benchmark Results on Sample Video (`data/video_sample1.mp4`)
* **Input Resolution**: 3840 × 2160 (4K UHD) @ 30 FPS, 240 frames
* **Total Unique Visitors Registered**: 3 (`VISITOR_001`, `VISITOR_002`, `VISITOR_003`)
* **Total Entry Events Logged**: 3 (with timestamped cropped snapshots in `logs/entries/`)
* **Total Exit Events Logged**: 3 (with timestamped cropped snapshots in `logs/exits/`)
* **Double-Count Rate**: 0%
* **Automated Unit & Integration Tests**: 21/21 passed (100% pass rate)

---

## 🛠️ Setup & Installation Instructions

### 1. Prerequisites
* Python 3.10+
* NVIDIA GPU with CUDA (optional; automatic fallback to CPU)

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/face-tracker-visitor-counter.git
cd face-tracker-visitor-counter

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Place Sample Video & Model Weights
1. Ensure the sample video is placed at `data/video_sample1.mp4`.
2. Ensure the YOLOv8-face model weights are at `models/yolov8n-face.pt`.
3. InsightFace ArcFace models (`buffalo_sc`) are automatically fetched by InsightFace on first run.

### 4. Running the Pipeline

```bash
# Standard run (GUI display + video output saving)
python main.py

# Clean reset run (clears previous database and logs before running)
python main.py --reset

# Headless run (for remote servers or automated benchmarking)
python main.py --no-display

# Run on live RTSP camera stream
python main.py --rtsp rtsp://username:password@192.168.1.100:554/stream

# Override frame skip interval (e.g. process detector every 4th frame)
python main.py --skip 4
```

### 5. Querying Visitor Counts & Rosters (CLI)

```bash
# Display full visitor summary with identity roster and recent events
python -m app.count --summary

# Print only the integer count of unique visitors (clean script output)
python -m app.count

# List only currently active visitors in-frame
python -m app.count --active

# Export metrics as JSON
python -m app.count --json
```

### 6. Running the Real-Time Web Dashboard

```bash
python web_dashboard.py
```
Open [http://localhost:5000](http://localhost:5000) in any web browser to view live visitor metrics, registered face cards, and real-time event logs.

### 7. Running the Automated Test Suite

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
All 21 comprehensive tests cover storage, video ingestion, face detection, embedding extraction, gallery re-identification, ByteTrack tracking, state machine transitions, and counting CLI.

---

## 🧠 Assumptions Made

1. **Camera Position & Framing**: The camera captures front or semi-frontal views of faces with a minimum resolution of 40x40 pixels.
2. **Lighting & Occlusions**: Temporary occlusions (e.g., turning sideways or brief hand occlusions) are sustained by ByteTrack's Kalman motion model up to `exit_patience_frames` (30 frames) without falsely triggering exits.
3. **Identity Verification Threshold**: Cosine similarity threshold of `0.45`–`0.55` on ArcFace 512-d unit-normalized vectors provides optimal separation between distinct individuals under varying angles.
4. **Stream Termination Guarantee**: Upon video stream completion or user interruption (`Ctrl+C` or `q`), all currently active visitor sessions are gracefully flushed to trigger final exit events, maintaining 100% audit integrity.
5. **Atomic File Safety**: All cropped snapshots are written to `.tmp_{name}` files first and atomically renamed to prevent partial writes during unexpected crashes.

---

## 📂 Repository File Structure

```
├── app/
│   ├── __init__.py            # Package root & exports
│   ├── config.py              # Strongly-typed configuration dataclasses
│   ├── count.py               # Standalone unique visitor counter CLI
│   ├── detector.py           # YOLOv8-face detector & Laplacian blur filter
│   ├── embedder.py           # InsightFace ArcFace 512-d L2 feature extractor
│   ├── events.py             # Entry/Exit lifecycle state machine & atomic saving
│   ├── identity.py           # In-memory gallery, cosine similarity & auto-registration
│   ├── logger.py             # Thread-safe rotating audit logger
│   ├── pipeline.py           # Master video processing orchestrator & HUD overlay
│   ├── tracker.py            # ByteTrack wrapper, Kalman prediction & identity binding
│   ├── video_source.py       # RTSP reconnect ring buffer & video file ingest
│   └── storage/
│       ├── __init__.py
│       ├── base.py           # Abstract Base Class repository interface
│       └── sqlite_repo.py    # Production SQLite WAL repository with BLOB serialization
├── config.json               # Centralized pipeline configuration
├── data/
│   ├── faces.db              # SQLite database (faces & events tables)
│   └── video_sample1.mp4     # Sample input video
├── docs/
│   ├── architecture.md       # Architectural blueprints & sequence flow
│   ├── compute_estimate.md   # Compute load & benchmark estimates
│   ├── features.md           # Engineering specifications & quality gates
│   └── planning.md           # AI planning & phased breakdown
├── logs/
│   ├── entries/              # Timestamped entry face crops (YYYY-MM-DD)
│   ├── exits/                # Timestamped exit face crops (YYYY-MM-DD)
│   ├── events.log            # System audit event log
│   └── output_annotated.mp4  # Output video with bounding boxes & HUD overlay
├── models/
│   └── yolov8n-face.pt       # YOLOv8 face detector weights
├── tests/                    # 21 unit & integration tests
│   ├── test_counting.py
│   ├── test_embedder_and_identity.py
│   ├── test_state_machine.py
│   ├── test_storage_and_config.py
│   ├── test_tracker.py
│   └── test_video_and_detector.py
├── main.py                   # Main CLI entry point
├── web_dashboard.py          # Flask visual web dashboard
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

This project is a part of a hackathon run by https://katomaran.com
