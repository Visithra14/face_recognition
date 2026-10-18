# Project Planning Document
## Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Business & Computer Vision Challenge
Conventional foot-traffic systems rely on beam sensors, floor mats, or generic bounding-box tripwires (e.g., counting whenever any human silhouette crosses a virtual line). These methods suffer from severe double-counting:
- A staff member pacing back and forth generates dozens of false counts.
- A visitor browsing two aisles triggers multiple entry counts.
- Occluded individuals in dense crowds are repeatedly counted as new targets when tracker IDs drop out.

### 1.2 The Solution
This project implements an intelligent, automated, computer-vision visitor analytics system that decouples **physical motion tracking** from **biometric facial identity**:
1. **Dynamic Auto-Registration**: Novel faces are detected and embedded in real-time on first sight, creating unique identities (`VISITOR_001`, `VISITOR_002`, etc.) without pre-enrollment.
2. **Persistent Biometric Re-Identification**: If an individual leaves the frame and returns minutes or hours later, deep facial embeddings (InsightFace ArcFace 512-d) match them against the gallery, recognizing them as an existing visitor and **strictly preserving the unique visitor count**.
3. **Deterministic Entry/Exit Auditing**: A formal state machine (`OUTSIDE` -> `TENTATIVE` -> `INSIDE` -> `EXIT_PENDING` -> `OUTSIDE`) enforces that every physical visit produces **exactly one Entry event and one Exit event**, each accompanied by an atomically written face crop snapshot, timestamp, and SQLite database record.
4. **Production Resiliency**: Dual-source ingestion (file vs. live RTSP with exponential backoff reconnect), WAL-mode ACID persistence, thread-safe memory vector caching, and frame-skipping optimizations.

---

## 2. Technical Decisions & Architectural Rationales

| Decision | Selected Choice | Rejected Alternatives | Technical Interview Rationale |
| :--- | :--- | :--- | :--- |
| **Face Detection** | **YOLOv8-Face (`yolov8n-face.pt`)** | Generic YOLO `person` class, Haar Cascades, MTCNN | Generic `person` class detects whole bodies, which leads to poor crops when bodies overlap. Haar cascades are fragile to lighting/rotations. MTCNN is too slow for real-time video on CPU/edge. YOLOv8-Face provides high mAP on faces even in extreme angles and runs at >100 FPS on GPU. |
| **Face Embedding** | **InsightFace ArcFace (512-d ResNet/MobileFaceNet)** | `face_recognition` (dlib ResNet-34), Facenet | `face_recognition` is based on older dlib models trained with triplet loss on small datasets, failing on partial yaw angles and masks. ArcFace utilizes Additive Angular Margin loss to maximize inter-class variance and normalize intra-class variance on a hypersphere. |
| **Multi-Object Tracking** | **ByteTrack (Kalman Filter + Low/High Conf Association)** | Simple Centroid Tracking, SORT, DeepSORT | Centroid tracking fails on crossing paths. SORT drops tracks during occlusion. DeepSORT runs a heavy Re-ID network per bounding box on every frame. ByteTrack associates both high-score and low-score detection boxes, keeping tracks alive during motion blur and partial occlusions with zero GPU overhead. |
| **Identity Association Strategy** | **Track-to-Face Binding (`track_id` -> `face_id`)** | Run face recognition on every frame | Extracting 512-d embeddings for 10 faces every frame consumes 100+ ms, crippling FPS. By binding `track_id` to `face_id` after confirmation, recognition is only executed on unverified tracks and periodically refreshed (`reverify_every_n_frames`), reducing recognition overhead by >80%. |
| **Database Engine** | **SQLite with WAL (Write-Ahead Logging)** | PostgreSQL, MongoDB, Plain CSV/JSON | SQLite requires zero daemon configuration, is completely self-contained for hackathons/evaluations, and WAL mode allows concurrent non-blocking reads while writing events. A repository pattern abstracts all SQL operations so switching to Postgres requires changing only the database adapter. |
| **Image Persistence** | **Atomic Temp-to-Rename Filesystem Writes** | Writing directly to final destination | Direct writes risk leaving corrupted/truncated JPEG images if the system crashes or encounters `SIGINT` mid-write. Writing to `.tmp` and invoking atomic `os.replace` guarantees file integrity. |

---

## 3. System Module Decomposition

```
face-tracker/
├── app/
│   ├── __init__.py
│   ├── config.py           # Strictly validated configuration loader (JSON -> Dataclass)
│   ├── video_source.py     # Resilient frame grabber (File reader & RTSP with auto-reconnect)
│   ├── detector.py         # YOLOv8-face inference wrapper with quality checks (blur/size)
│   ├── embedder.py         # InsightFace ArcFace 512-d feature extractor
│   ├── identity.py         # In-memory vector gallery + Cosine Similarity matching engine
│   ├── tracker.py          # ByteTrack multi-object tracking & detection-skip updater
│   ├── events.py           # Formal Entry/Exit state machine & atomic snapshot writer
│   ├── count.py            # CLI and programmatically accessible unique visitor counter
│   ├── pipeline.py         # Master pipeline orchestrator tying all modules together
│   ├── logger.py           # Thread-safe rotating logger for events.log and console
│   └── storage/
│       ├── __init__.py
│       ├── base.py         # Abstract Database Repository Interface
│       └── sqlite_repo.py  # SQLite implementation (WAL, transactions, indexes)
├── tests/
│   ├── test_state_machine.py
│   ├── test_identity_matching.py
│   └── test_counting.py
├── docs/
│   ├── planning.md
│   ├── features.md
│   ├── architecture.md
│   └── compute_estimate.md
├── logs/
│   ├── events.log
│   ├── entries/YYYY-MM-DD/
│   └── exits/YYYY-MM-DD/
├── data/
│   ├── video_sample1.mp4
│   └── faces.db
├── config.json
├── requirements.txt
├── main.py
└── README.md
```

---

## 4. Edge Cases & Engineering Mitigations

### 4.1 ID Switches in Dense Crowds
* **The Problem**: Two individuals cross paths; the tracker Kalman filter swaps bounding box IDs.
* **Mitigation**: Periodic re-verification (`reverify_every_n_frames: 30`). When a track is re-verified, its new embedding is compared with its bound `face_id`. If similarity drops significantly and matches a different active identity, the binding updates gracefully without creating a phantom new visitor.

### 4.2 Face Masks, Profile Angles, and Low-Light Blur
* **The Problem**: A visitor walks into frame looking down; the detector crops a blurry forehead. Registering a low-quality crop poisons the gallery with poor embeddings.
* **Mitigation**: **Multi-stage Quality Gate** in `detector.py`:
  1. Minimum bounding box dimension: $\ge 40 \times 40$ px.
  2. Detector confidence cutoff: $\ge 0.50$.
  3. Laplacian variance blur check: $\text{Var}(\nabla^2 I) \ge \text{threshold}$ (rejects motion-blurred frames).
  4. Best-face caching: The tracker updates the reference image dynamically whenever a higher-resolution, sharper crop is detected.

### 4.3 RTSP Stream Dropouts and Network Jitter
* **The Problem**: IP camera feeds drop packets, disconnect, or stall on slow WiFi.
* **Mitigation**: Decoupled background frame reader thread with ring buffer (`queue.Queue(maxsize=1)`) and automatic exponential backoff reconnect logic (1s, 2s, 4s, up to 30s) so video decoding never blocks the AI pipeline.

### 4.4 Graceful Shutdown & Interrupted Encounters
* **The Problem**: System terminates via `Ctrl+C` while 5 people are actively in frame, missing their exit events.
* **Mitigation**: Registered `signal.signal(SIGINT, ...)` and `SIGTERM` handlers flush active sessions, write their exit snapshots, log exit events to the database, and sync the WAL file before shutting down.

---

## 5. Phased Implementation Roadmap

1. **Phase 1: Planning & Architecture Specification** *(Current Phase)*
   - `docs/planning.md`, `docs/features.md`, `docs/architecture.md`, `docs/compute_estimate.md`.
2. **Phase 2: Project Scaffold & Storage Infrastructure**
   - Clean module layout, validated `config.py`, rotating `logger.py`, repository interfaces, and SQLite WAL database schema.
3. **Phase 3: Resilient Video Ingestion & YOLOv8-Face Detection**
   - `video_source.py` supporting files and auto-reconnecting RTSP, and `detector.py` with Laplacian blur and size quality filtering.
4. **Phase 4: InsightFace ArcFace Embedding & Identity Engine**
   - 512-d normalized embedding extraction, memory gallery, cosine similarity matching, and auto-registration logic.
5. **Phase 5: ByteTrack Tracking & Detection Frame Skipping**
   - Multi-target tracking, detection skipping ($N$ frames), Kalman motion prediction, and track-to-face binding.
6. **Phase 6: Entry/Exit Lifecycle State Machine & Image Storage**
   - Formal de-duplication state machine, atomic JPEG writing (`.tmp` -> rename), and structured database logging.
7. **Phase 7: Unique Visitor Analytics, Live HUD, and Pipeline Resiliency**
   - `pipeline.py` master orchestrator, real-time OpenCV HUD overlay, `app.count` CLI, and graceful shutdown handlers.
8. **Phase 8: Test Suite, Benchmark Runs, and Submission Packaging**
   - Unit tests, test execution on `data/video_sample1.mp4`, sample output verification, and `README.md` completion.
