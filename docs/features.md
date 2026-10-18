# System Features & Functional Specifications
## Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting

---

## 1. Functional Features Catalog

### 1.1 Ingestion & Input Handling
* **Dual-Source Ingestion**: Ingests video from either local video files (`.mp4`, `.avi`, `.mov`) or live RTSP IP camera streams (`rtsp://username:password@ip:port/stream`) configured solely via `config.json`.
* **Zero-Lag RTSP Ring Buffer**: Dedicated worker thread constantly drains the RTSP buffer, keeping the internal queue size to 1. This prevents network latency buffering and ensures real-time inference without lag accumulation.
* **Auto-Reconnect with Exponential Backoff**: If network hiccups, camera reboot, or RTSP socket disconnection occurs, the ingestion engine automatically attempts reconnection with exponential backoff (1s, 2s, 4s, up to 30s) without crashing the application.
* **Dynamic Resolution Scaling**: Allows configuring maximum inference resolution (e.g., 640x640) while maintaining original aspect ratios.

### 1.2 Face Detection & Quality Filtering
* **YOLOv8-Face Model Inference**: Dedicated facial bounding box detection powered by YOLOv8 trained specifically on WIDER Face, handling extreme yaw, pitch, and profile angles.
* **Laplacian Variance Blur Filter**: Computes the variance of the Laplacian operator on every candidate crop:
  $$\text{BlurScore} = \text{Var}(\nabla^2 I_{\text{crop}})$$
  Rejects motion-blurred or out-of-focus faces below `blur_threshold` to prevent gallery contamination.
* **Minimum Size Gate**: Rejects candidate bounding boxes smaller than `min_face_size` (e.g. $40 \times 40$ px) to prevent noisy background false positives.
* **Configurable Detection Frame Skipping**: Supports skipping detection cycles every $N$ frames (`detection_skip_frames`), running YOLO only on key frames and letting the tracker predict intermediate motion, cutting compute consumption by up to 75%.

### 1.3 Face Recognition & Auto-Registration
* **InsightFace ArcFace 512-d Feature Extraction**: Generates deep L2-normalized 512-dimensional facial feature vectors using Additive Angular Margin loss networks (e.g. ResNet50 or MobileFaceNet).
* **Strict Avoidance of `face_recognition`**: Implements pure state-of-the-art ONNX / InsightFace deep neural models, completely avoiding the forbidden legacy `face_recognition` (dlib) library.
* **In-Memory Fast Cosine Similarity Matching**: Maintains an active gallery matrix $\mathbf{G} \in \mathbb{R}^{M \times 512}$ in memory. Computes vector dot-product similarity against all registered identities in sub-millisecond time:
  $$\text{Sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} \quad (\text{since } \|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1)$$
* **Automatic Dynamic Enrollment**: When a candidate face produces $\max(\text{similarity}) < \text{threshold}$, the system automatically registers the subject as a new unique identity (`VISITOR_001`, `VISITOR_002`), saves their embedding to the gallery, and persists it to the database.
* **Re-Identification (Zero Double Counting)**: When a registered subject reappears, $\max(\text{similarity}) \ge \text{threshold}$ re-links them to their existing `face_id`. The unique visitor counter remains strictly unaltered.
* **Best-Crop Updating**: Continuously compares candidate crops during a track session, storing the highest-resolution, sharpest image as the official representative portrait.

### 1.4 Multi-Object Tracking & Association
* **ByteTrack Motion Association**: Associates bounding boxes across frames using Kalman filters and low/high confidence IoU matching.
* **Track-to-Face Binding (`track_id` -> `face_id`)**: Once a track is confirmed and matched to a `face_id`, recognition is cached on that track, avoiding redundant deep embedding calculations on every subsequent frame.
* **Periodic Re-Verification**: Evaluates active tracks every $K$ frames (`reverify_every_n_frames: 30`) to detect and heal tracker ID-switches in dense crowd crossings.

### 1.5 Strict Entry/Exit Lifecycle State Machine
* **De-Duplication State Machine**: Enforces a strict state flow per physical visit:
  $$\text{OUTSIDE} \longrightarrow \text{TENTATIVE} \longrightarrow \text{INSIDE} \longrightarrow \text{EXIT\_PENDING} \longrightarrow \text{OUTSIDE}$$
* **Entry Confirmation Barrier**: Requires a track to be consistently detected for `min_confirm_frames` (e.g. 5 frames) before declaring an **ENTRY**. Filters out transient detection flickers.
* **Exit Patience Window**: When a track is lost or steps outside the frame, an exit is not fired immediately. A grace window of `exit_patience_frames` (e.g. 30 frames) allows tracking recovery through temporary occlusions. Once patience expires, exactly **one EXIT event** is generated.
* **Stream-End Flusher**: When the video ends or the user terminates the program, all active in-frame tracks are automatically flushed with final exit events to ensure total accounting symmetry ($\text{Entries} = \text{Exits} + \text{In-Frame}$).

### 1.6 Storage, Logging & Auditability
* **Atomic Snapshot Persistence**: Face crop images are written to a temporary filename (`.tmp`) and atomically renamed (`os.replace`) to their final path (`logs/entries/YYYY-MM-DD/{face_id}_entry_{timestamp}.jpg`), guaranteeing zero file corruption during unexpected power cuts or aborts.
* **Structured Directory Hierarchy**: Automatically organizes crops into date-partitioned folders:
  - `logs/entries/YYYY-MM-DD/`
  - `logs/exits/YYYY-MM-DD/`
* **ACID SQLite Database (WAL Mode)**:
  - Table `faces`: `face_id`, `embedding` (BLOB), `first_seen`, `last_seen`, `visit_count`.
  - Table `events`: `event_id`, `face_id`, `event_type`, `timestamp`, `image_path`, `track_id`.
  - WAL (Write-Ahead Logging) mode enabled to allow non-blocking concurrent reads while inserting events.
* **Mandatory Rotating Audit Log (`logs/events.log`)**: Records all milestones with millisecond timestamps:
  - Stream connection/disconnection
  - Face detection & quality assessment
  - Embedding extraction & registration
  - Re-identification matches with cosine scores
  - State machine transitions (ENTRY / EXIT)
  - System exceptions and shutdown summaries

### 1.7 Analytics, Visualization & Dashboards
* **CLI Unique Counter**: Dedicated CLI utility (`python -m app.count`) and programmatic query API `repo.get_unique_visitor_count()`.
* **Real-Time Heads-Up Display (HUD)**: Sleek OpenCV video overlay showing:
  - Green / Cyan bounding boxes with `face_id`, `track_id`, and cosine similarity.
  - Top status bar: **Unique Visitors**, **Current In-Frame Count**, **Instant FPS**, and **Mode** (`FILE` vs `RTSP`).
* **Interactive Web Dashboard (Bonus UI)**: Real-time web interface displaying:
  - Metric badges: Total Unique Visitors, Total Entries, Total Exits.
  - Live visitor card grid with photo thumbnails.
  - Live audit table of recent entry/exit events.

---

## 2. Non-Functional Features

| Metric | Target Specification |
| :--- | :--- |
| **Throughput (GPU)** | $\ge 60 \text{ FPS}$ on RTX 4060 Laptop GPU with `detection_skip_frames: 3` |
| **Throughput (CPU)** | $\ge 20 \text{ FPS}$ on standard 8-core CPU with frame skipping |
| **Embedding Generation Latency** | $\le 15 \text{ ms}$ per face on GPU |
| **Vector Matching Latency** | $\le 0.5 \text{ ms}$ for 1,000 registered identities |
| **Database Concurrency** | Zero lock contention under continuous frame writes (SQLite WAL) |
| **Crash Safety** | Crash-safe recovery: Gallery is restored from DB on startup |
| **Extensibility** | Database repository abstraction allows zero-refactor migration to Postgres or Mongo |
