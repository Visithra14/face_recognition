# System Architecture Specification
## Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting

---

## 1. High-Level System Architecture

The following diagram illustrates the end-to-end data processing pipeline, from frame ingestion to biometric feature extraction, state tracking, and persistence:

```mermaid
flowchart TD
    subgraph INGESTION["1. Stream Ingestion Layer"]
        SrcFile["Video File (.mp4)"] --> StreamGrabber["VideoSource / RTSP Thread"]
        SrcRTSP["Live RTSP Stream"] --> StreamGrabber
        StreamGrabber --> RingBuffer["Thread-Safe Frame Queue (size=1)"]
    end

    subgraph PIPELINE["2. Core Vision & Tracking Pipeline"]
        RingBuffer --> FrameCheck{"Frame Index % (Skip + 1) == 0?"}
        
        FrameCheck -- "Detection Cycle" --> YOLO["YOLOv8-Face Detector"]
        FrameCheck -- "Skipped Cycle" --> SkipPredict["Tracker Kalman Motion Predict"]
        
        YOLO --> BBoxes["Detected Bounding Boxes"]
        BBoxes --> QualGate{"Quality Gate:<br>1. Conf >= 0.50<br>2. Area >= 40x40<br>3. Blur (Laplacian) >= Thr"}
        
        QualGate -- "Pass" --> Tracker["ByteTrack Multi-Object Tracker"]
        QualGate -- "Reject" --> Discard["Discard Noisy Box"]
        
        SkipPredict --> Tracker
        Tracker --> ActiveTracklets["Active Tracklets (track_id, bbox, state)"]
    end

    subgraph RECOGNITION["3. Biometric Feature Extraction & Identity Engine"]
        ActiveTracklets --> IdentityCheck{"Is Track Bound to face_id?"}
        
        IdentityCheck -- "No (New / Unverified)" --> FaceAlign["Crop Face & Align (112x112)"]
        FaceAlign --> ArcFace["InsightFace ArcFace ONNX Model"]
        ArcFace --> Embedding["512-d L2-Normalized Vector"]
        
        Embedding --> CosineMatch{"Cosine Similarity Against Gallery:<br>max(Sim) >= threshold?"}
        CosineMatch -- "Match Found" --> BindKnown["Bind Existing face_id (VISITOR_XXX)"]
        CosineMatch -- "No Match" --> AutoReg["Auto-Register New face_id (VISITOR_YYY)<br>Update In-Memory Gallery"]
        
        IdentityCheck -- "Yes (Already Bound)" --> PeriodicCheck{"Frame % Reverify == 0?"}
        PeriodicCheck -- "No" --> FastForward["Reuse Bound face_id"]
        PeriodicCheck -- "Yes" --> FaceAlign
        
        BindKnown --> BoundTracklets["Tracklets Bound to Verified Identity"]
        AutoReg --> BoundTracklets
        FastForward --> BoundTracklets
    end

    subgraph STATE_MACHINE["4. Entry / Exit Lifecycle State Machine"]
        BoundTracklets --> StateEvaluator["State Machine (OUTSIDE -> TENTATIVE -> INSIDE -> EXIT_PENDING)"]
        
        StateEvaluator -- "Hits >= min_confirm_frames" --> EntryEvent["Trigger ENTRY Event:<br>1. Atomic Write Crop (logs/entries/)<br>2. Insert Event into DB<br>3. Log to events.log"]
        StateEvaluator -- "Lost >= exit_patience_frames" --> ExitEvent["Trigger EXIT Event:<br>1. Atomic Write Crop (logs/exits/)<br>2. Insert Event into DB<br>3. Log to events.log"]
    end

    subgraph PERSISTENCE["5. Persistence & Storage Layer"]
        EntryEvent --> SQLite[(SQLite Database<br>WAL Mode<br>faces & events tables)]
        ExitEvent --> SQLite
        EntryEvent --> FileLog["Rotating Log File<br>(logs/events.log)"]
        ExitEvent --> FileLog
        EntryEvent --> LocalFS["Local Atomic Image Store<br>(logs/entries/ & logs/exits/)"]
        ExitEvent --> LocalFS
    end

    subgraph PRESENTATION["6. Presentation & Analytics"]
        SQLite --> CLI["CLI Unique Counter (python -m app.count)"]
        BoundTracklets --> HUD["Real-Time OpenCV HUD Overlay"]
        SQLite --> WebUI["Flask Live Visitor Dashboard"]
    end
```

---

## 2. Entry / Exit Lifecycle State Machine

To eliminate duplicate event logging and guarantee exactly **one Entry** and **one Exit** per physical encounter, each detected face track is governed by a finite state machine:

```mermaid
stateDiagram-v2
    [*] --> OUTSIDE: Track Initialized

    OUTSIDE --> TENTATIVE: First Detection in Frame (hits = 1)
    
    TENTATIVE --> TENTATIVE: Track Re-detected (hits < min_confirm_frames)
    TENTATIVE --> OUTSIDE: Track Lost Before Confirmation (Transient False Positive Discarded)
    
    TENTATIVE --> INSIDE: hits >= min_confirm_frames AND face_id Resolved
    note right of INSIDE
        ACTION ON ENTRY:
        1. Atomic save cropped image to logs/entries/YYYY-MM-DD/
        2. Insert record into DB events table (event_type = 'ENTRY')
        3. Write structured audit record to events.log
        4. If new identity, increment registered gallery
    end note

    INSIDE --> INSIDE: Track Continuously Detected (lost_frames = 0)
    INSIDE --> EXIT_PENDING: Track Lost in Frame (lost_frames > 0)

    EXIT_PENDING --> INSIDE: Track Re-detected (lost_frames < exit_patience_frames)
    note right of EXIT_PENDING
        Temporary Occlusion Tolerance:
        Allows person to pass behind poles or people
        without firing false exits
    end note

    EXIT_PENDING --> OUTSIDE: lost_frames >= exit_patience_frames OR Stream Ends
    note right of OUTSIDE
        ACTION ON EXIT:
        1. Atomic save cropped image to logs/exits/YYYY-MM-DD/
        2. Insert record into DB events table (event_type = 'EXIT')
        3. Write structured audit record to events.log
        4. Clear active track session from memory
    end note
```

---

## 3. Storage Layer & Repository Pattern

The persistence tier decouples the database engine from the business logic using the **Repository Pattern**. This allows swapping SQLite for PostgreSQL or MongoDB with zero pipeline changes:

```mermaid
classDiagram
    class DatabaseRepository {
        <<interface>>
        +init_db() void
        +register_face(face_id, embedding, timestamp, crop_path) bool
        +update_face_last_seen(face_id, timestamp) void
        +log_event(face_id, event_type, timestamp, image_path, track_id) void
        +get_all_faces() List[FaceRecord]
        +get_unique_visitor_count() int
        +get_recent_events(limit) List[EventRecord]
        +close() void
    }

    class SQLiteRepository {
        -db_path: str
        -_get_connection(): Connection
        +init_db() void
        +register_face(...) bool
        +update_face_last_seen(...) void
        +log_event(...) void
        +get_all_faces() List[FaceRecord]
        +get_unique_visitor_count() int
        +get_recent_events(...) List[EventRecord]
        +close() void
    }

    class PostgresRepository {
        -conn_str: str
        +init_db() void
        +register_face(...) bool
        +update_face_last_seen(...) void
        +log_event(...) void
        +get_all_faces() List[FaceRecord]
        +get_unique_visitor_count() int
        +get_recent_events(...) List[EventRecord]
        +close() void
    }

    DatabaseRepository <|.. SQLiteRepository
    DatabaseRepository <|.. PostgresRepository
```

### Database Schema Definition:
* **Table `faces`**:
  * `face_id` TEXT PRIMARY KEY (e.g., `VISITOR_001`)
  * `embedding` BLOB NOT NULL (512 float32 binary buffer)
  * `first_seen` TEXT NOT NULL (ISO-8601 UTC timestamp)
  * `last_seen` TEXT NOT NULL (ISO-8601 UTC timestamp)
  * `visit_count` INTEGER DEFAULT 1
  * `first_crop_path` TEXT
* **Table `events`**:
  * `event_id` INTEGER PRIMARY KEY AUTOINCREMENT
  * `face_id` TEXT NOT NULL (Foreign Key -> `faces.face_id`)
  * `event_type` TEXT NOT NULL (`ENTRY` | `EXIT`)
  * `timestamp` TEXT NOT NULL (ISO-8601 UTC timestamp)
  * `image_path` TEXT NOT NULL
  * `track_id` INTEGER NOT NULL
* **Indexes**:
  * `idx_events_face_id` ON `events(face_id)`
  * `idx_events_timestamp` ON `events(timestamp)`

---

## 4. Pipeline Resilience & Fault Recovery

```mermaid
flowchart LR
    A["RTSP Video Stream"] --> B{"Connection Active?"}
    B -- "Yes" --> C["Read Frame to Queue"]
    B -- "No (Dropped Socket)" --> D["Log Warning in events.log"]
    D --> E["Exponential Backoff: 1s, 2s, 4s... 30s"]
    E --> F["Attempt RTSP Reconnect"]
    F --> B

    C --> G["Inference Cycle"]
    G --> H{"Processing Error on Frame?"}
    H -- "Yes" --> I["Catch Exception, Log Stacktrace in events.log, Skip Frame"]
    H -- "No" --> J["Advance Pipeline"]
    I --> J

    J --> K{"SIGINT / SIGTERM Received?"}
    K -- "Yes" --> L["Graceful Shutdown Handler"]
    L --> M["Flush all active in-frame tracks with final EXIT events"]
    M --> N["Sync SQLite WAL & Release Camera"]
    N --> O["Exit Cleanly (Code 0)"]
    K -- "No" --> C
```
