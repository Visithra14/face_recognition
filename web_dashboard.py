import os
import sqlite3
from flask import Flask, render_template_string, jsonify, send_from_directory
try:
    from app.storage.sqlite_repo import SQLiteRepository
except ImportError:
    SQLiteRepository = None

app = Flask(__name__)
DB_PATH = "data/faces.db" if os.path.exists("data/faces.db") else "logs/visitors.db"

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Face Tracker & Unique Visitor Counter</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-primary: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.25);
            --entry-color: #34d399;
            --exit-color: #f87171;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(52, 211, 153, 0.05) 0%, transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 3rem;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }

        .title-group h1 {
            font-size: 1.85rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .title-group p {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .badge-live {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(52, 211, 153, 0.1);
            border: 1px solid rgba(52, 211, 153, 0.3);
            color: var(--entry-color);
            padding: 0.45rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background: var(--entry-color);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--entry-color);
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
        }

        /* Metric Cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-primary), transparent);
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-value {
            font-size: 2.75rem;
            font-weight: 800;
            color: var(--text-main);
            margin-top: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Content Layout */
        .dashboard-layout {
            display: grid;
            grid-template-columns: 1.2fr 1.8fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-layout {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.75rem;
        }

        .panel-header h2 {
            font-size: 1.25rem;
            font-weight: 600;
        }

        /* Visitors Gallery */
        .visitor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 1rem;
            max-height: 480px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }

        .visitor-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.75rem;
            text-align: center;
            transition: transform 0.2s, border-color 0.2s;
        }

        .visitor-card:hover {
            transform: translateY(-3px);
            border-color: var(--accent-primary);
        }

        .visitor-img {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid var(--accent-primary);
            margin: 0 auto 0.5rem;
            background: #1e293b;
        }

        .visitor-id {
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--accent-primary);
        }

        .visitor-time {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }

        /* Event Table */
        .events-table-wrapper {
            max-height: 480px;
            overflow-y: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }

        th {
            text-align: left;
            padding: 0.75rem;
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--card-border);
            position: sticky;
            top: 0;
            background: var(--bg-color);
        }

        td {
            padding: 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-family: 'JetBrains Mono', monospace;
        }

        .event-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .badge-entry {
            background: rgba(52, 211, 153, 0.15);
            color: var(--entry-color);
            border: 1px solid rgba(52, 211, 153, 0.3);
        }

        .badge-exit {
            background: rgba(248, 113, 113, 0.15);
            color: var(--exit-color);
            border: 1px solid rgba(248, 113, 113, 0.3);
        }

        .table-thumb {
            width: 36px;
            height: 36px;
            border-radius: 6px;
            object-fit: cover;
            vertical-align: middle;
            border: 1px solid var(--card-border);
        }

        footer {
            margin-top: 3rem;
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-top: 1px solid var(--card-border);
            padding-top: 1.5rem;
        }
        footer a {
            color: var(--accent-primary);
            text-decoration: none;
        }
    </style>
</head>
<body>
    <header>
        <div class="title-group">
            <h1>Face Tracker & Unique Visitor Counter</h1>
            <p>AI Pipeline: YOLOv8 Face + ByteTrack + InsightFace ArcFace + SQLite</p>
        </div>
        <div class="badge-live">
            <div class="pulse-dot"></div>
            SYSTEM ONLINE
        </div>
    </header>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Unique Visitors</div>
            <div class="metric-value" id="val-unique">0</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Entry Events</div>
            <div class="metric-value" id="val-entries">0</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Exit Events</div>
            <div class="metric-value" id="val-exits">0</div>
        </div>
    </div>

    <div class="dashboard-layout">
        <!-- Unique Visitors Gallery -->
        <div class="panel">
            <div class="panel-header">
                <h2>Registered Visitor Gallery</h2>
                <span id="gallery-count" style="font-size: 0.85rem; color: var(--text-muted);">0 Identities</span>
            </div>
            <div class="visitor-grid" id="visitor-gallery">
                <!-- Dynamically populated -->
            </div>
        </div>

        <!-- Audit Trail / Events Log -->
        <div class="panel">
            <div class="panel-header">
                <h2>Live Event Stream (events.log Audit)</h2>
                <span style="font-size: 0.85rem; color: var(--text-muted);">Auto-refreshing</span>
            </div>
            <div class="events-table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Snapshot</th>
                            <th>Timestamp</th>
                            <th>Event</th>
                            <th>Visitor ID</th>
                            <th>Frame #</th>
                        </tr>
                    </thead>
                    <tbody id="events-body">
                        <!-- Dynamically populated -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <footer>
        This project is a part of a hackathon run by <a href="https://katomaran.com" target="_blank">https://katomaran.com</a>
    </footer>

    <script>
        async function fetchStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();

                // Metrics
                document.getElementById('val-unique').innerText = data.unique_visitors;
                document.getElementById('val-entries').innerText = data.total_entries;
                document.getElementById('val-exits').innerText = data.total_exits;
                document.getElementById('gallery-count').innerText = `${data.visitors.length} Identities`;

                // Visitor Gallery
                const galleryEl = document.getElementById('visitor-gallery');
                if (data.visitors.length === 0) {
                    galleryEl.innerHTML = '<div style="color: var(--text-muted); padding: 1rem; grid-column: 1/-1;">No visitors registered yet. Start video stream to begin tracking.</div>';
                } else {
                    galleryEl.innerHTML = data.visitors.map(v => `
                        <div class="visitor-card">
                            <img class="visitor-img" src="${v.crop_url || 'https://via.placeholder.com/80'}" alt="${v.visitor_id}" onerror="this.src='https://via.placeholder.com/80?text=Face'">
                            <div class="visitor-id">${v.visitor_id}</div>
                            <div class="visitor-time">${v.first_seen.split('T')[1].slice(0, 8)}</div>
                        </div>
                    `).join('');
                }

                // Events Table
                const eventsBody = document.getElementById('events-body');
                if (data.recent_events.length === 0) {
                    eventsBody.innerHTML = '<tr><td colspan="5" style="text-align:center; color: var(--text-muted);">No events recorded yet.</td></tr>';
                } else {
                    eventsBody.innerHTML = data.recent_events.map(ev => `
                        <tr>
                            <td><img class="table-thumb" src="${ev.crop_url || 'https://via.placeholder.com/36'}" onerror="this.style.display='none'"></td>
                            <td>${ev.timestamp.replace('T', ' ').slice(0, 19)}</td>
                            <td><span class="event-badge ${ev.event_type === 'ENTRY' ? 'badge-entry' : 'badge-exit'}">${ev.event_type}</span></td>
                            <td style="color: var(--accent-primary); font-weight: 600;">${ev.visitor_id}</td>
                            <td>#${ev.frame_number}</td>
                        </tr>
                    `).join('');
                }

            } catch (err) {
                console.error("Dashboard refresh error:", err);
            }
        }

        fetchStats();
        setInterval(fetchStats, 2000);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/stats")
def stats():
    db_file = "data/faces.db" if os.path.exists("data/faces.db") else DB_PATH
    if SQLiteRepository is not None and os.path.exists(db_file):
        repo = SQLiteRepository(db_file)
        unique_visitors = repo.get_unique_visitor_count()
        raw_visitors = repo.get_all_faces()
        recent_events = repo.get_recent_events(limit=50)

        total_entries = 0
        total_exits = 0
        try:
            with repo._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT LOWER(event_type), COUNT(*) FROM events GROUP BY LOWER(event_type)")
                for etype, count in cursor.fetchall():
                    if etype == "entry":
                        total_entries = count
                    elif etype == "exit":
                        total_exits = count
        except Exception:
            pass

        visitors = []
        for v in raw_visitors:
            crop_path = v.first_crop_path
            crop_url = None
            if crop_path and os.path.exists(crop_path):
                rel = os.path.relpath(crop_path, "logs").replace(os.sep, "/")
                crop_url = f"/crops/{rel}"
            visitors.append({
                "visitor_id": v.face_id,
                "first_seen": v.first_seen,
                "last_seen": v.last_seen,
                "total_visits": v.visit_count,
                "crop_url": crop_url
            })

        formatted_events = []
        for ev in recent_events:
            img_path = ev.image_path
            crop_url = None
            if img_path and os.path.exists(img_path):
                rel = os.path.relpath(img_path, "logs").replace(os.sep, "/")
                crop_url = f"/crops/{rel}"
            formatted_events.append({
                "visitor_id": ev.face_id,
                "event_type": ev.event_type.upper(),
                "timestamp": ev.timestamp,
                "frame_number": ev.track_id,
                "crop_url": crop_url
            })

        return jsonify({
            "unique_visitors": unique_visitors,
            "total_entries": total_entries,
            "total_exits": total_exits,
            "visitors": visitors,
            "recent_events": formatted_events
        })

    return jsonify({
        "unique_visitors": 0,
        "total_entries": 0,
        "total_exits": 0,
        "visitors": [],
        "recent_events": []
    })

@app.route("/crops/<path:filename>")
def serve_crop(filename):
    return send_from_directory("logs", filename)

if __name__ == "__main__":
    print("[Dashboard] Launching web dashboard at http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
