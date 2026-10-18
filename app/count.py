import argparse
import sys
from app.config import load_config
from app.storage.sqlite_repo import SQLiteRepository


def get_unique_count(db_path: str = "data/faces.db") -> int:
    """Returns the total number of distinct registered visitor identities."""
    repo = SQLiteRepository(db_path)
    count = repo.get_unique_visitor_count()
    repo.close()
    return count


def print_visitor_summary(db_path: str = "data/faces.db") -> None:
    """Prints a detailed, formatted summary of all unique visitors and recent events."""
    repo = SQLiteRepository(db_path)
    unique_count = repo.get_unique_visitor_count()
    faces = repo.get_all_faces()
    recent_events = repo.get_recent_events(limit=10)
    repo.close()

    print("=" * 60)
    print("         UNIQUE VISITOR COUNTER & AUDIT SUMMARY")
    print("=" * 60)
    print(f"Database Path           : {db_path}")
    print(f"TOTAL UNIQUE VISITORS   : {unique_count}")
    print("-" * 60)

    if faces:
        print("Registered Identity Roster:")
        for idx, face in enumerate(faces, 1):
            print(f"  {idx:3d}. [{face.face_id}] | First Seen: {face.first_seen} | Last Seen: {face.last_seen} | Visits: {face.visit_count}")
    else:
        print("  No visitors registered in the database yet.")

    print("-" * 60)
    if recent_events:
        print("Recent Lifecycle Events:")
        for ev in recent_events:
            print(f"  [{ev.timestamp}] {ev.event_type.upper():5s} | Face: {ev.face_id} (Track #{ev.track_id}) -> {ev.image_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Query Unique Visitor Count from Database")
    parser.add_argument("--db", type=str, default=None, help="Path to faces SQLite database")
    parser.add_argument("--summary", action="store_true", help="Print full visitor and event summary table")
    args = parser.parse_args()

    if args.db:
        db_path = args.db
    else:
        cfg = load_config("config.json")
        db_path = cfg.storage.db_path

    if args.summary:
        print_visitor_summary(db_path)
    else:
        count = get_unique_count(db_path)
        print(f"UNIQUE_VISITOR_COUNT={count}")


if __name__ == "__main__":
    main()
