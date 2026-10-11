#!/usr/bin/env bash
# purge_all_data.sh - Delete every job's logs, uploaded inputs and generated
# outputs from $VISION_DATA_DIR, and empty the job and upload records.
#
# Usage:  ./purge_all_data.sh [--dry-run] [--yes]
#   --dry-run   only show what would be deleted
#   --yes       do not ask for confirmation
#
# Deletes:   $VISION_DATA_DIR/jobs/<job_id>/ (one folder per job: args, logs,
#            output), everything in $VISION_DATA_DIR/jobs/_uploads/, and every
#            row of the `jobs` and `uploads` tables in jobs.db.
# Keeps:     jobs.db itself (and its schema, so a running gateway keeps
#            working), the _uploads folder, and everything else in the data
#            directory: envs/, weights/, wheels/, legacy/.
#
# Safety:    refuses to run while any job is queued or running (cancel them
#            first), and refuses unless the target folder is exactly
#            <data dir>/jobs. Deleted data cannot be recovered.
set -euo pipefail

DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --yes) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
[ -d "$DATA_DIR" ] || { echo "Data directory not found: $DATA_DIR" >&2; exit 1; }
DATA_DIR="$(cd "$DATA_DIR" && pwd -P)"
JOBS_DIR="$DATA_DIR/jobs"
UPLOADS_DIR="$JOBS_DIR/_uploads"
DB="$DATA_DIR/jobs.db"

# The folder being emptied must really be <data dir>/jobs, not a symlink to
# somewhere else.
[ -d "$JOBS_DIR" ] || { echo "Nothing to delete: $JOBS_DIR does not exist."; exit 0; }
if [ "$(cd "$JOBS_DIR" && pwd -P)" != "$DATA_DIR/jobs" ]; then
  echo "Refusing: $JOBS_DIR does not resolve to $DATA_DIR/jobs" >&2
  exit 1
fi

# db_query <sql>: run one statement against jobs.db with the system python3
# (stdlib sqlite3, no extra tools needed). Prints the first column of each row.
db_query() {
  python3 - "$DB" "$1" <<'EOF'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], timeout=30)
try:
    for row in conn.execute(sys.argv[2]):
        print(row[0])
    conn.commit()
finally:
    conn.close()
EOF
}

IN_FLIGHT=0
if [ -f "$DB" ]; then
  IN_FLIGHT="$(db_query "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')")"
fi
if [ "$IN_FLIGHT" != "0" ]; then
  echo "Refusing: $IN_FLIGHT job(s) are still queued or running. Cancel them first (DELETE /v1/jobs/<job_id>)." >&2
  exit 1
fi

JOB_FOLDERS="$(find "$JOBS_DIR" -mindepth 1 -maxdepth 1 -type d ! -name _uploads | wc -l)"
OTHER_ENTRIES="$(find "$JOBS_DIR" -mindepth 1 -maxdepth 1 ! -type d | wc -l)"
UPLOAD_FILES=0
[ -d "$UPLOADS_DIR" ] && UPLOAD_FILES="$(find "$UPLOADS_DIR" -mindepth 1 | wc -l)"
SIZE="$(du -sh "$JOBS_DIR" 2>/dev/null | cut -f1)"

echo "Data directory: $DATA_DIR"
echo "Will delete:    $JOB_FOLDERS job folder(s), $UPLOAD_FILES uploaded file(s), $OTHER_ENTRIES loose file(s) in $JOBS_DIR ($SIZE)"
echo "                and all job and upload records in $DB"
echo "Will keep:      jobs.db (schema), envs/, weights/, wheels/, legacy/"

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run: nothing was deleted."
  exit 0
fi

if [ "$ASSUME_YES" != "1" ]; then
  read -r -p "Type 'yes' to delete all of it (cannot be undone): " ANSWER
  [ "$ANSWER" = "yes" ] || { echo "Cancelled."; exit 1; }
fi

# Check again right before deleting: a job may have been submitted while the
# confirmation prompt was waiting.
if [ -f "$DB" ]; then
  IN_FLIGHT="$(db_query "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')")"
  [ "$IN_FLIGHT" = "0" ] || { echo "Refusing: a job was submitted meanwhile ($IN_FLIGHT in flight)." >&2; exit 1; }
fi

find "$JOBS_DIR" -mindepth 1 -maxdepth 1 ! -name _uploads -exec rm -rf -- {} +
[ -d "$UPLOADS_DIR" ] && find "$UPLOADS_DIR" -mindepth 1 -exec rm -rf -- {} +
# The gateway saves every upload into this folder; make sure it exists.
mkdir -p "$UPLOADS_DIR"

if [ -f "$DB" ]; then
  db_query "DELETE FROM jobs" >/dev/null
  db_query "DELETE FROM uploads" >/dev/null
fi

echo "Done: deleted $JOB_FOLDERS job folder(s) and $UPLOAD_FILES uploaded file(s); job and upload records cleared."
