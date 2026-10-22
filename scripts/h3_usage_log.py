#!/usr/bin/env python3
"""h3_usage_log.py - work with the H3 usage log ($VISION_DATA_DIR/metrics/h3_jobs.jsonl).

The gateway appends one numbers-only line to that file whenever an H3 job finishes (see
gateway/services/h3/usage_log.py for what a line holds). It is the data to fit the sizing
numbers in gateway/services/h3/config.py from.

Usage (run with the gateway's Python, so no extra packages are needed):

  $VISION_DATA_DIR/envs/gateway/bin/python scripts/h3_usage_log.py backfill [--dry-run]
      Add a line for every finished H3 job that is in jobs.db and still has its folder but is
      not in the log yet (jobs from before the log existed). Such a line is marked
      "source": "backfill" and holds what run.log and args.json still say, plus sacct.
      Nothing already in the log is changed; --dry-run only counts.

  $VISION_DATA_DIR/envs/gateway/bin/python scripts/h3_usage_log.py csv [--output FILE]
      Flatten the log into a spreadsheet: one row per job, one column per number
      (nested names joined with dots, e.g. sizing.denoise_rows, metrics.run.timings_s.denoise).
      Prints to the terminal unless --output is given; an existing file is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))

from common import job_store  # noqa: E402
from services.h3 import usage_log  # noqa: E402


def backfill(dry_run: bool) -> int:
    already_logged = usage_log.logged_job_keys()
    added = 0
    skipped_logged = 0
    skipped_gone = 0
    for row in job_store.list_finished_jobs("h3:"):
        if usage_log.job_key(row) in already_logged:
            skipped_logged += 1
        elif not Path(row["job_dir"]).is_dir():
            skipped_gone += 1  # nothing left to read (deleted by retention or incognito)
        else:
            if not dry_run:
                usage_log.append(usage_log.build_record(row, source="backfill"))
            added += 1
    verb = "would add" if dry_run else "added"
    print(f"{verb} {added} job(s) to {usage_log.LOG_PATH}; skipped {skipped_logged} already logged, {skipped_gone} without a job folder.")
    return 0


def read_records() -> list[dict]:
    records = []
    try:
        with usage_log.LOG_PATH.open(encoding="utf-8") as log_file:
            for line in log_file:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue  # a half-written last line
    except OSError as error:
        raise SystemExit(f"Cannot read {usage_log.LOG_PATH}: {error}")
    return records


def write_csv(output: str | None) -> int:
    rows = [usage_log.flatten(record) for record in read_records()]
    columns: list[str] = []
    for row in rows:
        columns += [name for name in row if name not in columns]

    def write(stream) -> None:
        writer = csv.DictWriter(stream, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)

    if output is None:
        write(sys.stdout)
    else:
        try:
            with open(output, "x", newline="", encoding="utf-8") as stream:  # "x": never overwrite
                write(stream)
        except FileExistsError:
            raise SystemExit(f"{output} already exists; choose another file name.")
        print(f"wrote {len(rows)} row(s), {len(columns)} column(s) to {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Work with the H3 usage log.")
    commands = parser.add_subparsers(dest="command", required=True)
    backfill_parser = commands.add_parser("backfill", help="add lines for finished jobs that are not in the log yet")
    backfill_parser.add_argument("--dry-run", action="store_true", help="only count, write nothing")
    csv_parser = commands.add_parser("csv", help="flatten the log into a spreadsheet")
    csv_parser.add_argument("--output", help="file to create (default: print to the terminal)")
    args = parser.parse_args()
    return backfill(args.dry_run) if args.command == "backfill" else write_csv(args.output)


if __name__ == "__main__":
    sys.exit(main())
