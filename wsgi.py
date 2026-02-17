#!/usr/bin/env python3
"""Production entry point for gunicorn (Railway / Render).

Usage:
    gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1
"""

import logging
import os

# Ensure logs/ and data/ directories exist
for d in ('logs', 'data'):
    os.makedirs(os.path.join(os.path.dirname(__file__), d), exist_ok=True)

from dashboard import app, start_scheduler, _get_tracker, _get_class_cfg, _ensure_logging

# Set up logging
_ensure_logging()

# Seed departments on startup + one-time conversation backfill
tracker = _get_tracker()
class_cfg = _get_class_cfg()
tracker.seed_default_departments(class_cfg.get('discipline_keywords', {}))
if not tracker.get_setting('conversations_backfilled'):
    logging.getLogger(__name__).info("Running one-time conversation backfill...")
    tracker.backfill_conversations()
    tracker.set_setting('conversations_backfilled', '1')
tracker.close()

# Start the background email scanner
start_scheduler()

logging.getLogger(__name__).info("Production server ready")
