#!/usr/bin/env bash
# Reproduce the full pipeline. Usage:
#   ./run_all.sh "Your Name your_email@example.com" YOUR_WRDS_USERNAME
set -euo pipefail
UA="$1"; WRDS_USER="$2"
python3 src/edgar_13d_activists.py --user-agent "$UA" --start 2010   # Phase 1: 13D campaigns from EDGAR
python3 src/event_study.py --wrds-user "$WRDS_USER"                   # Phase 2: stock reaction to filings
python3 src/build_panel.py --wrds-user "$WRDS_USER"                   # Phase 3a: firm-year features + labels
python3 src/train_model.py                                            # Phase 3b: walk-forward model + current scores
python3 src/post_comps.py --wrds-user "$WRDS_USER"                    # Case study: comparable companies
python3 src/post_history.py --wrds-user "$WRDS_USER"                  # Case study: Post's multiple vs. peers over time
