#!/bin/bash
# Paper figures F0-F8 from the shipped results/ (CPU, about a minute). Writes figures/.
source "$(dirname "$0")/common.sh"
python paper/figs_paper.py && python paper/f5_paper.py && python paper/f8_lines.py
