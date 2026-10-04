#!/usr/bin/env bash
# Starts the Wheat Advisor app. Opening the page loads the detector onto the GPU once; it then stays there
# until you press Ctrl+C. Run it with the Python environment that has the requirements installed.
cd "$(dirname "$0")" || exit 1
echo "Starting Wheat Advisor at http://localhost:8501 ..."
exec python -m streamlit run app.py
