#!/bin/bash
echo "========================================================="
echo "🚀 INITIATING HYBRID SOC AGENT DASHBOARD..."
echo "========================================================="
echo ""
echo "Loading environment and launching Streamlit server..."

# Use the absolute path to ensure it uses the environment with networkx installed
/Library/Frameworks/Python.framework/Versions/3.11/bin/streamlit run app.py
