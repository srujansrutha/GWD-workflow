@echo off
rem Starts the Wheat Head Counter. Opening the page loads the model onto the GPU once; it then stays there
rem until you close this window or press Ctrl+C.
cd /d "%~dp0"
echo Starting Wheat Head Counter at http://localhost:8501 ...
start "" /b cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"
"..\.venv\Scripts\python.exe" -m streamlit run app.py
pause
