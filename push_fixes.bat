@echo off
echo ================================================
echo   AgriAssist+ — Push fixes to GitHub
echo   (fixes web app link not opening)
echo ================================================
echo.

cd /d D:\Agri_Assist

echo Step 1: Checking git status...
git status

echo.
echo Step 2: Adding all changed files...
git add .gitignore whatsapp_bot.py app.py rag_engine.py plant_disease_cnn.py requirements.txt

echo.
echo Step 3: Committing...
git commit -m "fix: web app URL formatting in WhatsApp messages + gitignore fixes so models reach Streamlit Cloud"

echo.
echo Step 4: Pushing to GitHub...
git push origin main

echo.
echo ================================================
echo   Done! Streamlit Cloud will auto-redeploy.
echo   Your app URL: https://agri-assist481.streamlit.app
echo   It takes about 2-3 minutes to redeploy.
echo ================================================
pause
