#!/bin/bash
# This script can be used to prepare the deployment

echo "Preparing deployment for Render.com..."

# Check if Excel file exists
if [ ! -f "Base de datos estaciones SAMA.xlsx" ]; then
    echo "Warning: Excel file 'Base de datos estaciones SAMA.xlsx' not found."
    echo "Please upload this file to your repository or modify the code to work without it."
fi

echo "Deployment preparation complete!"
echo "To deploy on Render.com:"
echo "1. Push your code to GitHub"
echo "2. Connect your GitHub repo to Render.com"
echo "3. Create a new Web Service"
echo "4. Use these settings:"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: gunicorn Pronosticos:app.server"
echo "   - Environment: Python 3"
