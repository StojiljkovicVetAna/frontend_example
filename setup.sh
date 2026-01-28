#!/bin/bash

# Age Morphing App Setup Script

echo "Setting up Age Morphing App..."

# Install dependencies
pip3 install -r requirements.txt

# Download dlib's face landmark predictor
echo "Downloading facial landmark detector..."
curl -L -O https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

echo "Setup complete!"
echo ""
echo "To run the app:"
echo "  streamlit run age_morph_app.py"