# Adons_Final_Year_IT_Project

This repository contains the implementation of a comparative study evaluating three facial recognition models (ArcFace, Facenet, and Dlib) for student authentication in online learning environments.
Project Overview
This project aims to identify the most effective facial recognition solution for educational settings by comparing recognition confidence, processing speed, and reliability across different models. The research was conducted as part of a BSc in Information Technology at the University of Hertfordshire.
Key Components
Main Scripts
* all_models.py: Core testing and evaluation script that implements facial recognition using DeepFace framework
* database_operations.py: PostgreSQL database operations for storing and analyzing test results
* dlib_face_recognition.py: Early implementation using face_recognition library (proof of concept)
* camera_test.py: Initial camera testing script
Database Structure
The system uses a PostgreSQL database with the following tables:
* People
* Models
* Model_aggregate_stats
* Recognition_tests
* Failed_tests
Installation Guide (MacOS)
1. Install Python 3.12
2. Install Miniforge3 (conda environment manager) curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
3. bash Miniforge3-MacOSX-x86_64.sh
4. 
    * Follow prompts to complete installation
    * Verify with conda --version
5. Create and activate conda environment conda create -n all_models python=3.12
6. conda activate all_models
7. 
8. Install required packages pip install opencv-python
9. pip install deepface
10. pip install psycopg2-binary
11. 
Troubleshooting dlib installation
When installing the face_recognition library or using dlib directly, many users encounter installation issues. Here's how to resolve them:
1. Install cmake (critical dependency for dlib) conda install -c conda-forge cmake
2. 
3. Install dlib directly conda install -c conda-forge dlib
4. 
5. # Install cmake and required libraries
6. brew install cmake
7. brew install jpeg libpng
8. 
9. # Install dlib with specific compiler flags
10. CFLAGS="-stdlib=libc++" pip install dlib
11. 
12. Install XCode Command Line Tools if needed: xcode-select --install
13. 
14. PostgreSQL Setup
    * Install PostgreSQL if not already installed
    * Create a database named 'facial_recognition_data'
    * Update DB_CONFIG in database_operations.py with your credentials
Usage
1. Activate the environment conda activate all_models
2. 
3. Run the main script python all_models.py
4. 
    * The script will prompt for model selection and participant ID
    * Testing will run for 15 seconds
    * Results will be stored in the database

Research Methodology
The system evaluates three key metrics:
1. Confidence Scores: Certainty of facial recognition match
2. Processing Time: Speed of recognition in seconds
3. Recognition Rate: Percentage of successful recognitions
Testing was conducted with 15 participants under controlled conditions, followed by structured interviews to assess user experience and acceptance factors.
Ethics Approval
This study was approved by the University of Hertfordshire Ethics Committee (UH Protocol Number: 0207 2025 Jan HSET).
Results Summary
* Dlib: Highest confidence scores (97%) and fastest processing (2.216s) but significant variability (SD=15.37%)
* Facenet: Most consistent performer (SD=6.94%) with strong recognition rates (91.74%)
* ArcFace: Balanced performance with moderate variability (SD=11.73%)
Based on these findings, Facenet is recommended for educational environments where consistency and reliability are crucial.




