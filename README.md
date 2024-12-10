# Predictive Chatbot for Real-Time Pharmaceutical Demand Management

This project aims to develop an AI-driven chatbot to forecast pharmaceutical demand, specifically for the distribution of medical products in Tunisia. The chatbot utilizes advanced machine learning algorithms, including time-series forecasting models, and interacts with real-time data to provide precise analyses and forecasts for various sectors in the pharmaceutical market. It is built using Python (Flask for the backend), Flutter for the frontend (mobile application), and integrates multiple machine learning tools for improved accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributors](#contributors)

## Project Overview

This project develops a chatbot capable of forecasting pharmaceutical product quantities for the year 2024 based on historical data from major distributors in Tunisia. The system includes:
- Data collection and preprocessing
- Time-series forecasting using models like Exponential Smoothing
- Integration of a chatbot interface to interact with the data
- Deployment as a web and mobile application

The main objective is to help the pharmaceutical supply chain managers by providing forecasts for specific products, sectors, and regions to optimize decision-making.

## Key Features
- **Forecasting:** Uses Exponential Smoothing and other forecasting models to predict pharmaceutical demand.
- **Chatbot Interface:** Interactive interface built in Flask and Flutter that responds to user queries about forecasted data.
- **Data Analytics:** Provides detailed insights based on real-time and historical data.
- **Multi-Platform Deployment:** Available as both a web application (via Flask) and a mobile application (via Flutter).
- **Email Authentication:** Secure login and registration with email verification.
- **User-Friendly Interface:** Simple, intuitive interface for interacting with the system, designed for both technical and non-technical users.

## Technologies Used
- **Backend:** Python, Flask, SQLAlchemy
- **Frontend:** Flutter, HTML, CSS, JavaScript
- **Machine Learning:** scikit-learn, pandas, NumPy, statsmodels
- **Database:** XAMPP (MySQL)
- **Deployment:** Heroku (for web app), Android Studio (for mobile app)
- **Version Control:** Git, GitHub
- **Libraries:** bcrypt (for password hashing), Flask-WTF (for form handling)

## Installation Instructions

Follow these steps to get the project running locally:

### Clone the Repository:
    ```bash
    git clone https://github.com/oussema-fdhila/Predictive-Chatbot-for-Real-Time-Pharmaceutical-Demand-Management.git
    cd forecast-chatbot
    
### Set up the Backend:    

1. Create a virtual environment:
   ```bash
   python -m venv venv
2. Activate the virtual environment:
   - **On Windows**:
     ```bash
     venv\Scripts\activate
   - **On Mac/Linux**:
     ```bash
     source venv/bin/activate
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
4. Set up the database:
   - Configure the MySQL database with XAMPP.
   - Update the database credentials in the config.py file.
5. Run the backend:
   ```bash
   python app.py

### Set up the Frontend:
Run the Flutter application on your emulator or device:
    ```bash
    flutter run

## Usage
   - Once the backend and frontend are set up, you can access the web version of the chatbot at http://localhost:5000.
   - The mobile app can be tested by launching it from Android Studio.

## Folder Structure
forecast-chatbot/
│
├── backend/                # Backend code (Flask)
│   ├── app.py              # Main Flask app
│   ├── config.py           # Database configurations
│   ├── models.py           # SQLAlchemy models
│   ├── templates/          # HTML templates for the web app
│   └── static/             # Static files (CSS, JS)
│
├── frontend/               # Mobile app code (Flutter)
│   ├── lib/                # Dart code for the Flutter app
│   ├── assets/             # Images, fonts, etc.
│   └── android/            # Android-specific files
│
└── requirements.txt        # Python dependencies

## Contributors
   - **Oussama Fdhila** - Project Lead - Data Scientist
   - **Mme. Dalel Ayed Lakhal** - Technical Supervisor
   - **M. Lassaad Saidani** - Academic Supervisor

  



