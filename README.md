# Soft Skills Development Platform

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-brightgreen)

An AI-powered platform that helps users assess and improve critical soft skills like communication, leadership, and emotional intelligence through personalized feedback and interactive training modules.

## 🚀 Key Features

- **Skill Assessment**: Predicts current skill levels based on user inputs
- **Progress Tracking**: Visualizes improvement over time
- **Interactive Training**: Recommends personalized exercises
- **Speech Analysis**: (Future) Integration with speech recognition for interview practice

## 🛠️ Tech Stack

**Backend**:
- Python 3.11
- Flask (Web Framework)
- scikit-learn (Machine Learning)
- Pandas (Data Processing)

**Frontend**:
- HTML5/CSS3
- Bootstrap (Optional)

**Machine Learning**:
- Random Forest Regressor
- Feature Engineering for skill prediction

## 📂 Repository Structure
soft-skills/
├── static/ # CSS/JS assets
├── templates/ # HTML templates
│ └── index.html # Main interface
├── soft_skills_model.pkl # Trained ML model
├── app.py # Flask application
├── requirements.txt # Dependencies
└── README.md

text

## 🖥️ Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/soft-skills-platform.git
   cd soft-skills-platform
Create and activate virtual environment:

bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate    # Windows
Install dependencies:

bash
pip install -r requirements.txt
Run the Flask app:

bash
python app.py

run on local host

