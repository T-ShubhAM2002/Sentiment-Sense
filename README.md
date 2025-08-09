# Sentiment Sense

Live demo: [Sentiment Sense-knxj.onrender.com](https://Sentiment Sense-knxj.onrender.com/)

## Description

Sentiment Sense is a web application that helps users manage their mental health and well‑being. It provides tools for detecting stress levels and fostering resilience through personalized recommendations and motivational content. Machine learning models analyze user input to provide insights into a user's emotional state.

## Table of Contents

- [Live Demo](#Sentiment Sense)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. (Recommended) Create and activate a virtual environment:
   - Windows (PowerShell):
     ```bash
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application locally:

- Using Flask CLI (if a Flask entry is configured):

  ```bash
  set FLASK_APP=app.py    # Windows
  export FLASK_APP=app.py # macOS/Linux
  flask run
  ```

- Or run the app module directly (if supported by the project):
  ```bash
  python app.py
  ```

Then open `http://localhost:5000` in your browser.

## Features

- Mental health detector: Analyzes user input to detect stress levels and provides personalized feedback.
- Emotional wellness companion: Offers motivational quotes and resources to help users build resilience.
- User-friendly interface: Intuitive design for easy navigation and interaction.
- Responsive layout: Works across desktop and mobile.

## Tech Stack

- Python: Backend development
- Flask: Web framework
- HTML/CSS/JavaScript: Frontend
- Tailwind CSS: Utility-first styling
- Scikit-learn: Machine learning for predictive analysis

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature`
3. Make changes and commit: `git commit -m "Add feature"`
4. Push to your branch: `git push origin feature`
5. Open a pull request
