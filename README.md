# Opportunity_Application_Ranker_Deployment_Stack
This repository contains the code and resources for the cloud deployed Streamlit based Application developed as part of the larger Opportunity Applcation Ranker project. The application provides an interactive and user-friendly interface to explore the project's end results. Users will be able to view the suitable jobs obtained through optimized TFIDF weighted Word2Vec model when the candidate details are entered. 

The cloud deployed application can be accessed using the link: https://opp-app-matcher-938827348329.asia-south1.run.app

# Table of Contents

1. Introduction
2. Project Features
3. Prerequisites
4. Setup Instructions
5. How to Run
6. Folder Structure
7. Screenshots
9. Future Work
10. Contributing
11. License
12. Contact


# Introduction

This deployed application serves as a front-end interface for the Opportunity Application Ranker project, making it easier for users to see the end results the analysis. Users can enter the details of the candidates to get the list of close jobs applicable to the candidate. Thus, demonstrating the Opportunity Application Project. Out of the three models viz. TFIDF weighted Word2Vec, BERT based uncased and distilBERT based uncased, the TFIDF weighted Word2Vec was determined to the optimized model. The details of this testing are available in a separate repository and can be accessed using the link: https://github.com/rathishsekhar/Opportunity_Application_Ranker . 

# Project Features
- Data Exploration: Upload your own candidate profile to view the matching jobs. 
- Optimized Model: TFIDF weighted model has been choosen as the optimized model.

# Prerequisites
Make sure you have the following installed on your machine:

Python: Version 3.6 or higher <br>
Streamlit: Latest version<br>
Google Cloud SDK available at https://cloud.google.com/sdk/?hl=en <br>
Other dependencies specified in the requirements.txt file <br>

Owing to the large size of the data, users are required to contact the author to get the necessary data for running the app. 

# Setup Instructions
- Clone the Repository: git clone https://github.com/rathishsekhar/Opportunity_Application_Matching_App
- Create a Virtual Environment: python -m venv venv
- Activate the Virtual Environment:
    * On Windows: venv\Scripts\activate
    * On macOS/Linux: source venv/bin/activate
- Install Dependencies: pip3 install -r requirements.txt
- Set up a project in google cloud console and obtain the project id
- Initiate google cloud SDK: gcloud init
- Enable APIs: 
        gcloud services enable run.googleapis.com
        gcloud services enable cloudbuild.googleapis.com
- Build Docker Image: docker build -t opp_app # Users may change the application's name
- Tag Docker image: docker tag opp_app gcr.io/project_id/your-app-name # Users are required to enter the project_id of the project created here
- Push Docker image to google cloud registry: docker push gcr.io/project_id/opp_app # Enter the project_id
- Run Google Cloud Run: 
        gcloud run deploy your-service-name --image gcr.io/your-project-id/your-app-name --platform managed 


# How to Run
After deployment, the SDK will provide a link. Copy and paste in the browser.

# Folder Structure
```plaintext
streamlit_app_all_models
│
├── app.py                   # Main Streamlit app script
├── app.yaml                 # Configuration file
├── Dockerfile               # Docker commands file
├── data                     # Folder for default data files
├── resources                # Custom scripts for apps running
├── src                      # Images, logos, or other static assets
├── getter/                  # Scripts for fetching data
│   ├── preprocessing/       # Preprocessing scripts
│   ├── featurization/       # Feature engineering scripts
│   └── model training/      # Model training scripts
├── utils                    # Helper functions for the app
├── requirements.txt         # List of dependencies
└── README.md                # Documentation for this folder
```


# Future Work
**Enhancements** : Potential features include additional models, improved visualizations, or support for larger datasets.
**User Feedback**: We welcome suggestions and feedback for improving the app’s functionality and user experience.

# Contributing
We welcome contributions to improve this Streamlit application! Please follow the contribution guidelines in the main project's README.md and submit a pull request.

# License

This project is licensed under the MIT License. See the LICENSE file in the main project directory for more details.

# Contact
For any questions or feedback, please contact:<br>
Name: Rathish Sekhar <br>
Email: rathishsekhar@gmail.com

