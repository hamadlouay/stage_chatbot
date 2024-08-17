# AI Chatbot Application

This repository contains an AI chatbot application built with Flask, Groq API, and LangChain. The chatbot is designed to answer questions specifically related to "INCLUSION ÉCONOMIQUE DES JEUNES (IEJ)" and can respond in English, French, or Arabic. The application is integrated with a web interface for interaction.

## Table of Contents

- Installation
- Usage
- Files Description
- Customization
- Docker Deployment
- Troubleshooting
- License

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.9+
- Flask
- Docker (for containerized deployment)

### Clone the Repository

```bash
git clone https://github.com/hamadlouay/stage_chatbot
cd stage_chatbot
```
## Create a Virtual Environment
```python
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
# Install Dependencies
```bash
pip install -r requirements.txt
```
# Set Up Environment Variables
Create a .env file in the root directory of the project and add your Groq API key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
## Load Vector Store
Ensure you have the vector_store.pkl file in the project directory. This file contains the precomputed vector store for your chatbot's knowledge base.


## Start the Flask application:

```bash

python app.py
```
Open your web browser and go to http://localhost:5000 to interact with the chatbot.

## Docker Deployment
Build the Docker image:
```bash

docker build -t ai-chatbot .
```
Run the Docker container:

```bash

docker run -d -p 5000:5000 ai-chatbot
```
## Files Description
app.py: Main Flask application file that handles the chat interaction.
requirements.txt: Contains all Python dependencies needed to run the application.
Dockerfile: Dockerfile for containerizing the application.
templates/chat.html: HTML template for the chatbot interface.
static/style.css: CSS styles for the chatbot interface.
.env: Environment variables (not included in the repository, must be created).
vector_store.pkl: Pickle file containing the vector store for the chatbot.
## Customization
You can customize the chatbot's appearance by modifying the style.css file located in the static directory. Additionally, you can adjust the chatbot's prompt behavior and responses by editing the system message in the app.py file.

## Troubleshooting
If you encounter issues during installation or while running the application, here are some common solutions:

Missing Dependencies: Ensure that all dependencies are installed by running pip install -r requirements.txt.
API Key Issues: Verify that your Groq API key is correctly set in the .env file.
Docker Issues: Make sure Docker is installed and running properly on your system.
## License
This project is licensed under the MIT License.

