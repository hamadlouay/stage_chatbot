import os
import pickle
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

# Custom Document class
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Save the vector store to disk
def save_vector_store(vector_store, path):
    with open(path, 'wb') as f:
        pickle.dump(vector_store, f)

# Function to fetch HTML content using Selenium
def fetch_html_content_with_selenium(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run headless Chrome
    service = Service("C:/Users/hamad louay/Downloads/chromedriver-win64 (1)/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)
    try:
        # Wait for content to load, adjust the selector as needed
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
        html_content = driver.page_source
    finally:
        driver.quit()
    return html_content

# Function to process local URLs
def process_local_urls(urls):
    all_docs = []
    for url in urls:
        try:
            print(f"Fetching content from local URL: {url}")
            html_content = fetch_html_content_with_selenium(url)

            # Debug: Print out the beginning of the HTML content
            print(f"HTML content from {url} (first 500 characters): {html_content[:500]}")

            soup = BeautifulSoup(html_content, "html.parser")

            # Extract the main content, depending on your HTML structure
            content = soup.get_text(separator=' ', strip=True)
            if not content:
                raise ValueError("No content extracted from HTML.")

            document = Document(
                page_content=content,
                metadata={"source": url}
            )
            documents = [document]
            print(f"Loaded 1 document from {url} using fallback method")

            # Split the content into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)
            print(f"Split into {len(docs)} chunks from {url}")
        except Exception as e:
            print(f"Failed to load or process {url} using fallback method: {e}")
    return all_docs

# Function to process non-local URLs
def process_non_local_urls(urls):
    all_docs = []
    for url in urls:
        try:
            print(f"Fetching content from non-local URL: {url}")
            loader = WebBaseLoader(url)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents from {url}")

            # Split the content into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)
            print(f"Split into {len(docs)} chunks from {url}")
        except Exception as e:
            print(f"Failed to load or process {url}: {e}")
    return all_docs

# Function to process and embed custom text
def process_custom_text(text):
    # Create document from the provided text
    document = Document(
        page_content=text,
        metadata={"source": "custom_text"}
    )

    # Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents([document])
    
    return docs

# Load and process website content from sitemap
def load_and_process_website_content_from_sitemap(vector_store_path, custom_text=None):
    local_urls = [
        "http://172.16.1.32/public/portail/iej",
        "http://172.16.1.32/public/portail/iej/about",
        "http://172.16.1.32/public/portail/iej/activites",
        "http://172.16.1.32/public/portail/iej/auth/signup",
        "http://172.16.1.32/public/portail/iej/faq",
        "http://172.16.32/public/portail/iej/contact"
    ]

    non_local_urls = []  # Add your non-local URLs here if needed

    # Process both local and non-local URLs
    local_docs = process_local_urls(local_urls)
    non_local_docs = process_non_local_urls(non_local_urls)

    all_docs = local_docs + non_local_docs
    custom_text='''
Le Gouvernement du Cameroun avec l’appui de la Banque mondiale lance un programme d’Inclusion Economique des Jeunes (IEJ) en faveur de 65.000 jeunes femmes et hommes âgés de 18 à 35 ans en trois (03) phases sur la période 2024 à 2028.
(i) La phase 1 en 2024 au profit de 15.000 bénéficiaires.
(ii) La phase 2 en 2025/2026 au profit de 20.000 bénéficiaires.
(iii) La phase 3 en 2026/2027 au profit de 30.000 bénéficiaires.


Pour être éligible, chaque candidat intéressé par le programme d’inclusion économique devrait être un jeune pauvre et vulnérable âgé de 18 à 35 ans et avoir l’un des profils ci-après :

✔

Il faut résider dans l’un des chefs-lieux des 10 régions du Cameroun notamment :

- La phase 1 concerne les villes de Maroua, Ebolowa, Douala et Yaoundé.

- La phase 2 concerne les villes de Maroua, Ebolowa, Douala, Yaoundé, Bafoussam, Garoua, Ngaoundéré et Bertoua.

- La phase 3 concerne les villes de Maroua, Ebolowa, Douala, Yaoundé, Bafoussam, Garoua, Ngaoundéré, Bertoua, Bamenda et Buea.

✔

Etre un travailleur du secteur informel menant une Activité Génératrice de Revenu (AGR) pour son propre compte et prêt à être accompagné dans son AGR.

✔

Etre déscolarisé au chômage et intéressé à mener une AGR.

✔

Etre un déplacé interne.

✔

Etre une femme menant une AGR ou pas et intéressée par le programme IEJ.

✔

Etre motivé et intéressé à démarrer ou à développer une activité entrepreneuriale.
Présentation du programme IEJ
Le programme d’Inclusion Economique des Jeunes (IEJ) qui est mis en œuvre au Cameroun avec l’appui financier de la Banque mondiale, cible en milieu urbain, les jeunes âgés de 18 à 35 ans pauvres et vulnérables ayant au moins l’un des profils ci-après :
(i) Etre un travailleur du secteur informel menant une Activité Génératrice de Revenu (AGR) pour son propre compte.
(ii) Etre au chômage.
(iii) Etre un déplacé interne sans activité.
(iv) Etre une femme menant une AGR ou pas mais intéressée par le programme.
L’objectif du programme vise à soutenir les jeunes motivés à entreprendre une AGR à :
(i) Développer des activités génératrices de revenus (AGR).
(ii) Accroître leur productivité.
(iii) Renforcer leur résilience économique.
Chaque jeune retenu après l’inscription en ligne via le formulaire recevra au total une subvention de 275.000 FCFA.
Activité 1
Sensibilisation communautaire (petit film) sur la réussite d’une AGR. Il s’agit de la diffusion d’un petit film qui montre la réussite d’un jeune qui a pu sortir d’une situation d’extrême pauvreté à travers l’initiation d’une activité génératrice de revenus.
Formations en compétences de vie (ACV) et en gestion de base (GERME). Il s’agit de deux (02) types de formations dispensées par groupe chacune l’une sur le développement des compétences de vie (ACV) et l’autre sur les modules de la microentrepreunariat
Sensibilisation à la constitution et à la gestion de l’épargne individuelle et/ou groupe. Il s’agit des séances de sensibilisation à la constitution de l’épargne individuelle et/ou groupe. L’épargne sera vivement encouragée et permettra à chaque bénéficiaire de renforcer son autonomie financière, d’acquérir des biens productifs et de diversifier ses sources de revenus à travers le financement d’autres AGR.
Activité 2
Formations en compétences de vie (ACV) et en gestion de base (GERME). Il s’agit de deux (02) types de formations dispensées par groupe chacune l’une sur le développement des compétences de vie (ACV) et l’autre sur les modules de la microentrepreunariat.Activité 2
Formations en compétences de vie (ACV) et en gestion de base (GERME). Il s’agit de deux (02) types de formations dispensées par groupe chacune l’une sur le développement des compétences de vie (ACV) et l’autre sur les modules de la microentrepreunariat.'''
    if custom_text:
        custom_docs = process_custom_text(custom_text)
        all_docs.extend(custom_docs)

    if not all_docs:
        print("No documents were loaded and processed.")
        return

    print(f"Total documents loaded and processed: {len(all_docs)}")

    # Create embeddings for the chunks
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize the FAISS vector store
    vector_store = FAISS.from_documents(all_docs, embeddings)
    save_vector_store(vector_store, vector_store_path)
    print(f"Vector store saved to {vector_store_path}")

# Parameters
vector_store_path = 'vector_store.pkl'
custom_text = "This is some custom text that should be processed and added to the vector store."

# Generate and save the vector store
load_and_process_website_content_from_sitemap(vector_store_path, custom_text)
