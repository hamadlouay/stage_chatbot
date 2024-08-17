import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import WebBaseLoader


# Custom Document class
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Save the vector store to disk
def save_vector_store(vector_store, path):
    with open(path, 'wb') as f:
        pickle.dump(vector_store, f)


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
def load_and_process_website_content_from_sitemap(vector_store_path, custom_text=None, nonlocalurl=[]):
    non_local_urls =nonlocalurl  # Add your non-local URLs here if needed

    # Process  non-local URLs
    
    non_local_docs = process_non_local_urls(non_local_urls)

    all_docs =non_local_docs
    custom_text=custom_text
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
    print(all_docs)

# Parameters
non_local_urls = [ 'https://pfs-aie.cm/public/portail/iej']
vector_store_path = 'vector_store.pkl'
custom_text_iej = '''
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
custom_text_for_cpa='''Le Gouvernement du Cameroun avec l’appui de la Banque Mondiale lance à travers le Projet filets sociaux adaptatifs et d’inclusion économique, un programme du concours des plans d’affaires (CPA). Le dépôt des candidatures est ouvert à partir du 28 avril au 3 mai 2024 pour les jeunes entrepreneurs âgés de 15 à 35 ans.


CE PROGRAMME VOUS CONCERNE, BIEN VOULOIR REMPLIR CE
Dans sa Stratégie Nationale de Développement 2020-2030 (la SND30) pour la transformation structurelle et le développement inclusif qui constitue le vecteur de la recherche de la croissance économique et de la redistribution des fruits de la croissance jusqu'aux couches les plus vulnérables de la population, le Gouvernement s’est engagé Dans le domaine de l’emploi, à promouvoir des politiques qui favorisent le développement des activités productives, la création d’emplois décents, l’entrepreneuriat, la créativité et l’innovation et qui stimulent la croissance des micro, petites et moyennes entreprises tout en facilitant leur intégration dans le secteur formel.
Objectifs du programme CPA
Le programme CPA permet d’apporter des appuis aux jeunes âgés de 18 à 35 ans, entrepreneurs dans les zones urbaines ayant déjà initié une micro, petite et moyenne entreprise (MPME) informelle ou géré une petite et moyenne entreprise (PME) formelle en phase de démarrage (entre 1 à 3 ans) et présentant un potentiel de croissance significatif et un potentiel de création d'emplois dans les secteurs porteurs et prioritaires pour le gouvernement définis dans la SND30.
Age
18 à 35 ans
Nationalité
Etre citoyen camerounais résident et exerçant son activité dans l’une des zones urbaines retenues par le programme (preuve d’identité telle que la carte nationale d’identité, l’acte de naissance, le permis de conduire, le passeport, la carte d’électeur etc. )
Entreprise
Formelle ou informelle, en phase de démarrage, entre 1 à 3 ans d’activité dans l’une des zones urbaines ciblée par le Projet.
Secteurs
Les secteurs prioritaires de la SND-30 à l’exception des entreprises impliquées dans la production et la distribution d'armes, de boissons alcoolisées, de tabac et/ou de jeux de hasard ainsi que celles dont les activités sont en contradiction avec la loi camerounaise.
Candidature
Le candidat doit disposer d’une carte nationale d’identité et soumettre une proposition de projet d’affaire selon le formulaire d’inscription.Étapes de sélection des bénéficiaires et de mise en oeuvre des activités
01
Campagne de sensibilisation
Campagne de sensibilisation de masse pour amener les candidats à s’inscrire en ligne sur une plate-forme ouverte à cet effet.

Lire moins
02
Inscription des jeunes
Inscription des jeunes et soumission des notes conceptuelles en ligne sur la base d’un formulaire d’inscription élaboré à cet effet.

Lire moins
03
Sélection des notes
Sélection des notes conceptuelles par le prestataire d’accompagnement sur la base d’une grille d’évaluation qu’il conçoit et qui est validée par le Projet.

Lire moins
04
Publication par Projet
Publication par le Projet des listes des candidats sélectionnés par le prestataire d’accompagnement.

Lire moins
05
Accompagnement
Accompagnement des candidats sélectionnés dans l’élaboration pour les amener à améliorer la qualité de leurs notes conceptuelles des plans d’affaires afin qu’ils puissent proposer des plans d’affaires plus élaborés pour la suite du concours.

Lire moins
06
Soumission en ligne des plans d’affaires
Soumission en ligne des plans d’affaires, les candidats vont resoumettre leur plan d’affaires amélioré en ligne sur la plate-forme qui sera réouverte pour une semaine à cet effet.

Lire la suite
07
Sélection des bénéficiaires finaux
Sélection des bénéficiaires finaux qui repose sur une évaluation en 2 étapes : (i) l’évaluation du plan d’affaires final et (ii) l’évaluation du pitch devant le jury .

Lire la suite
08
Octroi des financements
Octroi des financements : les bénéficiaires finaux sélectionnés recevront par virement bancaire dans leur compte ouvert dans les livres des institutions financières, une subvention dont le montant est évalué sur la base du besoin de financement tel que défini dans leur plan d’affaires.

Lire la suite
09
Formaliser les entreprises sélectionnées
Formaliser les entreprises sélectionnées opérant dans le secteur informel : il s’agit d’accompagner les entreprises évoluant dans le secteur informel vers le secteur formel.

Lire la suite
10
Accompagnement technique
Accompagnement technique des bénéficiaires dans le développement de leurs activités qui consistera principalement en un suivi personnalisé des entreprises bénéficiaires par des coachs.

Lire la suite
Couverture géographique'''
custom_text=custom_text_for_cpa+custom_text_iej
# Generate and save the vector store
load_and_process_website_content_from_sitemap(vector_store_path, custom_text,non_local_urls)
