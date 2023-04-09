import os
import sqlite3

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

# Get the full path to the Bear database file
db_path = os.path.expanduser("~/Library/Group Containers/9K33E3U3T4.net.shinyfrog.bear/Application Data/database.sqlite")

# Connect to the database
conn = sqlite3.connect(db_path)

# Get a cursor object
cur = conn.cursor()

# Read the notes that haven't been deleted
cur.execute("SELECT ZTITLE, ZTEXT FROM ZSFNOTE WHERE ZTRASHED = 0")

# Create the 'bear notes' directory if it doesn't exist
if not os.path.exists('bear notes'):
    os.makedirs('bear notes')

# Empty the directory of any existing note files
for filename in os.listdir("bear notes"):
    file_path = os.path.join("bear notes", filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error deleting file {filename}: {e}")

# Loop through the results and save each note as a text file
for title, content in cur.fetchall():
    # Remove any characters from the title that can't be used in a file name
    title = ''.join(c for c in title if c.isalnum() or c in (' ', '_', '-'))
    # Construct the file name by adding the '.txt' extension to the title
    filename = f"bear notes/{title}.txt"
    # Write the note content to the file
    with open(filename, 'w') as f:
        f.write(content)

# Close the cursor and the database connection
cur.close()
conn.close()

# Set up a directory loader to load documents from the specified folder
loader = DirectoryLoader('bear notes/', glob='*.txt')
documents = loader.load()

# Set up a text splitter to split documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Set up an OpenAIEmbeddings object to generate text embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

# Set up a Chroma object to index the embeddings and perform similarity searches
docsearch = Chroma.from_documents(texts, embeddings)

# Set up a RetrievalQA object to perform question-answering on the indexed documents
# possible chain types: stuff, map_reduce, refine, map-rerank
qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="refine", retriever=docsearch.as_retriever())

# Begin an infinite loop to allow the user to enter queries and receive results
while True: 
    query = input("\nType prompt: ")
    results = qa.run(query)
    print(results)
