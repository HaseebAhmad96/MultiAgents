import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

INPUT_DOCS_DIR = os.getenv("INPUT_DOCS_DIR", "./input_docs")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "multi_agent_docs"

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".csv": "csv",
}

def buildEmbeddingModel():
   
    EmbeddingModel = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return EmbeddingModel

def loadSingleDocument(FilePath: str) -> list:
  
    Extension = os.path.splitext(FilePath)[1].lower()

    if Extension == ".pdf":
        Loader = PyPDFLoader(FilePath)
    elif Extension == ".txt":
        Loader = TextLoader(FilePath, encoding="utf-8")
    elif Extension == ".csv":
        Loader = CSVLoader(FilePath, encoding="utf-8")
    else:
        print(f"[SKIP] Unsupported file type: {FilePath}")
        return []

    Documents = Loader.load()

    for Doc in Documents:
        Doc.metadata["source_file"] = os.path.basename(FilePath)

    return Documents

def loadAllDocuments() -> list:
    
    if not os.path.exists(INPUT_DOCS_DIR):
        print(f"[WARNING] Input folder not found: {INPUT_DOCS_DIR}")
        return []

    AllDocuments = []
    FilesFound = 0
    FilesLoaded = 0

    for FileName in os.listdir(INPUT_DOCS_DIR):
        Extension = os.path.splitext(FileName)[1].lower()
        if Extension not in SUPPORTED_EXTENSIONS:
            continue

        FilesFound += 1
        FilePath    = os.path.join(INPUT_DOCS_DIR, FileName)

        print(f"[LOADING] {FileName}")
        Docs = loadSingleDocument(FilePath)

        if Docs:
            AllDocuments.extend(Docs)
            FilesLoaded += 1
            print(f"-> {len(Docs)} document(s) loaded")

    print(f"\nFiles found: {FilesFound} | Files loaded: {FilesLoaded}")
    print(f"Total raw documents: {len(AllDocuments)}")
    return AllDocuments

def chunkDocuments(Documents: list) -> list:
 
    Splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    Chunks = Splitter.split_documents(Documents)

    for Index, Chunk in enumerate(Chunks):
        Chunk.metadata["chunk_index"] = Index

    print(f"  Total chunks after splitting: {len(Chunks)}")
    return Chunks

def buildVectorStore(Chunks: list, EmbeddingModel) -> Chroma:

    print(f"  Embedding {len(Chunks)} chunks into ChromaDB...")
    print(f"  (First run downloads the embedding model — ~90MB, one time only)")

    VectorStore = Chroma.from_documents(
        documents=Chunks,
        embedding=EmbeddingModel,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    print(f"  ChromaDB collection '{COLLECTION_NAME}' saved to: {CHROMA_PERSIST_DIR}")
    return VectorStore

def getVectorStore(EmbeddingModel) -> Chroma:
    
    VectorStore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EmbeddingModel,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return VectorStore

def getDocumentStats() -> dict:
  
    EmbeddingModel = buildEmbeddingModel()
    VectorStore = getVectorStore(EmbeddingModel)
    Collection = VectorStore._collection

    TotalChunks = Collection.count()
    AllMetadata = Collection.get(include=["metadatas"])["metadatas"]

    SourceFiles = set()
    for Meta in AllMetadata:
        SourceFile = Meta.get("source_file", "unknown")
        SourceFiles.add(SourceFile)

    return {
        "total_chunks": TotalChunks,
        "total_files": len(SourceFiles),
        "source_files": sorted(list(SourceFiles)),
        "collection": COLLECTION_NAME,
        "persist_dir": CHROMA_PERSIST_DIR,
    }

def runDocumentPipeline(ForceRebuild: bool = False) -> Chroma:

    EmbeddingModel = buildEmbeddingModel()

    ExistingStore = getVectorStore(EmbeddingModel)
    ExistingCount = ExistingStore._collection.count()

    if ExistingCount > 0 and not ForceRebuild:
        print(f"ChromaDB already contains {ExistingCount} chunks.")
        print(f"Skipping ingestion. Pass ForceRebuild=True to re-embed.")
        print("[PIPELINE] Done.\n")
        return ExistingStore

    print(f"No existing data found (or rebuild requested).")
    RawDocuments = loadAllDocuments()

    if not RawDocuments:
        print("[WARNING] No documents loaded. Drop files into input_docs/ and re-run.")
        print("[PIPELINE] Done (empty).\n")
        return ExistingStore

    Chunks = chunkDocuments(RawDocuments)
    VectorStore = buildVectorStore(Chunks, EmbeddingModel)

    print("[PIPELINE] Done.\n")
    return VectorStore