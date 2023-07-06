import os
import annoy
import tinydb
import dotenv
import json
import openai
from app import embeddings


def generate_st_embeddings_index(db: tinydb.TinyDB):
    print("Generating SentenceTransformer embeddings index. This may take a while.")
    st_embeddings_index = annoy.AnnoyIndex(
        f=int(os.getenv("st_embeddings_dim")), metric=os.getenv("distance_metric")
    )
    st_embeddings_transformer = embeddings.SentenceTransformerEmbeddings(
        os.getenv("st_embeddings_model_name")
    )
    documents = db.all()
    for document in documents:
        vector = st_embeddings_transformer.generate(document["content"])
        st_embeddings_index.add_item(document.doc_id, vector)
    st_embeddings_index.build(10)
    st_embeddings_index.save(os.getenv("st_embeddings_index_name"))


def generate_openai_embeddings_index(db: tinydb.TinyDB):
    print("Generating OpenAI embeddings index. This may take a while.")
    openai.api_key = os.getenv("openai_api_key")
    openai_embeddings_index = annoy.AnnoyIndex(
        f=int(os.getenv("openai_embeddings_dim")), metric=os.getenv("distance_metric")
    )
    openai_embeddings_transformer = embeddings.OpenAIEmbeddings(
        os.getenv("openai_embeddings_model_name")
    )
    documents = db.all()
    vectors = openai_embeddings_transformer.generate(
        [document["content"] for document in documents]
    )
    for doc_id, vector in zip(documents, vectors):
        openai_embeddings_index.add_item(doc_id, vector)
        print(f"Added {doc_id} out of {len(documents)} documents to the openai index")
    openai_embeddings_index.build(10)
    openai_embeddings_index.save(os.getenv("openai_embeddings_index_name"))


def generate_indices(db: tinydb.TinyDB):
    # TODO Copy the indices to a backup location before creating new ones
    generate_st_embeddings_index(db)
    # generate_openai_embeddings_index(db)


def generate_db(data: list[dict]) -> tinydb.TinyDB:
    # TODO Copy the db to a backup location before dropping
    db = tinydb.TinyDB(os.getenv("docs_db_name"))
    db.drop_tables()
    db.insert_multiple(data)
    return db


def get_faqs() -> list[dict]:
    with open("parsed_faqs.json", "r") as f:
        parsed_faqs = json.load(f)
    faq_entries = parsed_faqs["entries"]
    return faq_entries


def main():
    # TODO Some other steps, like scraping data, parsing, and rebuilding the db
    data = []
    faqs = get_faqs()
    data.extend(faqs)
    db = generate_db(data)
    generate_indices(db)


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
