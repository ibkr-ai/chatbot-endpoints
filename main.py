import annoy
import tinydb
import timeit
import openai
import fastapi
from app import embeddings, config


app = fastapi.FastAPI()
settings = config.Settings()
openai.api_key = settings.openai_api_key
# openai_embeddings_index = annoy.AnnoyIndex(
#     f=settings.openai_embeddings_dim, metric=settings.distance_metric
# )
st_embeddings_index = annoy.AnnoyIndex(
    f=settings.st_embeddings_dim, metric=settings.distance_metric
)
# openai_embeddings_index.load(settings.openai_embeddings_index_name)
st_embeddings_index.load(settings.st_embeddings_index_name)
docs_db = tinydb.TinyDB(settings.docs_db_name)


def get_matching_documents_from_db(ids: list[int]) -> list[dict]:
    documents = docs_db.get(doc_ids=ids)
    return documents


def search_index(
    query: str, top_k: int = 10, threshold: float | None = None
) -> list[dict]:
    # Based on a user query documents that most closely resemble it semantically. Returns a list of documents.
    include_distances = True
    try:
        st_embeddings_model_name = settings.st_embeddings_model_name
        query_embeddings = embeddings.SentenceTransformerEmbeddings(
            st_embeddings_model_name
        ).generate(query)
        ids, scores = st_embeddings_index.get_nns_by_vector(
            query_embeddings, top_k, include_distances=include_distances
        )
    except:
        openai_embeddings_model_name = settings.openai_embeddings_model_name
        query_embeddings = embeddings.OpenAIEmbeddings(
            openai_embeddings_model_name
        ).generate(query)
        ids, scores = openai_embeddings_index.get_nns_by_vector(
            query_embeddings, top_k, include_distances=include_distances
        )
    finally:
        if threshold is not None:
            # TODO Filter out results whose scores are below a predefined threshold
            filtered_ids = filter(lambda x: x[1] > threshold, zip(ids, scores))
            # TOOD Not very optimal, but it works for now
            if len(list(filtered_ids)) == 0:
                return []
            ids, scores = zip(*filtered_ids)
        documents = get_matching_documents_from_db(ids)
        print(scores)
        return documents


@app.get("/search", response_model=list[dict])
async def search(query: str, top_k: int = 10):
    # Return a list of documents that are semantically similar to the query promts
    search_start_time = timeit.default_timer()
    results = search_index(query, top_k)
    search_elapsed_time = timeit.default_timer() - search_start_time
    print(f"Search elapsed time: {search_elapsed_time}")
    return results


@app.get("/completions", response_model=dict)
async def get_completion(query: str):
    threshold = None
    context_docs = search_index(query, threshold=threshold, top_k=1)
    if len(context_docs) == 0:
        return {"message": "No matching documents found"}
    # TODO Parse docs from DB to form a single context string that we pass to OpenAI
    context_string = "; ".join([doc["content"] for doc in context_docs])
    completion = await openai.ChatCompletion.acreate(
        model=settings.openai_completions_model,
        messages=[
            {
                "role": "system",
                "content": f"Answer the user's questions, based on the following context: {context_string}",
            },
            {"role": "user", "content": query},
        ],
    )
    return completion
