import pydantic


class Settings(pydantic.BaseSettings):
    openai_api_key: str
    openai_completions_model: str
    openai_embeddings_index_name: str
    openai_embeddings_model_name: str
    openai_embeddings_dim: int
    st_embeddings_index_name: str
    st_embeddings_model_name: str
    st_embeddings_dim: int
    distance_metric: str
    docs_db_name: str

    class Config:
        env_file = ".env"
