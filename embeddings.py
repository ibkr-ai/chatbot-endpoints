import abc
import openai
import sentence_transformers


class EmbeddingsModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model_name: str):
        pass

    @abc.abstractmethod
    def generate(self, sentences: str | list[str]) -> list[list[float]]:
        pass


class SentenceTransformerEmbeddings(EmbeddingsModel):
    def __init__(self, model_name: str):
        self.model = sentence_transformers.SentenceTransformer(model_name)

    def generate(self, sentences: str | list[str]) -> list[list[float]]:
        return self.model.encode(sentences, normalize_embeddings=True)


class OpenAIEmbeddings(EmbeddingsModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, sentences: str | list[str]) -> list[list[float]]:
        embeddings_list = []
        for sentence in sentences:
            try:
                # TODO Handle errors raised by OpenAI API
                embeddings_response = openai.Embedding.create(
                    model=self.model_name,
                    input=sentence,
                )
            except:
                continue
            embeddings = embeddings_response["data"][0]["embedding"]
            embeddings_list.append(embeddings)
        return embeddings_list
