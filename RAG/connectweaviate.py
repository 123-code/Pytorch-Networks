import weaviate 
import weaviate.classes as wvc
import requests 
import json

client = weaviate.Client(
     url="https://embeddings-test1-z95cu7l6.weaviate.network",
     auth_client_secret = weaviate.AuthApiKey(api_key),
    additional_headers = {
        "X-OpenAI-Api-Key":
    }
)

class_obj = {
    "class":"Question1",
    "vectorizer":"text2vec-openai",
    "moduleConfig":{
        "text2vec-openai":{

        },
        "generative-openai":{}
    }

}
client.schema.create_class(class_obj)

response = requests.get("'https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json'")
data = json.loads(response.text)

client.batch.configure(batch_size = 100)

with client.batch as batch:
    for i, d in enumerate(data):
        print(f"Importing question:{i+1}")
        properties = {
            "answer":d["Answer"],
            "question":d["Question"],
            "category":d["Category"]

        }
        batch.add_data_object(
            data_object = properties,
            class_name = "Question"

        )



response = (
    client.query
    .get("Question",["question","answer","category"])
    .with_near_text({"concepts":["music"]})
    .with_limit(10)
    .do()
)

print(json.dumps(response,indent = 4))
