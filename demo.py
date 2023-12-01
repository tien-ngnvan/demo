import os
import pickle

from flask import Flask, request, jsonify
from faiss_database import FaissDB
from inference import OVModel


app = Flask(__name__)


CKPT_PATH = "ovmodel"
embedding_fn = OVModel(CKPT_PATH)
faissdb =  None


def init_model():
    global faissdb
    with open(os.path.join("data", "doanh_nghiep.pkl"), 'rb') as f:
        reps = pickle.load(f)

    faissdb = FaissDB.from_embeddings(reps, embedding=embedding_fn)
    
@app.route("/law.retrieval", methods=["POST"])
def query_result():
    req = request.json
    result = faissdb.similarity_search_with_score(req["query"], k=int(req["knn"]))
    return jsonify(result)


if __name__ == "__main__":
    print("Waiting load database . . .")
    init_model()
    app.run(host="0.0.0.0", port=5000)
