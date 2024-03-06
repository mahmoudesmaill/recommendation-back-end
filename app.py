from flask import Flask, jsonify,make_response,request
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading
import queue

app = Flask(__name__)

# Initialize Redis client

# Create a thread-safe queue for incoming requests
request_queue = queue.Queue()

# Load Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Routes
@app.route('/recommend', methods=['GET'])
def recommendations():
    # Get the JSON data from the request body
    json_data = request.json
    # Check if JSON data is present
    if json_data and 'segment_description' in json_data:
        try:
            segment_description = json_data['segment_description']
            embeddings = np.load('embeddings.npy')
            segment_ids = np.load('segment_ids.npy')            
            recommendations = calculate_recommendations(model=model,input_segment=segment_description,skydeo_segment_ids=segment_ids, embeddings=embeddings)
            print("recommendations:", recommendations)
            return make_response({"recommendations": recommendations},200)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Invalid or missing JSON data"}), 400
    
@app.route('/update_embeddings', methods=['POST'])
def update_embeddings():
    try:
        json_data = request.json
        if json_data and 'segment_description' in json_data and 'segment_id' in json_data:
            segment_description = request.json['segment_description']
            segment_id = request.json['segment_id']
            update_thread = threading.Thread(target=update_embeddings_segment_ids_in_thread, args=(segment_description,segment_id))
            update_thread.start()
            return jsonify({"message": "Update process started successfully"})
        else:
            return jsonify({"error": "Invalid or missing JSON data"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/handle_request', methods=['POST'])
def handle_request():
    try:
        request_queue.put(request.json)
        return jsonify({"message": "Request added to the queue"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Calculate recommendations
def calculate_recommendations(model, input_segment, skydeo_segment_ids, embeddings):
    provider_segment = model.encode(input_segment)
    cs = cosine_similarity([provider_segment], embeddings)
    top_indices = np.argsort(cs[0])[-3:][::-1]
    recommendations = [{"id": skydeo_segment_ids[score], "score": float(cs[0][score])}
                       for score in top_indices]
    return recommendations

# Define a function to update embeddings and cache in a thread
def update_embeddings_segment_ids_in_thread(segment_description, segment_id):
    try:
        # Update embeddings
        new_embedding = model.encode(segment_description)

        # Load existing embeddings from file
        existing_embeddings = np.load('embeddings.npy')

        # Append new embedding
        updated_embeddings = np.concatenate((existing_embeddings, [new_embedding]), axis=0)

        # Save the updated embeddings back to the file
        np.save('embeddings.npy', updated_embeddings)

        # Load existing segment IDs from file
        existing_segment_ids = np.load('segment_ids.npy')

        # Append the new segment ID to the list
        updated_segment_ids = np.concatenate((existing_segment_ids, [segment_id]), axis=0)

        # Save the updated segment IDs back to the file
        np.save('segment_ids.npy', updated_segment_ids)

        print("Data updated successfully!")

        # Process queued requests
        while not request_queue.empty():
            request = request_queue.get()
            handle_request(request)
    except Exception as e:
        print(f"Error updating embeddings in thread: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
