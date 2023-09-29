from annoy import AnnoyIndex
import os
from facedatabase import FaceDatabase
from dotenv import load_dotenv

class ANN:
    def __init__(self):
        load_dotenv()
        self.annoy = self.create_index() # Either create a new index or load saved index
        self.db = FaceDatabase() # Initalize DB object

    def update_index(self, new_embeddings):
        # Add new embeddings to the index and rebuild
        self.annoy.unload() # Unload index
        annoy = AnnoyIndex(512, 'euclidean') # Create new annoy index
        # Get all saved embeddings from the db and add new embeddings
        all_embeddings = self.db.get_all_embeddings() + new_embeddings 
        for i in range(len(all_embeddings)):
            annoy.add_item(i, all_embeddings[i])
        # Rebuild and save 
        annoy.build(100)
        annoy.save(os.getenv('ANNOY_INDEX'))

        self.annoy = annoy

    def create_index(self):
        annoy = AnnoyIndex(512, 'euclidean') # Initalize ANNOY with vector size of 512, use euclidean distance
        # Check if an index has already been created
        if os.path.exists(os.getenv('ANNOY_INDEX')):
            annoy.load(os.getenv('ANNOY_INDEX')) # Load the previous index
        return annoy

    def search_face(self, embedding):
        # Preform an Approximate Nearest Neighbors search on our index
        nearest_vectors, distances = self.annoy.get_nns_by_vector(embedding, 1, include_distances=True)
        if not nearest_vectors:
            # If no vectors were found
            return None, None
        
        vector_id = nearest_vectors[0]
        distance = distances[0]
        nearest_face = self.annoy.get_item_vector(vector_id) # Find the nearest face's embedding
        print(f"Nearest Neighbor Distance: {distance}")
        if distance <= .7:
             # If distance is within threshold find the user's information in the database
            name, user_id = self.db.find_embedding_in_db(nearest_face)
            return name, user_id
        else:
            return None, None
    