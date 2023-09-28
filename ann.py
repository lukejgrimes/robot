from annoy import AnnoyIndex
import os
from facedatabase import FaceDatabase
from dotenv import load_dotenv

class ANN:
    def __init__(self):
        load_dotenv()
        self.annoy = self.create_index() # Either create a new index or load saved index
        self.db = FaceDatabase() # Initalize DB object

    def add_item_to_index(self, embedding):
        len_of_data = self.annoy.get_n_items() # Length of the annoy index
        self.annoy.add_item(len_of_data, embedding) # Add embedding to the end of the annoy index
        self.annoy.build(100) # Rebuild the index with new embedding
        self.annoy.save("index.ann") # Save the new index to the file

    def create_index(self):
        # Check if an index has already been created
        if os.path.exists(os.getenv('ANNOY_INDEX')):
            annoy = AnnoyIndex(512, 'euclidean') # Initalize ANNOY with vector size of 512, use euclidean distance
            annoy.load(os.getenv('ANNOY_INDEX')) # Load the previous index
        else:
            annoy = AnnoyIndex(512, 'euclidean') # Initalize ANNOY with vector size of 512, use euclidean distance
        
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
    
        if distance <= .2:
             # If distance is within threshold find the user's information in the database
            name, user_id = self.db.find_embedding_in_db(nearest_face)
            return name, user_id
        else:
            return None, None
    