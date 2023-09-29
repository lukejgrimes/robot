import numpy as np
import psycopg2
import json

class FaceDatabase:
    def __init__(self):
        # Connect to database
        self.conn = psycopg2.connect(
            host="localhost",
            database="robot",
            user="postgres"
        )

        self.cursor = self.conn.cursor()

    def add_new_user(self, name):
        # Insert a new user into the database, return the user_id created
        query = f"INSERT INTO users (name) VALUES(%s) RETURNING id"
        self.cursor.execute(query, (name,))
        user_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return user_id
    
    def add_new_face(self, user_id, face):
        # Insert a new face embedding into the database
        query = f"INSERT INTO faces (user_id, embedding) VALUES(%s, %s)"
        self.cursor.execute(query, (user_id, face))
        self.conn.commit()

    def add_new_conversation(self, user_id, conversation):
        # Insert a new conversation into the database
        query = f"INSERT INTO conversations (user_id, conversation) VALUES(%s, %s)"
        self.cursor.execute(query, (user_id, json.dumps(conversation)))
        self.conn.commit()

    def update_user_summary(self, user_id, summary):
        # Update user summary in the database
        query = f"UPDATE users SET summary = %s WHERE id = %s"
        self.cursor.execute(query, (summary, user_id))
        self.conn.commit()

    def find_embedding_in_db(self, embedding):
        # Get the matching embedding in the db and return the corresponding name and user id
        query = f"SELECT user_id FROM faces WHERE embedding = %s::double precision[]"
        self.cursor.execute(query, (embedding,))
        user_id = self.cursor.fetchone()
        query = f"SELECT name FROM users WHERE id = %s"
        self.cursor.execute(query, (user_id,))
        name = self.cursor.fetchone()
        self.conn.commit()
        return name, user_id
    
    def get_all_embeddings(self):
        # Get every embedding from the db
        query = "SELECT embedding FROM faces"
        self.cursor.execute(query)
        all_embeddings = self.cursor.fetchall()
        all_embeddings = [embedding[0] for embedding in all_embeddings]
        return all_embeddings

    def close_connection(self):
        # Close DB connection
        self.cursor.close()
        self.conn.close()
