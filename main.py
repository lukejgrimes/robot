from chat import Chat
from facialrecognition import FacialRecognition
from facedatabase import FaceDatabase
from ann import ANN

# Initialize the facial recognition, chat, database, and ANN search objects
fr = FacialRecognition()
chat = Chat()
db = FaceDatabase()
a = ANN()

# Create an embedding from the user's face
face = fr.recognize_face()
# Search annoy index for the user's face
name, user_id = a.search_face(face)

# If no matching face is found, ask user for their name and add their name and face to the database
if not name:
    name = chat.get_name()
    user_id = db.add_new_user(name)
    db.add_new_face(user_id, face)
    a.add_item_to_index(face)

# Begin conversation with user
conversation = chat.conversation(name)
# Once conversation is complete, produce a user summary and add the summary and conversationto the db.
summary = chat.summary(conversation)
db.add_new_conversation(user_id, conversation)
db.update_user_summary(user_id, summary)
db.close_connection() # Close database connection
