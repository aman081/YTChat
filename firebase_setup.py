import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase app with your service account
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Test connection
doc_ref = db.collection("test").document("check")
doc_ref.set({"connected": True})
print("✅ Firebase connected successfully!")
