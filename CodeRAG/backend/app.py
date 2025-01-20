from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chatbot import CodebaseRAGBot
import os
import tempfile
import shutil

app = Flask(__name__)
CORS(app)  # Simple CORS setup

# Create a temporary directory to store uploaded files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'codebase_upload')
chatbot = None

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Create base upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Clear existing contents if directory exists
        for item in os.listdir(UPLOAD_DIR):
            item_path = os.path.join(UPLOAD_DIR, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        uploaded_files = []
        # Save uploaded files
        for file in files:
            if file.filename:
                try:
                    # Create safe path maintaining directory structure
                    safe_path = os.path.normpath(file.filename)
                    if safe_path.startswith(os.pardir):
                        continue  # Skip files that try to write outside upload dir
                    
                    full_path = os.path.join(UPLOAD_DIR, safe_path)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    
                    # Save file
                    file.save(full_path)
                    uploaded_files.append(safe_path)
                    
                except Exception as file_error:
                    print(f"Error saving file {file.filename}: {str(file_error)}")
                    return jsonify({'error': f'Error saving file {file.filename}: {str(file_error)}'}), 500
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files were uploaded'}), 400
            
        # Initialize new chatbot instance with uploaded files
        global chatbot
        chatbot = CodebaseRAGBot(UPLOAD_DIR)
        
        return jsonify({
            'message': 'Files uploaded successfully',
            'files': uploaded_files
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not chatbot:
            return jsonify({'error': 'Please upload files first'}), 400
        
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        response = chatbot.chat(query)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.teardown_appcontext
def cleanup(error):
    try:
        shutil.rmtree(UPLOAD_DIR)
    except:
        pass

if __name__ == '__main__':
    app.run(debug=True, port=5001) 