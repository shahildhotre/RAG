const API_URL = 'http://localhost:5001';  // Change port to 5001

const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const uploadStatus = document.getElementById('uploadStatus');
const fileList = document.getElementById('fileList');

function addMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = message;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function uploadFiles() {
    const files = fileInput.files;
    if (files.length === 0) {
        uploadStatus.textContent = 'Please select files to upload';
        return;
    }

    uploadStatus.textContent = 'Uploading...';
    const formData = new FormData();
    
    // Append each file to the FormData object
    for (let file of files) {
        formData.append('files', file);
    }

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        uploadStatus.textContent = 'Files uploaded successfully!';
        userInput.disabled = false;
        sendButton.disabled = false;

        // Display uploaded files
        fileList.innerHTML = '<h3>Uploaded Files:</h3>' + 
            data.files.map(file => `<div>${file}</div>`).join('');

    } catch (error) {
        console.error('Upload error:', error);
        uploadStatus.textContent = `Error uploading files: ${error.message}`;
    }
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addMessage(message, true);
    userInput.value = '';

    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message loading';
    loadingDiv.textContent = 'Thinking...';
    chatContainer.appendChild(loadingDiv);

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message }),
        });

        const data = await response.json();
        
        // Remove loading indicator
        chatContainer.removeChild(loadingDiv);

        if (data.error) {
            addMessage('Error: ' + data.error);
        } else {
            addMessage(data.response);
        }
    } catch (error) {
        // Remove loading indicator
        chatContainer.removeChild(loadingDiv);
        addMessage('Error: Could not connect to the server');
    }
}

// Add event listeners
uploadButton.addEventListener('click', uploadFiles);
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Add file input change listener for immediate feedback
fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    uploadStatus.textContent = `Selected ${files.length} files. Click Upload to process.`;
    
    // Debug information about selected files
    console.log('Selected files:', Array.from(files).map(f => ({
        name: f.name,
        path: f.webkitRelativePath || f.name,
        type: f.type,
        size: f.size
    })));
});

document.addEventListener('DOMContentLoaded', () => {
    const uploadButton = document.getElementById('uploadButton');
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const fileList = document.getElementById('fileList');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');

    uploadButton.addEventListener('click', async () => {
        const files = fileInput.files;
        if (files.length === 0) {
            uploadStatus.textContent = 'Please select files to upload';
            return;
        }

        uploadStatus.textContent = 'Uploading...';
        const formData = new FormData();
        
        // Append each file to the FormData object
        for (let file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            uploadStatus.textContent = 'Files uploaded successfully!';
            userInput.disabled = false;
            sendButton.disabled = false;

            // Display uploaded files
            fileList.innerHTML = '<h3>Uploaded Files:</h3>' + 
                data.files.map(file => `<div>${file}</div>`).join('');

        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.textContent = `Error uploading files: ${error.message}`;
        }
    });

    // Rest of your chat functionality...
}); 