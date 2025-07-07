import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import time
from datetime import datetime
import io

# Configure page
st.set_page_config(
    page_title="Intelligent Document Chunking System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .chunk-preview {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .doc-type-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .strategy-badge {
        background-color: #007bff;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_document_types():
    """Get available document types from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/document-types")
        if response.status_code == 200:
            return response.json()["document_types"]
        return []
    except:
        return []

def get_chunking_strategies():
    """Get available chunking strategies from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/chunking-strategies")
        if response.status_code == 200:
            return response.json()["strategies"]
        return []
    except:
        return []

def chunk_document(content: str, filename: str = "", doc_id: str = None):
    """Send document to API for chunking"""
    try:
        payload = {
            "content": content,
            "filename": filename,
            "document_id": doc_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chunk",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def chunk_file(file_content: bytes, filename: str):
    """Send file to API for chunking"""
    try:
        files = {"file": (filename, file_content, "text/plain")}
        response = requests.post(f"{API_BASE_URL}/chunk-file", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def display_metrics(result: Dict[str, Any]):
    """Display chunking metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{result['total_chunks']}</h3>
            <p>Total Chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_tokens = sum(chunk['token_count'] for chunk in result['metadata']) / len(result['metadata'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_tokens:.0f}</h3>
            <p>Avg Token Count</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3><span class="doc-type-badge">{result['document_type']}</span></h3>
            <p>Document Type</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3><span class="strategy-badge">{result['chunking_strategy']}</span></h3>
            <p>Strategy Used</p>
        </div>
        """, unsafe_allow_html=True)

def create_token_distribution_chart(metadata: List[Dict[str, Any]]):
    """Create token distribution chart"""
    token_counts = [chunk['token_count'] for chunk in metadata]
    
    fig = px.histogram(
        x=token_counts,
        nbins=20,
        title="Token Distribution Across Chunks",
        labels={'x': 'Token Count', 'y': 'Number of Chunks'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Token Count",
        yaxis_title="Number of Chunks"
    )
    
    return fig

def create_chunk_size_chart(metadata: List[Dict[str, Any]]):
    """Create chunk size progression chart"""
    chunk_indices = [chunk['chunk_index'] for chunk in metadata]
    token_counts = [chunk['token_count'] for chunk in metadata]
    
    fig = px.line(
        x=chunk_indices,
        y=token_counts,
        title="Chunk Size Progression",
        labels={'x': 'Chunk Index', 'y': 'Token Count'},
        markers=True
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Chunk Index",
        yaxis_title="Token Count",
        showlegend=False
    )
    
    return fig

def display_chunks(chunks: List[str], metadata: List[Dict[str, Any]]):
    """Display chunks with metadata"""
    st.subheader("📄 Document Chunks")
    
    # Chunk navigation
    chunk_options = [f"Chunk {i+1}" for i in range(len(chunks))]
    selected_chunk = st.selectbox("Select chunk to view:", chunk_options)
    
    if selected_chunk:
        chunk_index = int(selected_chunk.split()[1]) - 1
        chunk_data = chunks[chunk_index]
        chunk_meta = metadata[chunk_index]
        
        # Chunk metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chunk ID", chunk_meta['chunk_id'])
        with col2:
            st.metric("Token Count", chunk_meta['token_count'])
        with col3:
            if chunk_meta.get('section_title'):
                st.metric("Section", chunk_meta['section_title'])
        
        # Chunk content
        st.markdown("**Chunk Content:**")
        st.markdown('<div class="chunk-preview">' + chunk_data.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Intelligent Document Chunking System</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🚨 API Server is not running. Please start the FastAPI server on port 8000.")
        st.code("uvicorn main:app --reload --port 8000", language="bash")
        return
    
    st.success("✅ API Server is running")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Input Method:",
        ["Text Input", "File Upload", "Sample Documents"]
    )
    
    # Document processing
    result = None
    
    if input_method == "Text Input":
        st.subheader("📝 Text Input")
        
        # Text input
        content = st.text_area(
            "Enter your document content:",
            height=300,
            placeholder="Paste your document content here..."
        )
        
        filename = st.text_input("Filename (optional):", placeholder="document.txt")
        doc_id = st.text_input("Document ID (optional):", placeholder="Auto-generated if empty")
        
        if st.button("🚀 Process Document", type="primary"):
            if content.strip():
                with st.spinner("Processing document..."):
                    result = chunk_document(content, filename, doc_id)
            else:
                st.warning("Please enter some content to process.")
    
    elif input_method == "File Upload":
        st.subheader("📁 File Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=['txt', 'md', 'py', 'js', 'json', 'sql'],
            help="Supported formats: TXT, MD, PY, JS, JSON, SQL"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process File", type="primary"):
                with st.spinner("Processing file..."):
                    file_content = uploaded_file.getvalue()
                    result = chunk_file(file_content, uploaded_file.name)
    
    else:  # Sample Documents
        st.subheader("📚 Sample Documents")
        
        sample_docs = {
            "API Documentation": {
                "content": """# User Management API

## Authentication
All API requests require authentication using Bearer tokens.

### Get Token
```
POST /auth/token
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## User Endpoints

### Create User
```
POST /users
Authorization: Bearer {token}
Content-Type: application/json

{
  "username": "newuser",
  "email": "newuser@example.com",
  "password": "securepassword"
}
```

### Get User Profile
```
GET /users/{user_id}
Authorization: Bearer {token}
```

### Update User
```
PUT /users/{user_id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "email": "updated@example.com",
  "profile": {
    "name": "Updated Name"
  }
}
```

## Error Handling
The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error""",
                "filename": "api_docs.md"
            },
            "Support Ticket": {
                "content": """# Support Ticket #12345

## Issue Summary
Customer reports login authentication failure on mobile application.

## Customer Details
- Customer ID: CUST-789
- Account Type: Premium
- Platform: iOS App v2.1.3
- Device: iPhone 12 Pro
- OS Version: iOS 15.6

## Problem Description
User cannot log in to the mobile application. Error message displays: "Invalid credentials" even with correct username and password. The same credentials work fine on the web application.

## Steps to Reproduce
1. Open mobile app
2. Enter valid username: john.doe@email.com
3. Enter correct password
4. Tap "Login" button
5. Error message appears: "Invalid credentials"

## Troubleshooting Steps Performed
1. Verified credentials in web app - Working
2. Cleared app cache and data - No change
3. Uninstalled and reinstalled app - No change
4. Checked server logs - No authentication attempts recorded

## Root Cause Analysis
Investigation revealed that the mobile app API endpoint was using an outdated authentication method that was deprecated in v2.0.

## Solution Implemented
1. Updated mobile app to use new OAuth 2.0 authentication flow
2. Deployed hotfix to production
3. Verified fix with customer

## Prevention Measures
- Updated API documentation
- Added automated tests for authentication flows
- Implemented API version deprecation warnings

## Follow-up Actions
- Monitor authentication success rates
- Schedule customer satisfaction survey
- Update mobile app to latest version""",
                "filename": "support_ticket_12345.md"
            },
            "Technical Policy": {
                "content": """# Data Security and Privacy Policy

## 1. Purpose and Scope

This policy establishes the framework for protecting sensitive data and ensuring privacy compliance across all systems and applications within the organization.

### 1.1 Applicability
This policy applies to all employees, contractors, and third-party vendors who have access to company data systems.

### 1.2 Compliance Requirements
All data handling must comply with:
- GDPR (General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- SOC 2 Type II requirements
- ISO 27001 standards

## 2. Data Classification

### 2.1 Public Data
Information that can be freely shared without risk to the organization.

### 2.2 Internal Data
Information intended for use within the organization but not for external distribution.

### 2.3 Confidential Data
Sensitive information that requires protection and controlled access.

### 2.4 Restricted Data
Highly sensitive information requiring the highest level of protection.

## 3. Access Controls

### 3.1 Authentication Requirements
- Multi-factor authentication MUST be enabled for all user accounts
- Password complexity requirements SHALL be enforced
- Session timeouts MUST be configured appropriately

### 3.2 Authorization Principles
- Principle of least privilege SHALL be applied
- Role-based access control MUST be implemented
- Regular access reviews SHALL be conducted quarterly

## 4. Data Protection Measures

### 4.1 Encryption Standards
- Data at rest MUST be encrypted using AES-256
- Data in transit MUST use TLS 1.3 or higher
- Encryption keys SHALL be managed through approved key management systems

### 4.2 Backup and Recovery
- Regular backups MUST be performed and tested
- Recovery procedures SHALL be documented and tested annually
- Backup data MUST be encrypted and stored securely

## 5. Incident Response

### 5.1 Reporting Requirements
Security incidents MUST be reported within 24 hours of discovery.

### 5.2 Response Procedures
1. Immediate containment of the incident
2. Assessment of impact and scope
3. Notification to relevant stakeholders
4. Implementation of remediation measures
5. Post-incident review and documentation

## 6. Compliance Monitoring

Regular audits SHALL be conducted to ensure policy compliance. Non-compliance may result in disciplinary action up to and including termination.

## 7. Policy Updates

This policy will be reviewed annually and updated as necessary to reflect changes in regulations, technology, and business requirements.""",
                "filename": "data_security_policy.md"
            },
            "Python Tutorial": {
                "content": """# Getting Started with Python Flask Web Development

## Introduction

Flask is a lightweight web framework for Python that makes it easy to build web applications. This tutorial will guide you through creating your first Flask application.

## Prerequisites

Before starting, ensure you have:
- Python 3.7 or higher installed
- Basic knowledge of Python programming
- A text editor or IDE

## Installation

First, let's install Flask using pip:

```bash
pip install flask
```

## Creating Your First Flask App

### Step 1: Basic Application Structure

Create a new file called `app.py` and add the following code:

```python
from flask import Flask, render_template, request, redirect, url_for

# Create Flask application instance
app = Flask(__name__)

# Configure secret key for sessions
app.secret_key = 'your-secret-key-here'

@app.route('/')
def home():
    return '<h1>Welcome to Flask!</h1>'

@app.route('/about')
def about():
    return '<h1>About Page</h1><p>This is a Flask application.</p>'

if __name__ == '__main__':
    app.run(debug=True)
```

### Step 2: Running Your Application

Save the file and run it:

```bash
python app.py
```

Open your web browser and navigate to `http://localhost:5000` to see your application.

### Step 3: Adding Templates

Create a folder called `templates` in your project directory. Flask uses the Jinja2 template engine.

Create `templates/base.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Flask App{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('home') }}">Flask App</a>
            <div class="navbar-nav">
                <a class="nav-link" href="{{ url_for('home') }}">Home</a>
                <a class="nav-link" href="{{ url_for('about') }}">About</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
```

### Step 4: Handling Forms

Let's add a contact form. First, update your `app.py`:

```python
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Here you would typically save to database
        # For now, we'll just redirect with a success message
        return redirect(url_for('thank_you', name=name))
    
    return render_template('contact.html')

@app.route('/thank-you/<name>')
def thank_you(name):
    return render_template('thank_you.html', name=name)
```

### Step 5: Database Integration

For database operations, we'll use SQLAlchemy:

```bash
pip install flask-sqlalchemy
```

Add to your `app.py`:

```python
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    
    def __repr__(self):
        return f'<User {self.username}>'

# Create tables
with app.app_context():
    db.create_all()
```

## Next Steps

Now you have a basic Flask application with:
- Route handling
- Template rendering
- Form processing
- Database integration

Continue learning by exploring:
- User authentication
- File uploads
- API development
- Deployment strategies""",
                "filename": "flask_tutorial.md"
            },
            "Code Documentation": {
                "content": """# Authentication Module Documentation

## Overview
This module provides authentication and password management functionality for secure user access control.

## Classes

### AuthManager
Main authentication manager class that handles user authentication, token generation, and verification.

```python
class AuthManager:
    def __init__(self, db_connection, secret_key):
        \"\"\"
        Initialize authentication manager.
        
        Args:
            db_connection: Database connection instance
            secret_key: Secret key for token generation
        \"\"\"
        self.db = db_connection
        self.secret_key = secret_key
        self.token_expiry = 3600  # 1 hour
    
    def authenticate_user(self, username, password):
        \"\"\"
        Authenticate user credentials.
        
        Args:
            username (str): User's username or email
            password (str): User's password
            
        Returns:
            dict: Authentication result with user data or error
        \"\"\"
        user = self.db.get_user_by_username(username)
        
        if not user:
            return {'success': False, 'error': 'User not found'}
        
        if not self.verify_password(password, user.password_hash):
            return {'success': False, 'error': 'Invalid password'}
        
        token = self.generate_token(user.id)
        return {
            'success': True,
            'user': user.to_dict(),
            'token': token
        }
    
    def generate_token(self, user_id):
        \"\"\"
        Generate JWT token for user.
        
        Args:
            user_id (int): User's unique identifier
            
        Returns:
            str: JWT token string
        \"\"\"
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token):
        \"\"\"
        Verify JWT token validity.
        
        Args:
            token (str): JWT token to verify
            
        Returns:
            dict: Token verification result
        \"\"\"
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return {'valid': True, 'user_id': payload['user_id']}
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
```

### PasswordManager
Utility class for password hashing and verification.

```python
class PasswordManager:
    @staticmethod
    def hash_password(password):
        \"\"\"
        Hash password using bcrypt.
        
        Args:
            password (str): Plain text password
            
        Returns:
            str: Hashed password
        \"\"\"
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password, hashed_password):
        \"\"\"
        Verify password against hash.
        
        Args:
            password (str): Plain text password
            hashed_password (str): Stored password hash
            
        Returns:
            bool: True if password matches, False otherwise
        \"\"\"
        return bcrypt.checkpw(
            password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
```

## Usage Examples

### Basic Authentication

```python
# Initialize authentication manager
auth_manager = AuthManager(db_connection, 'your-secret-key')

# Authenticate user
result = auth_manager.authenticate_user('john@example.com', 'password123')

if result['success']:
    user_token = result['token']
    print(f"Login successful! Token: {user_token}")
else:
    print(f"Login failed: {result['error']}")
```

### Token Verification

```python
# Verify token in API endpoints
@app.route('/protected')
def protected_route():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'No token provided'}), 401
    
    # Remove 'Bearer ' prefix
    token = token.replace('Bearer ', '')
    
    verification = auth_manager.verify_token(token)
    if not verification['valid']:
        return jsonify({'error': verification['error']}), 401
    
    return jsonify({'message': 'Access granted'})
```

## Security Considerations

1. **Password Storage**: Never store plain text passwords. Always use bcrypt or similar hashing algorithms.

2. **Token Security**: 
   - Use HTTPS in production
   - Set appropriate token expiry times
   - Implement token refresh mechanisms

3. **Rate Limiting**: Implement rate limiting for authentication endpoints to prevent brute force attacks.

4. **Session Management**: 
   - Clear tokens on logout
   - Implement session timeout
   - Use secure cookie flags

## Error Handling

The authentication system provides detailed error messages for debugging while maintaining security:

```python
ERROR_CODES = {
    'USER_NOT_FOUND': 'Invalid credentials',
    'INVALID_PASSWORD': 'Invalid credentials',
    'TOKEN_EXPIRED': 'Session expired, please login again',
    'INVALID_TOKEN': 'Invalid authentication token'
}
```

## Testing

Example test cases for the authentication module:

```python
def test_successful_authentication():
    auth_manager = AuthManager(mock_db, 'test-secret')
    result = auth_manager.authenticate_user('testuser', 'testpass')
    assert result['success'] is True
    assert 'token' in result

def test_invalid_credentials():
    auth_manager = AuthManager(mock_db, 'test-secret')
    result = auth_manager.authenticate_user('testuser', 'wrongpass')
    assert result['success'] is False
    assert result['error'] == 'Invalid password'
```""",
                "filename": "auth_module.py"
            }
        }
        
        selected_sample = st.selectbox(
            "Choose a sample document:",
            list(sample_docs.keys())
        )
        
        if selected_sample:
            sample_data = sample_docs[selected_sample]
            st.markdown(f"**Sample: {selected_sample}**")
            
            # Show preview
            with st.expander("Preview Content"):
                st.markdown(sample_data["content"][:500] + "..." if len(sample_data["content"]) > 500 else sample_data["content"])
            
            if st.button("🚀 Process Sample Document", type="primary"):
                with st.spinner("Processing sample document..."):
                    result = chunk_document(
                        sample_data["content"],
                        sample_data["filename"]
                    )
    
    # Display results
    if result:
        st.success("✅ Document processed successfully!")
        
        # Display metrics
        display_metrics(result)
        
        # Analytics section
        st.subheader("📊 Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Token distribution chart
            token_chart = create_token_distribution_chart(result['metadata'])
            st.plotly_chart(token_chart, use_container_width=True)
        
        with col2:
            # Chunk size progression
            size_chart = create_chunk_size_chart(result['metadata'])
            st.plotly_chart(size_chart, use_container_width=True)
        
        # Chunk details table
        st.subheader("📋 Chunk Details")
        chunk_df = pd.DataFrame(result['metadata'])
        st.dataframe(chunk_df, use_container_width=True)
        
        # Display chunks
        display_chunks(result['chunks'], result['metadata'])
        
        # Download options
        st.subheader("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as JSON
            json_data = json.dumps(result, indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"chunks_{result['document_id']}.json",
                mime="application/json"
            )
        
        with col2:
            # Download as CSV
            csv_data = chunk_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"chunks_{result['document_id']}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Download chunks as text
            chunks_text = "\n\n" + "="*50 + "\n\n".join([f"CHUNK {i+1}:\n{chunk}" for i, chunk in enumerate(result['chunks'])])
            st.download_button(
                label="📝 Download Text",
                data=chunks_text,
                file_name=f"chunks_{result['document_id']}.txt",
                mime="text/plain"
            )
    
    # Sidebar information
    st.sidebar.header("ℹ️ Information")
    st.sidebar.info("""
    **Document Types Supported:**
    - Technical Documentation
    - Support Tickets
    - API References
    - Policies & Procedures
    - Tutorials & Guides
    - Code Files
    
    **Chunking Strategies:**
    - **Semantic**: Context-aware splitting
    - **Code-Aware**: Preserves code structure
    - **Hierarchical**: Maintains document hierarchy
    - **Fixed-Size**: Uniform chunk sizes
    """)
    
    # API info
    st.sidebar.header("🔧 API Information")
    doc_types = get_document_types()
    strategies = get_chunking_strategies()
    
    if doc_types:
        st.sidebar.success(f"✅ {len(doc_types)} document types available")
    
    if strategies:
        st.sidebar.success(f"✅ {len(strategies)} chunking strategies available")
    
    # Performance tips
    st.sidebar.header("💡 Performance Tips")
    st.sidebar.markdown("""
    **For Best Results:**
    - Use descriptive filenames
    - Include section headers
    - Maintain consistent formatting
    - Separate code from documentation
    """)

if __name__ == "__main__":
    main()