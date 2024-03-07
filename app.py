from flask import Flask, render_template, request, redirect, url_for
from PyPDF2 import PdfReader
import io
from flask_sqlalchemy import SQLAlchemy

# Importing functions from combined_code module
from combined_code import get_mca_questions

app = Flask(__name__)
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Route for the main page
@app.route('/')
def index():
    return render_template('main.html')

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials. Please try again.")
    return render_template('login.html')

# Route for the signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
    
        if existing_user:
            return render_template('signup.html', error="Username already exists. Please choose a different one.")
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('dashboard'))
    return render_template('signup.html')

# Route for the dashboard page
@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html')

# Route for uploading PDF and generating questions
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']
    if file.filename == '':
        return 'No file selected'

    num_questions = int(request.form['num_questions'])
    if file:
        pdf_text = read_pdf(file)
        mcq_questions = process_text(pdf_text, num_questions)
        return render_template('questions.html', questions=mcq_questions)


# Helper function to read PDF content
def read_pdf(file):
    pdf_reader = PdfReader(io.BytesIO(file.read()))
    text = ''
    num_pages = len(pdf_reader.pages)
    for page_number in range(num_pages):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()
    return text

# Helper function to process text and generate MCQs
def process_text(input_text, num_questions):
    mcq_questions = get_mca_questions(input_text, num_questions)
    return mcq_questions

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
