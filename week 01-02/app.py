import os
import sqlite3
import json
import random
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Drop table during development to apply schema clean
    cursor.execute('DROP TABLE IF EXISTS students')
    # Create students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            course TEXT NOT NULL,
            roll_number TEXT NOT NULL,
            phone TEXT NOT NULL,
            semester TEXT NOT NULL,
            admission_date TEXT NOT NULL
        )
    ''')
    conn.commit()

    # Seed data (database stores ISO format YYYY-MM-DD)
    sample_students = [
        ("Aarav Patel", "aarav.patel@iit.ac.in", "B.Tech Computer Science", "CSE/2024/041", "+91 98123 45678", "Semester V (3rd Year)", "2024-07-12"),
        ("Ananya Sharma", "ananya.sharma@nit.ac.in", "B.Tech Information Technology", "IT/2024/083", "+91 98234 56789", "Semester V (3rd Year)", "2024-07-15"),
        ("Vihaan Iyer", "vihaan.iyer@bits-pilani.ac.in", "MCA (Master of Computer Applications)", "MCA/2025/012", "+91 98345 67890", "Semester III (2nd Year)", "2025-08-01")
    ]
    cursor.executemany(
        'INSERT INTO students (name, email, course, roll_number, phone, semester, admission_date) VALUES (?, ?, ?, ?, ?, ?, ?)',
        sample_students
    )
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

@app.route('/')
def index():
    search_query = request.args.get('search', '').strip()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if search_query:
        # Search query matching columns
        like_query = f"%{search_query}%"
        cursor.execute('''
            SELECT * FROM students 
            WHERE name LIKE ? OR email LIKE ? OR course LIKE ? OR roll_number LIKE ? OR semester LIKE ? OR phone LIKE ? 
            ORDER BY id DESC
        ''', (like_query, like_query, like_query, like_query, like_query, like_query))
    else:
        cursor.execute('SELECT * FROM students ORDER BY id DESC')
        
    raw_students = cursor.fetchall()
    conn.close()
    
    # Format admission_date from YYYY-MM-DD to DD-MM-YYYY for frontend display
    students = []
    for row in raw_students:
        student = dict(row)
        try:
            dt = datetime.strptime(student['admission_date'], "%Y-%m-%d")
            student['admission_date_display'] = dt.strftime("%d-%m-%Y")
        except Exception:
            student['admission_date_display'] = student['admission_date']
        students.append(student)
        
    return render_template('index.html', students=students, search_query=search_query)

@app.route('/add', methods=['POST'])
def add_student():
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    course = request.form.get('course', '').strip()
    roll_number = request.form.get('roll_number', '').strip()
    phone = request.form.get('phone', '').strip()
    semester = request.form.get('semester', '').strip()
    admission_date = request.form.get('admission_date', '').strip()

    if name and email and course and roll_number and phone and semester and admission_date:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO students (name, email, course, roll_number, phone, semester, admission_date) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, email, course, roll_number, phone, semester, admission_date))
        conn.commit()
        conn.close()
        
    return redirect(url_for('index'))

@app.route('/edit/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        course = request.form.get('course', '').strip()
        roll_number = request.form.get('roll_number', '').strip()
        phone = request.form.get('phone', '').strip()
        semester = request.form.get('semester', '').strip()
        admission_date = request.form.get('admission_date', '').strip()
        
        if name and email and course and roll_number and phone and semester and admission_date:
            cursor.execute('''
                UPDATE students 
                SET name = ?, email = ?, course = ?, roll_number = ?, phone = ?, semester = ?, admission_date = ? 
                WHERE id = ?
            ''', (name, email, course, roll_number, phone, semester, admission_date, student_id))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))
            
    cursor.execute('SELECT * FROM students WHERE id = ?', (student_id,))
    student = cursor.fetchone()
    conn.close()
    
    if not student:
        return "Student not found", 404
        
    return render_template('edit.html', student=dict(student))

@app.route('/delete/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM students WHERE id = ?', (student_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/generate', methods=['POST'])
def generate_student():
    import urllib.request
    try:
        req = urllib.request.Request(
            'https://randomuser.me/api/?nat=in',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            user_info = res_data['results'][0]
            
            first_name = user_info['name']['first']
            last_name = user_info['name']['last']
            name = f"{first_name} {last_name}"
            email = user_info['email']
            phone = user_info['phone'].replace("-", " ").strip()
            
            courses = [
                "B.Tech Computer Science",
                "B.Tech Information Technology",
                "B.Tech Electronics & Comm.",
                "MCA (Master of Computer Applications)"
            ]
            course = random.choice(courses)
            
            # Generate roll number
            year = random.choice(["2023", "2024", "2025"])
            dept_code = "CSE" if "Computer" in course else ("IT" if "Information" in course else ("ECE" if "Electronics" in course else "MCA"))
            num = str(random.randint(10, 99))
            roll_number = f"{dept_code}/{year}/0{num}"
            
            # Semester selection
            semesters = [
                "Semester I (1st Year)",
                "Semester III (2nd Year)",
                "Semester V (3rd Year)",
                "Semester VII (4th Year)"
            ]
            semester = random.choice(semesters)
            
            # Parsing registration date (YYYY-MM-DD)
            admission_date = user_info['registered']['date'][:10]
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO students (name, email, course, roll_number, phone, semester, admission_date) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, email, course, roll_number, phone, semester, admission_date))
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"Error fetching API student: {e}")
        
    return redirect(url_for('index'))

@app.route('/clear', methods=['POST'])
def clear_students():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM students')
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
