
from flask import Flask, render_template, request, redirect
import sqlite3
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# ----------------------------------
# PROJECT PATHS
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "students.db"
MODEL_PATH = BASE_DIR / "placement_model.pkl"

FEATURE_COLUMNS = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks"
]


# ----------------------------------
# DATABASE CREATION AND UPGRADATION
# ----------------------------------

def create_database():

    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            usn TEXT,
            cgpa REAL,
            attendance REAL,
            backlogs INTEGER,
            skills TEXT
        )
    """)

    ml_columns = {
        "internships": "INTEGER",
        "projects": "INTEGER",
        "workshops": "INTEGER",
        "aptitude": "REAL",
        "soft_skills": "REAL",
        "extracurricular": "TEXT",
        "training": "TEXT",
        "ssc": "REAL",
        "hsc": "REAL",
        "predicted_status": "TEXT"
    }

    existing_columns = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(students)"
        ).fetchall()
    }

    for column, data_type in ml_columns.items():

        if column not in existing_columns:
            conn.execute(
                f"ALTER TABLE students ADD COLUMN {column} {data_type}"
            )

    conn.commit()
    conn.close()


# ----------------------------------
# DATABASE CONNECTION
# ----------------------------------

def get_connection():

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    return conn


# ----------------------------------
# ELIGIBILITY CHECK
# ----------------------------------

def check_eligibility(cgpa, attendance, backlogs):

    return (
        cgpa >= 7.0
        and attendance >= 75
        and backlogs == 0
    )


# ----------------------------------
# FORM DATA VALIDATION
# ----------------------------------

def get_student_data(form, existing=None):

    if existing is None:
        existing = {}

    name = form.get("name", "").strip()
    usn = form.get("usn", "").strip()
    skills = form.get("skills", "").strip()

    if not name or not usn or not skills:
        raise ValueError("Please fill all required fields.")

    cgpa = float(form["cgpa"])
    attendance = float(form["attendance"])
    backlogs = int(form["backlogs"])

    if not (0 <= cgpa <= 10):
        raise ValueError("CGPA must be between 0 and 10.")

    if not (0 <= attendance <= 100):
        raise ValueError("Attendance must be between 0 and 100.")

    if backlogs < 0:
        raise ValueError("Backlogs cannot be negative.")

    data = {
        "name": name,
        "usn": usn,
        "cgpa": cgpa,
        "attendance": attendance,
        "backlogs": backlogs,
        "skills": skills
    }

    ml_fields = {
        "internships": ("int", 0, None),
        "projects": ("int", 0, None),
        "workshops": ("int", 0, None),
        "aptitude": ("float", 0, 100),
        "soft_skills": ("float", 0, 5),
        "extracurricular": ("choice", "No", None),
        "training": ("choice", "No", None),
        "ssc": ("float", 0, 100),
        "hsc": ("float", 0, 100)
    }

    for field, (kind, minimum, maximum) in ml_fields.items():

        raw_value = form.get(field)

        if raw_value is None:
            data[field] = (
                existing[field] if field in existing else None
            )
            continue

        raw_value = raw_value.strip()

        if raw_value == "":
            data[field] = None
            continue

        if kind == "int":
            value = int(raw_value)

        elif kind == "float":
            value = float(raw_value)

        else:
            if raw_value not in ("Yes", "No"):
                raise ValueError(f"Invalid value for {field}.")

            value = raw_value

        if kind != "choice":

            if value < minimum:
                raise ValueError(f"Invalid value for {field}.")

            if maximum is not None and value > maximum:
                raise ValueError(f"Invalid value for {field}.")

        # Save the validated value
        data[field] = value

    return data


# ----------------------------------
# DASHBOARD
# ----------------------------------

@app.route("/")
def home():

    search = request.args.get("search", "").strip()

    conn = get_connection()

    all_students = conn.execute(
        "SELECT * FROM students"
    ).fetchall()

    students = conn.execute("""
        SELECT * FROM students
        WHERE name LIKE ? OR usn LIKE ?
        ORDER BY id DESC
    """, (f"%{search}%", f"%{search}%")).fetchall()

    conn.close()

    total = len(all_students)

    eligible = sum(
        check_eligibility(
            s["cgpa"],
            s["attendance"],
            s["backlogs"]
        )
        for s in all_students
    )

    avg_cgpa = (
        sum(s["cgpa"] for s in all_students) / total
        if total else 0
    )

    avg_attendance = (
        sum(s["attendance"] for s in all_students) / total
        if total else 0
    )

    placed_count = sum(
        s["predicted_status"] == "Placed"
        for s in all_students
    )

    not_placed_count = sum(
        s["predicted_status"] == "NotPlaced"
        for s in all_students
    )

    predicted_count = placed_count + not_placed_count

    return render_template(
        "index.html",
        students=students,
        total=total,
        eligible=eligible,
        avg_cgpa=round(avg_cgpa, 2),
        avg_attendance=round(avg_attendance, 2),
        placed_count=placed_count,
        not_placed_count=not_placed_count,
        predicted_count=predicted_count,
        check_eligibility=check_eligibility,
        search=search
    )


# ----------------------------------
# REGISTER STUDENT
# ----------------------------------

@app.route("/register", methods=["POST"])
def register():

    conn = get_connection()

    try:

        data = get_student_data(request.form)

        existing = conn.execute(
            "SELECT id FROM students WHERE usn = ?",
            (data["usn"],)
        ).fetchone()

        if existing:
            conn.close()
            return "This USN is already registered.", 400

        conn.execute("""
            INSERT INTO students (
                name, usn, cgpa, attendance, backlogs, skills,
                internships, projects, workshops, aptitude,
                soft_skills, extracurricular, training, ssc, hsc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(data[field] for field in [
            "name", "usn", "cgpa", "attendance", "backlogs",
            "skills", "internships", "projects", "workshops",
            "aptitude", "soft_skills", "extracurricular",
            "training", "ssc", "hsc"
        ]))

        conn.commit()

    except (ValueError, KeyError):
        conn.close()
        return "Invalid student details. Please check your inputs.", 400

    conn.close()

    return redirect("/")


# ----------------------------------
# DELETE STUDENT
# ----------------------------------

@app.route("/delete/<int:student_id>", methods=["POST"])
def delete_student(student_id):

    conn = get_connection()

    conn.execute(
        "DELETE FROM students WHERE id = ?",
        (student_id,)
    )

    conn.commit()
    conn.close()

    return redirect("/")


# ----------------------------------
# EDIT STUDENT
# ----------------------------------

@app.route("/edit/<int:student_id>", methods=["GET", "POST"])
def edit_student(student_id):

    conn = get_connection()

    student = conn.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    ).fetchone()

    if student is None:
        conn.close()
        return "Student not found.", 404

    if request.method == "POST":

        try:

            data = get_student_data(request.form, student)

            duplicate = conn.execute("""
                SELECT id FROM students
                WHERE usn = ? AND id != ?
            """, (data["usn"], student_id)).fetchone()

            if duplicate:
                conn.close()
                return "USN already registered.", 400

            conn.execute("""
                UPDATE students SET
                    name = ?,
                    usn = ?,
                    cgpa = ?,
                    attendance = ?,
                    backlogs = ?,
                    skills = ?,
                    internships = ?,
                    projects = ?,
                    workshops = ?,
                    aptitude = ?,
                    soft_skills = ?,
                    extracurricular = ?,
                    training = ?,
                    ssc = ?,
                    hsc = ?,
                    predicted_status = NULL
                WHERE id = ?
            """, (
                data["name"],
                data["usn"],
                data["cgpa"],
                data["attendance"],
                data["backlogs"],
                data["skills"],
                data["internships"],
                data["projects"],
                data["workshops"],
                data["aptitude"],
                data["soft_skills"],
                data["extracurricular"],
                data["training"],
                data["ssc"],
                data["hsc"],
                student_id
            ))

            conn.commit()
            conn.close()

            return redirect("/")

        except (ValueError, KeyError):
            conn.close()
            return "Invalid student details. Please check your inputs.", 400

    conn.close()

    return render_template(
        "edit.html",
        student=student
    )


# ----------------------------------
# ML PREDICTION FUNCTION
# ----------------------------------

def predict_placement(student_data):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Placement model not found. Please train the model first."
        )

    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame(
        [student_data],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(input_data)[0]

    return str(prediction)


# ----------------------------------
# MANUAL PLACEMENT PREDICTION
# ----------------------------------

@app.route("/predict", methods=["GET", "POST"])
def manual_prediction():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            form_data = {
                "name": "Prediction",
                "usn": "TEMP",
                "skills": "Prediction"
            }

            form_data.update(request.form)

            data = get_student_data(form_data)

            feature_data = {
                "CGPA": data["cgpa"],
                "Internships": data["internships"],
                "Projects": data["projects"],
                "Workshops/Certifications": data["workshops"],
                "AptitudeTestScore": data["aptitude"],
                "SoftSkillsRating": data["soft_skills"],
                "ExtracurricularActivities": data["extracurricular"],
                "PlacementTraining": data["training"],
                "SSC_Marks": data["ssc"],
                "HSC_Marks": data["hsc"]
            }

            if any(value is None for value in feature_data.values()):
                raise ValueError("Please complete all prediction fields.")

            prediction = predict_placement(feature_data)

        except (ValueError, KeyError, FileNotFoundError) as e:
            error = str(e)

    return render_template(
        "predict.html",
        prediction=prediction,
        error=error
    )


# ----------------------------------
# PREDICT REGISTERED STUDENT
# ----------------------------------

@app.route("/predict/<int:student_id>")
def predict_registered_student(student_id):

    conn = get_connection()

    student = conn.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    ).fetchone()

    if student is None:
        conn.close()
        return "Student not found.", 404

    required_fields = [
        "internships",
        "projects",
        "workshops",
        "aptitude",
        "soft_skills",
        "extracurricular",
        "training",
        "ssc",
        "hsc"
    ]

    missing_fields = [
        field for field in required_fields
        if student[field] is None
    ]

    if missing_fields:
        conn.close()

        return (
            "Student details are incomplete. "
            "Please update all placement details before prediction. "
            f'<a href="/edit/{student_id}">Edit Student Details</a>'
        )

    feature_data = {
        "CGPA": student["cgpa"],
        "Internships": student["internships"],
        "Projects": student["projects"],
        "Workshops/Certifications": student["workshops"],
        "AptitudeTestScore": student["aptitude"],
        "SoftSkillsRating": student["soft_skills"],
        "ExtracurricularActivities": student["extracurricular"],
        "PlacementTraining": student["training"],
        "SSC_Marks": student["ssc"],
        "HSC_Marks": student["hsc"]
    }

    try:

        prediction = predict_placement(feature_data)

        # Save prediction in database
        conn.execute("""
            UPDATE students
            SET predicted_status = ?
            WHERE id = ?
        """, (prediction, student_id))

        conn.commit()

    except (ValueError, FileNotFoundError) as e:
        conn.close()
        return str(e), 500

    conn.close()

    return render_template(
        "predict.html",
        prediction=prediction,
        error=None,
        student=student
    )


# ----------------------------------
# START APPLICATION
# ----------------------------------

create_database()

if __name__ == "__main__":
    app.run(debug=True)
