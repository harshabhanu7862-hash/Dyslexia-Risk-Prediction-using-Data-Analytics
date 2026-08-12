from flask import Flask, render_template, request, redirect, url_for, session, flash
from tinydb import TinyDB, Query
import pandas as pd
import joblib
import os
import shap
import numpy as np

app = Flask(__name__)
app.secret_key = "dyslexia_secret"

db = TinyDB("users.json")
User = Query()

MODEL_PATH = r"C:\Users\harsh\PycharmProjects\UpdatedDyslexia\.venv\New_DyslexiaComputer Games_2026\ModelFiles\best_model.pkl"
SCALER_PATH = r"C:\Users\harsh\PycharmProjects\UpdatedDyslexia\.venv\New_DyslexiaComputer Games_2026\ModelFiles\scaler.pkl"
FEATURE_PATH = r"C:\Users\harsh\PycharmProjects\UpdatedDyslexia\.venv\New_DyslexiaComputer Games_2026\ModelFiles\feature_columns.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)

explainer = shap.TreeExplainer(model)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        username = request.form["username"]
        password = request.form["password"]
        email = request.form["email"]
        mobile = request.form["mobile"]

        if db.search(User.username == username):
            flash("Username already exists")
            return redirect(url_for("register"))

        db.insert({
            "name": name,
            "username": username,
            "password": password,
            "email": email,
            "mobile": mobile
        })

        flash("Registration successful")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = db.search((User.username == username) & (User.password == password))

        if user:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files["file"]

        if not file:
            flash("Please upload a CSV file")
            return redirect(url_for("dashboard"))

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df.columns = df.columns.map(str)

        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        scaled = scaler.transform(df.values)

        probs = model.predict_proba(scaled)[:, 1]
        preds = model.predict(scaled)


        shap_values = explainer.shap_values(scaled)

        explanations = []

        for i in range(len(df)):

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_row = shap_values[1][i]  # binary class 1
            else:
                shap_row = shap_values[i]

            shap_row = np.array(shap_row).flatten()

            feature_importance = list(zip(feature_columns, shap_row))
            feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

            top_features = feature_importance[:5]

            explanation_text = "<br>".join([
                f"<b>{feat}</b>: {'Increases Risk' if float(val) > 0 else 'Decreases Risk'} ({round(float(val), 3)})"
                for feat, val in top_features
            ])

            explanations.append(explanation_text)

        df["Prediction"] = [
            '<span style="color:red; font-weight:bold;">Dyslexia</span>'
            if p == 1
            else
            '<span style="color:green; font-weight:bold;">No Dyslexia</span>'
            for p in preds
        ]
        df["Probability"] = probs.round(3)

        df["Explanation (Top Factors)"] = explanations

        result_path = os.path.join("static", "results.csv")
        df.to_csv(result_path, index=False)

        return render_template(
            "result.html",
            table=df.to_html(classes="table table-bordered table-striped", index=False, escape=False)
        )

    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)