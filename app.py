from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

app = Flask(__name__)
app.secret_key = "supersecretkey"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Dummy user model for demo
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# Demo users
users = {"admin": User(1, "admin", "password123")}

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == int(user_id):
            return user
    return None

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    departments = [
        {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
        {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
        {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
        {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
        {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
        {"name": "Rancho Cordova", "key": "ranchocordova", "icon": "ranchocordova.jpeg"},
        {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
        {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
        {"name": "Office of Energy Infrastructure", "key": "energy", "icon": "energy.jpeg"},
    ]
    return render_template("dashboard.html", departments=departments)



@app.route("/ftb")
@login_required
def ftb():
    return render_template("ftb.html")

@app.route("/modules/<dept>")
@login_required
def modules(dept):
    icons = {
        "ftb": "ftb.jpeg",
        "dmv": "dmv.jpeg",
        "sanjose": "sanjose.jpeg",
        "edd": "edd.jpeg",
        "fiscal": "fiscal.jpeg",
        "ranchocordova": "ranchocordova.jpeg",
        "calpers": "calpers.jpeg",
        "cdfa": "cdfa.jpeg",
        "energy": "energy.jpeg",
    }
    display_names = {
        "ftb": "Franchise Tax Board (FTB)",
        "dmv": "Dept of Motor Vehicles",
        "sanjose": "City of San Jose",
        "edd": "Employment Development Dept",
        "fiscal": "Fi$cal",
        "ranchocordova": "Rancho Cordova",
        "calpers": "CalPERS",
        "cdfa": "CDFA",
        "energy": "Office of Energy Infrastructure",
    }
    # Fallback icon and name if an unknown dept is supplied
    company_icon = icons.get(dept, "default.jpeg")
    company_name = display_names.get(dept, "Department")
    return render_template("modules.html", company_icon=company_icon, company_name=company_name)

@app.route("/oops")
@login_required
def oops():
    return render_template("oops.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/insights")
@login_required
def insights():
    return render_template("insights.html")


if __name__ == "__main__":
    app.run(debug=True)
