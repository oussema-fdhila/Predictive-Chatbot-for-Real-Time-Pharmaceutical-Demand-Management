from flask import Flask, render_template, url_for, redirect, flash, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from datasets import Dataset

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  # Chemin vers votre base de données SQLite
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True  # Optionnel : désactive le suivi des modifications
app.config['SECRET_KEY'] = 'thisisasecretkey'

dataset_path = 'forecast_data_produit_secteur_temp.xlsx'
df = pd.read_excel(dataset_path)

# Préparation des données pour l'entraînement
def prepare_data(df):
    data = []
    for idx, row in df.iterrows():
        produit = row['Produit']
        secteur = row['Secteur']
        gouvernorat = row['Gouvernorat']
        marche = row['Marché']
        previsions = row['Forecast']
        questions = [
            f"Quels sont les prévisions pour le produit {produit} dans le secteur {secteur}?",
            f"Quel est le secteur du produit {produit}?",
            f"Quel est le marché {marche} dans le secteur {secteur}?",
            f"Quel est le contexte des prévisions pour le produit {produit}?",
            f"Quel est le gouvernorat {gouvernorat} du secteur {secteur}?"
        ]
        for question in questions:
            data.append({
                "question": str(question),
                "context": str(previsions),  # Utilisation des prévisions réelles comme contexte
                "answers": {
                    "text": [str(previsions)],  # Réponse en tant que chaîne de caractères
                    "answer_start": [0]  # Utilisation de 0 comme position de début (pour simplification)
                }
            })
    return data

# Préparation des données
data = prepare_data(df)
dataset = Dataset.from_pandas(pd.DataFrame(data))

# Division des données en ensembles d'entraînement et de validation
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
valid_dataset = train_test_split['test']

# Charger le tokenizer et le modèle
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Tokenizer les données
def preprocess_function(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        answer_start = examples["answers"][i]["answer_start"][0]
        answer_end = answer_start + len(examples["answers"][i]["text"][0])

        start_position = cls_index
        end_position = cls_index

        for idx, (start, end) in enumerate(offsets):
            if start <= answer_start < end:
                start_position = idx
            if start < answer_end <= end:
                end_position = idx

        start_positions.append(start_position)
        end_positions.append(end_position)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    return tokenized_examples

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["question", "context", "answers"])
tokenized_valid = valid_dataset.map(preprocess_function, batched=True, remove_columns=["question", "context", "answers"])

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.1
)

# Créer l'objet Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer
)

# Entraîner le modèle
trainer.train()

# Évaluation du modèle
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Fonction pour poser une question au modèle
def ask_question(question, context):
    encoded_dict = tokenizer(question, context, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**encoded_dict)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded_dict.input_ids[0][answer_start:answer_end]))
    return answer


db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')

class RegisterForm_fr(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Nom d\'utilisateur"})
    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Mot de passe"})
    submit = SubmitField('S\'inscrire')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'Ce nom d\'utilisateur existe déjà. Veuillez choisir un autre!')        

class LoginForm(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

class LoginForm_fr(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Nom d\'utilisateur"})
    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Mot de passe"})
    submit = SubmitField('Se connecter')    

@app.route('/')
def home():
    login_form = LoginForm()
    register_form = RegisterForm()
    user = current_user if current_user.is_authenticated else None
    return render_template('index_en.html', form=login_form, register_form=register_form, user=user)

@app.route('/fr')
def home_fr():
    login_form = LoginForm_fr()
    register_form = RegisterForm_fr()
    user = current_user if current_user.is_authenticated else None
    return render_template('index_fr.html', form=login_form, register_form=register_form, user=user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Verify your data !', 'error')
    return render_template('login.html', form=form)

@app.route('/login_fr', methods=['GET', 'POST'])
def login_fr():
    form = LoginForm_fr()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('home_fr'))
        else:
            flash('Veuillez vérifier vos données !', 'error')
    return render_template('login_fr.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/logout_fr', methods=['GET', 'POST'])
@login_required
def logout_fr():
    logout_user()
    return redirect(url_for('home_fr'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('This username is already taken. Please choose another one.', 'error')            
            return redirect(url_for('home'))
        else:
            hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')  # decode hashed password to string
            new_user = User(username=form.username.data, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('You are now registered! Log in to access your account.', 'success')
            return redirect(url_for('home'))
    return render_template('register.html', form=form)

@app.route('/register_fr', methods=['GET', 'POST'])
def register_fr():
    form = RegisterForm_fr()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Ce nom d\'utilisateur est déjà pris. Veuillez en choisir un autre.', 'error')
            return redirect(url_for('home_fr'))
        else:
            hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')  # decode hashed password to string
            new_user = User(username=form.username.data, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Vous êtes maintenant inscrit ! Connectez-vous pour accéder à votre compte.', 'success')
            return redirect(url_for('home_fr'))
    return render_template('register_fr.html', form=form)

@app.route('/chatbot', methods=['POST'])
@login_required
def chatbot():
    data = request.get_json()
    question = data.get('message')
    produit = data.get('produit').strip().lower()
    secteur = data.get('secteur').strip().lower()

    # Debug prints
    print(f"Produit reçu: {produit}")
    print(f"Secteur reçu: {secteur}")

    # Nettoyage et formatage des données dans le DataFrame pour correspondre à la requête
    df['Produit'] = df['Produit'].str.strip().str.lower()
    df['Secteur'] = df['Secteur'].str.strip().str.lower()

    filtered_data = df[(df['Produit'] == produit) & (df['Secteur'] == secteur)]['Forecast']

    # Debug prints
    print(f"Filtered data: {filtered_data}")

    if not filtered_data.empty:
        filtered_data = pd.to_numeric(filtered_data, errors='coerce')
        context = str(filtered_data.mean())
        response = ask_question(question, context)
        response_text = f"Voici la prévision pour {produit} et {secteur} : {response}"
    else:
        response_text = f"Aucune donnée trouvée pour {produit} et {secteur}."

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.static_folder = 'C:\\Users\\User\\Desktop\\flask_deploy\\static\\css'
    app.run(debug=True)
