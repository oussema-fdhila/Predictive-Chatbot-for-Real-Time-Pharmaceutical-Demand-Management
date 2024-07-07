from app import app, db, User, bcrypt

# Fonction pour vérifier le mot de passe
def check_password(username, password):
    # Créez un contexte d'application
    with app.app_context():
        # Recherchez l'utilisateur par nom d'utilisateur
        user = User.query.filter_by(username=username).first()
        if user:
            # Vérifiez si le mot de passe correspond au hachage stocké
            if bcrypt.check_password_hash(user.password, password):
                return True
            else:
                return False
        else:
            return False

# Exemple d'utilisation
username = 'oussema'
password = 'ess1151925'

if check_password(username, password):
    print("Mot de passe correct")
else:
    print("Mot de passe incorrect")
