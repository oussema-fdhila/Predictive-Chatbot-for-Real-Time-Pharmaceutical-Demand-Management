from app import app, db, User

# Créez un contexte d'application
with app.app_context():
    # Obtenez tous les utilisateurs
    users = User.query.all()
    
    # Imprimez les utilisateurs
    for user in users:
        print(f'ID: {user.id}, Username: {user.username}')
