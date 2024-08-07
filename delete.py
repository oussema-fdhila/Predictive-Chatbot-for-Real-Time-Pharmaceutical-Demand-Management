from flask import Flask, request, jsonify
from app import app, db, User

@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user_to_delete = User.query.get(user_id)

    if user_to_delete:
        db.session.delete(user_to_delete)
        db.session.commit()
        return jsonify({'message': f'User with ID {user_id} deleted successfully'}), 200
    else:
        return jsonify({'error': 'User not found'}), 404
