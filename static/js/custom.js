function showRegisterForm() {
    document.querySelector('.login-form-container').style.display = 'none';
    document.querySelector('#register-form').style.display = 'block';
}

function showLoginForm() {
    document.querySelector('#register-form').style.display = 'none';
    document.querySelector('.login-form-container').style.display = 'block';
}
