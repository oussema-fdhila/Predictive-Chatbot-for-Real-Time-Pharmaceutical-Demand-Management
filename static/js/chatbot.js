document.addEventListener('DOMContentLoaded', function() {
    document.getElementById("open-chatbot").addEventListener("click", function() {
        document.getElementById("chatbot-container").classList.remove("hidden");
        document.getElementById("open-chatbot").style.display = 'none';
    });

    document.getElementById("close-chatbot").addEventListener("click", function() {
        document.getElementById("chatbot-container").classList.add("hidden");
        document.getElementById("open-chatbot").style.display = 'block';
    });

    document.getElementById('send-chatbot-message').addEventListener('click', function() {
        let userMessage = document.getElementById('chatbot-input').value;
        let product = document.getElementById('chatbot-product').value;
        let sector = document.getElementById('chatbot-sector').value;

        if (userMessage.trim() !== "" && product.trim() !== "" && sector.trim() !== "") {
            addMessageToChat("You: " + userMessage);
            document.getElementById('chatbot-input').value = "";

            // Call the Python chatbot
            callPythonChatbot(userMessage, product, sector);
        }
    });

    function addMessageToChat(message) {
        let messageElement = document.createElement('div');
        messageElement.textContent = message;
        document.getElementById('chatbot-messages').appendChild(messageElement);
    }

    function callPythonChatbot(message, product, sector) {
        fetch('/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message, produit: product, secteur: sector })
        })
        .then(response => response.json())
        .then(data => {
            addMessageToChat("Bot: " + data.response);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});