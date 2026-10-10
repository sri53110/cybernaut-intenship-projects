const messageInput = document.getElementById('message');
const checkBtn = document.getElementById('checkBtn');
const errorBox = document.getElementById('error');
const resultBox = document.getElementById('result');
const resultLabel = document.getElementById('resultLabel');
const resultMessage = document.getElementById('resultMessage');
const confidence = document.getElementById('confidence');
const progressBar = document.getElementById('progressBar');

function showError(message) {
    errorBox.textContent = message;
    resultBox.hidden = true;
}

function clearError() {
    errorBox.textContent = '';
}

function setProgress(value) {
    progressBar.style.width = `${value}%`;
}

async function checkMessage() {
    const text = messageInput.value.trim();

    if (!text) {
        showError('Please enter a message.');
        return;
    }

    clearError();
    checkBtn.disabled = true;
    resultBox.hidden = false;
    resultBox.className = 'result';
    resultLabel.textContent = 'Checking...';
    resultMessage.textContent = 'Analyzing your message...';
    confidence.textContent = '—';
    setProgress(25);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Something went wrong.');
        }

        resultLabel.textContent = data.label;
        resultMessage.textContent = data.message;
        confidence.textContent = `${data.confidence}%`;
        resultBox.classList.add(data.label.toLowerCase() === 'spam' ? 'spam' : 'ham');
        setProgress(data.confidence);
    } catch (error) {
        showError(error.message || 'Unable to classify this message.');
        setProgress(0);
    } finally {
        checkBtn.disabled = false;
    }
}

function clearMessage() {
    messageInput.value = '';
    clearError();
    resultBox.hidden = true;
    setProgress(0);
}
