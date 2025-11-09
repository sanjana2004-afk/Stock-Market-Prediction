document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target;
    const submitBtn = document.getElementById('submit-btn');
    const spinner = submitBtn.querySelector('.spinner-border');
    const errorMessage = document.getElementById('error-message');
    const results = document.getElementById('results');
    
    // Show loading state
    submitBtn.disabled = true;
    spinner.classList.remove('d-none');
    errorMessage.classList.add('d-none');
    results.classList.add('d-none');
    
    try {
        const formData = new FormData(form);
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'An error occurred while processing your request');
        }
        
        // Update images with new data
        document.getElementById('ema-20-50').src = data.charts.ema_20_50 + '?t=' + new Date().getTime();
        document.getElementById('ema-100-200').src = data.charts.ema_100_200 + '?t=' + new Date().getTime();
        document.getElementById('prediction').src = data.charts.prediction + '?t=' + new Date().getTime();
        
        // Update download link
        document.getElementById('download-btn').href = data.dataset_link;
        
        // Show results
        results.classList.remove('d-none');
    } catch (error) {
        errorMessage.textContent = error.message;
        errorMessage.classList.remove('d-none');
    } finally {
        // Reset loading state
        submitBtn.disabled = false;
        spinner.classList.add('d-none');
    }
});