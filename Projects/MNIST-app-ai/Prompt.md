# ROLE: Senior AI/ML Engineer, Python Flask Architect & Premium UI/UX Developer

Act as a Senior Full Stack Python Developer, AI/ML Engineer, Software Architect, and UI/UX Designer.

I want you to build a complete, production-ready MNIST Handwritten Digit Recognition Web Application using Flask.

## 1. Project Objective

Build a modern AI-powered web application where users can:

1. Upload an image containing a handwritten digit.
2. Preview the uploaded image.
3. Click the Predict button.
4. Get the predicted digit (0–9).
5. View the prediction confidence percentage.
6. View probability distribution for all 10 digits.
7. Clear the uploaded image and previous prediction.
8. Upload another image and perform a new prediction.

The application must have a premium, modern SaaS dashboard experience.

## 2. Existing AI Model

IMPORTANT:

I already have a trained MNIST intelligent model file. file name is digit_brain_intel.h5

Before writing any code:

- Inspect the existing project directory.
- Identify the trained model file and its extension.
- Determine whether it is a TensorFlow/Keras model, HDF5 (.h5), SavedModel, PyTorch model, pickle file, or another supported format.
- Inspect existing preprocessing requirements if available.
- Identify the expected input shape, image dimensions, channel configuration, normalization, and output format.
- Reuse my existing trained model.
- DO NOT train a new model.
- DO NOT replace my existing model.
- DO NOT create a dummy prediction function.
- DO NOT hardcode predictions.

If model details cannot be determined from the file alone, explain what information is required before implementing the prediction integration.

Implement a reusable model-loading service that loads the model once when the Flask application starts.

## 3. Technology Stack

Backend:

- Python 3.12
- Flask
- TensorFlow/Keras or the framework identified from my existing model
- NumPy
- Pillow
- OpenCV only if necessary

Frontend:

- HTML5
- CSS3
- JavaScript
- Bootstrap 5 or custom responsive CSS
- Chart.js for prediction visualization

Production:

- Gunicorn
- Nginx
- Environment-based configuration
- Python logging

## 4. Premium UI/UX Requirements

Create a visually impressive AI application.

Design direction:

- Premium SaaS dashboard.
- Modern AI product aesthetic.
- Clean and minimal interface.
- Professional typography.
- Smooth animations and transitions.
- Proper spacing and alignment.
- Responsive design for desktop, tablet, and mobile.

Color palette:

- Primary background: #0B1120
- Secondary background: #111C32
- Primary accent: #6366F1
- Secondary accent: #06B6D4
- Success: #10B981
- Main text: #F8FAFC
- Secondary text: #94A3B8

Use gradients carefully. Avoid excessive colors and unnecessary visual elements.

## 5. Main Dashboard Layout

Create the following sections:

### A. Header

- Application logo/icon.
- Application name: MNIST AI.
- Subtitle: Handwritten Digit Recognition.
- AI Model Status indicator.
- Modern navigation.

### B. Welcome Section

Heading:
"Turn Handwriting Into Intelligence"

Description:
"Upload a handwritten digit and let AI identify it instantly."

Display a small AI-powered badge.

### C. Image Upload Section

Create a large drag-and-drop upload area.

Requirements:

- Drag and drop image support.
- Click to browse files.
- Support PNG, JPG, JPEG.
- Display image preview.
- Show uploaded filename.
- Show image dimensions and file size.
- Provide a remove image option.
- Provide clear validation messages.

### D. Prediction Controls

Provide:

- Predict Digit button.
- Clear / Reset button.

Predict button behavior:

- Disabled until a valid image is uploaded.
- Show loading animation during prediction.
- Prevent duplicate API submissions.
- Display meaningful errors.

Clear button behavior:

- Remove uploaded image.
- Clear image preview.
- Clear previous predicted digit.
- Clear confidence score.
- Reset probability chart.
- Reset validation messages.
- Return the page to its initial state.

### E. Prediction Result Section

Display the result in a premium result card.

Show:

- Predicted digit in large typography.
- Confidence percentage.
- Prediction status.
- Uploaded image thumbnail.
- Prediction execution time.

Example:

Predicted Digit: 7
Confidence: 98.75%

### F. Probability Distribution

Show probabilities for digits 0–9.

Use a horizontal bar chart or modern visualization.

Requirements:

- Display all 10 digit labels.
- Highlight the predicted digit.
- Display percentage values.
- Use Chart.js.
- Update the chart dynamically after prediction.
- Reset chart when Clear is clicked.

### G. Footer

Display:
"Powered by Machine Learning | MNIST Digit Recognition"

## 6. Image Preprocessing

This is one of the most important requirements.

The uploaded image must be processed exactly according to the existing model's training preprocessing.

Inspect and implement:

- Image resizing.
- Grayscale conversion.
- Pixel normalization.
- Channel dimensions.
- Input tensor shape.
- Inversion of pixel colors if required.
- Centering and padding if required by the training pipeline.

Do not blindly assume that every MNIST model expects identical preprocessing.

Ensure inference uses the same preprocessing logic as training.

## 7. Backend Architecture

Implement clean Flask architecture.

Create separate modules for:

- Application initialization.
- Configuration.
- Model loading.
- Image preprocessing.
- Prediction service.
- API routes.
- Error handling.
- Logging.

Create the following APIs:

POST /api/predict/

- Accept multipart image upload.
- Validate the image.
- Preprocess the image.
- Run inference.
- Return predicted digit.
- Return confidence percentage.
- Return probability distribution for all digits.
- Return prediction execution time.

GET /api/health/

- Return application health.
- Return model loading status.

GET /api/model-info/

- Return model metadata where available.

Use JSON responses with consistent success and error formats.

## 8. API Response Example

Successful prediction:

{
"success": true,
"message": "Prediction completed successfully",
"data": {
"predicted_digit": 7,
"confidence": 98.75,
"probabilities": {
"0": 0.01,
"1": 0.02,
"2": 0.05,
"3": 0.10,
"4": 0.03,
"5": 0.04,
"6": 0.05,
"7": 98.75,
"8": 0.80,
"9": 0.15
},
"execution_time_ms": 45
}
}

The probabilities must be calculated from the actual model output, not hardcoded.

## 9. Production-Ready Folder Structure

Create a modular, maintainable project structure.

Separate:

- Flask application.
- Configuration.
- AI model.
- Model services.
- Image preprocessing.
- API routes.
- Templates.
- Static CSS.
- Static JavaScript.
- Tests.
- Deployment configuration.
- Documentation.

Keep the model file in a dedicated models directory.

Do not put the entire application inside app.py.

## 10. Security and Validation

Implement:

- Secure filename handling.
- File extension validation.
- MIME/content validation.
- Maximum upload size.
- Image dimension validation.
- Corrupted image handling.
- Request error handling.
- Environment variables.
- No hardcoded production secrets.
- Safe temporary file handling.
- Appropriate security headers.
- Production DEBUG=False.

Do not permanently store uploaded images unless explicitly required.

## 11. Performance

- Load the AI model once.
- Avoid loading the model for every request.
- Reuse preprocessing utilities.
- Optimize inference execution.
- Handle multiple requests safely.
- Configure Gunicorn workers appropriately for the model framework and available resources.
- Avoid unnecessary image processing.

## 12. Testing

Create tests for:

- Flask application startup.
- Model loading.
- Valid image prediction.
- Invalid file upload.
- Unsupported image formats.
- Corrupted images.
- Oversized images.
- Missing file.
- Health endpoint.
- Prediction API response.
- Clear/reset frontend behavior.

Use pytest.

## 13. Deployment Readiness

Prepare the application for deployment on Linux/AWS EC2.

Include:

- requirements.txt with compatible dependencies.
- .env.example.
- .gitignore.
- Gunicorn configuration.
- Nginx configuration example.
- Production startup commands.
- README.md.
- Deployment instructions.

Never commit the actual .env file or sensitive credentials.

## 14. Development Workflow

Follow these steps:

PHASE 1: Inspect the existing project and model file.

PHASE 2: Explain the model format, preprocessing requirements, and proposed architecture.

PHASE 3: Create the complete folder structure.

PHASE 4: Implement the backend and model integration.

PHASE 5: Implement the premium frontend.

PHASE 6: Integrate the frontend with Flask APIs.

PHASE 7: Implement validation, security, and error handling.

PHASE 8: Run automated tests.

PHASE 9: Start the application and verify actual image predictions.

PHASE 10: Prepare production deployment documentation.

IMPORTANT DEVELOPMENT RULES:

- Work directly inside my selected project directory.
- Create actual files, not just code snippets.
- Do not overwrite my existing trained model.
- Do not generate fake prediction results.
- Do not skip implementation steps.
- Do not leave TODO comments for core functionality.
- Keep code modular and reusable.
- Verify imports and dependencies.
- Run the application and fix errors.
- Ensure the Predict and Clear buttons actually work.
- Ensure the frontend is fully responsive.

First inspect my existing files and provide the proposed implementation plan. Then proceed with implementation.
