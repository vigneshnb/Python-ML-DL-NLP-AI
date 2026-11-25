# ROLE

Act as a Senior Python Flask Developer, AI/ML Engineer, Computer Vision Engineer, Software Architect, Product Designer, and DevOps Engineer.

I want you to build a complete production-ready CIFAR-10 Image Classification Web Application using Flask and my existing trained model.

My application must allow a user to upload an image and predict the image category using my existing trained model.

# 1. PROJECT OBJECTIVE

Build a modern AI-powered image classification web application using Flask.

Users must be able to:

1. Upload an image from their computer.
2. Preview the uploaded image.
3. Predict the image category using my existing trained model.
4. Display the predicted class name.
5. Display the prediction confidence percentage.
6. Display probability scores for all 10 CIFAR-10 classes.
7. Display a visual confidence chart.
8. Clear the uploaded image and previous prediction.
9. Upload a different image and predict again.
10. Use the application on desktop, tablet, and mobile devices.

The application must have premium SaaS-level UI/UX.

# 2. EXISTING TRAINED MODEL

My existing model file is:

cifar10_predict.h5

The model is already trained.

IMPORTANT:

- Do not train another model.
- Do not create a CNN.
- Do not create a new model architecture.
- Do not replace my existing model.
- Do not generate dummy predictions.
- Do not hardcode predicted results.
- Reuse the actual trained model file.
- Inspect the model before implementing integration.
- Determine the model's expected input shape.
- Determine the model's output shape.
- Determine whether the model uses Softmax or produces logits.
- Determine its expected image preprocessing.
- Determine whether the model expects float32, normalized pixels, or another format.

Load the existing model using TensorFlow/Keras where compatible.

If the model loading fails, inspect the error and investigate compatible TensorFlow/Keras versions before attempting any model modification.

Do not modify or overwrite the existing model.

# 3. CIFAR-10 CLASS LABELS

Use the standard CIFAR-10 class mapping:

0 - Airplane
1 - Automobile
2 - Bird
3 - Cat
4 - Deer
5 - Dog
6 - Frog
7 - Horse
8 - Ship
9 - Truck

Implement this mapping in a dedicated constants module.

Never hardcode the class mapping throughout multiple files.

# 4. TECHNOLOGY STACK

Backend:
- Python 3.12
- Flask
- TensorFlow/Keras
- NumPy
- Pillow
- Python-dotenv

Frontend:
- HTML5
- CSS3
- JavaScript
- Bootstrap 5 or custom responsive CSS
- Chart.js

Production:
- Gunicorn
- Nginx
- Linux
- Environment-based configuration
- Python logging

# 5. PREMIUM UI/UX DESIGN

Design a premium AI image classification SaaS application.

Design direction:

- Premium dark UI.
- Modern AI dashboard.
- Clean and professional layout.
- Minimal and visually appealing.
- Responsive.
- Smooth animations.
- Subtle gradients.
- Elegant glass-style cards where appropriate.
- Professional typography.
- Proper spacing.
- Accessible color contrast.

Color palette:

Primary background: #0B1120
Secondary background: #111C32
Card background: #172238
Primary accent: #6366F1
Secondary accent: #06B6D4
Success: #10B981
Main text: #F8FAFC
Secondary text: #94A3B8
Border: #263449

Use gradients carefully.

Do not create a cluttered UI.

Do not overuse neon colors.

# 6. APPLICATION LAYOUT

Create the following sections.

## A. Header

Include:

- Application icon.
- Application name: CIFAR Vision AI.
- Subtitle: AI Image Classification.
- AI Model Status.
- Clean navigation.
- Responsive mobile header.

## B. Hero section

Heading:

"Discover What's Inside Your Image"

Description:

"Upload an image and let AI identify the objects and categories in seconds."

Display a badge:

"Powered by CIFAR-10"

Add subtle visual elements to enhance the premium AI aesthetic.

## C. Upload area

Create a premium drag-and-drop image upload area.

Requirements:

- Drag-and-drop support.
- Click to browse.
- Support PNG, JPG, JPEG and WEBP.
- Display the uploaded image preview.
- Display filename.
- Display file size.
- Display image dimensions.
- Remove image option.
- Validation messages.
- Responsive design.

Provide an initial empty state with a clear call to action.

## D. Prediction buttons

Provide:

- Predict Image button.
- Clear / Reset button.

Predict button:

- Disabled before image upload.
- Enabled after valid image upload.
- Show loading state during prediction.
- Prevent repeated submissions while prediction is in progress.
- Display appropriate API errors.
- Update the result UI after prediction.

Clear button:

- Clear the uploaded image.
- Clear the image preview.
- Clear the predicted class.
- Clear confidence.
- Clear probability chart.
- Clear prediction execution time.
- Reset UI validation messages.
- Reset prediction state.
- Allow uploading a new image.

The Clear button must work correctly without requiring a page refresh.

## E. Prediction result card

After prediction, display:

- Predicted class name.
- Predicted class number.
- Confidence percentage.
- Uploaded image thumbnail.
- Prediction execution time.
- A visually prominent result card.

Example layout:

Predicted Object
Dog

Class: 5
Confidence: 94.87%
Prediction Time: 32 ms

The example values above are only UI placeholders. Never use them as actual prediction results.

## F. Confidence visualization

Create a probability distribution visualization for all 10 classes.

Use Chart.js.

Requirements:

- Horizontal bar chart.
- Display all 10 class names.
- Display percentages.
- Highlight the predicted class.
- Update dynamically after prediction.
- Reset chart after Clear.
- Responsive chart.

Display a secondary list/table containing all classes and their probability scores.

## G. Prediction history

Optionally create a client-side recent predictions section for the current browser session.

Requirements:

- Display thumbnail.
- Display predicted class.
- Display confidence.
- Allow clearing history.
- Do not persist uploaded images on the server.
- Use local browser state only if needed.
- Avoid storing sensitive user-uploaded content unnecessarily.

If this feature introduces excessive complexity, implement it as a lightweight optional feature.

## H. Footer

Display:

"Powered by CIFAR-10 | AI Image Classification"

# 7. IMAGE PREPROCESSING

Image preprocessing must match the existing trained model.

Before implementing preprocessing:

Inspect the model input shape.

Determine:

- Expected image height.
- Expected image width.
- Number of channels.
- RGB/BGR requirement.
- Pixel value range.
- Normalization.
- Resizing behavior.
- Data type.
- Batch dimension.

The standard CIFAR-10 image format is typically:

32 x 32 pixels
3 RGB channels

However, do not assume the model expects this format until the model is inspected.

Implement preprocessing using Pillow and NumPy.

The preprocessing service must:

1. Validate the image.
2. Open the uploaded image safely.
3. Convert to RGB if required by the model.
4. Resize according to the actual model input shape.
5. Preserve the model's expected color channel order.
6. Normalize pixel values according to training requirements.
7. Create the batch dimension.
8. Return the expected input tensor.

Do not blindly apply normalization.

Do not blindly divide by 255 if the training pipeline used different preprocessing.

Do not assume the model expects RGB if it was trained with a different channel format.

Document all preprocessing decisions.

# 8. MODEL INTEGRATION

Create a dedicated model-loading service.

Requirements:

- Load cifar10_predict.h5 once per application process.
- Avoid loading the model on every request.
- Expose reusable prediction functionality.
- Validate input shape.
- Validate output shape.
- Ensure 10 output classes.
- Handle model errors.
- Handle inference exceptions.
- Support concurrency safely where appropriate.

Inspect the model output.

If the model produces Softmax probabilities, use them directly.

If the model produces logits, apply numerically stable Softmax.

Do not apply Softmax twice.

If output interpretation cannot be determined, document the limitation and investigate the model architecture.

# 9. PREDICTION RESULT

Return:

- Predicted class index.
- Predicted class name.
- Confidence percentage.
- Probability distribution for all 10 classes.
- Prediction execution time.

Calculate prediction using the actual trained model.

Never fabricate predictions or confidence values.

# 10. BACKEND API

Implement:

POST /api/predict/

Accept a multipart image upload.

Validate:

- Missing file.
- Unsupported file type.
- Corrupted image.
- File size.
- Image dimensions.
- Model availability.

Return the predicted class and probabilities.

GET /api/health/

Return:

- Application status.
- Model loading status.
- Application name.

GET /api/model-info/

Return:

- Model name.
- Input shape.
- Output shape.
- Class names.
- Framework information where available.

Use consistent JSON response structures.

# 11. API RESPONSE FORMAT

Example:

{
  "success": true,
  "message": "Image classified successfully",
  "data": {
    "predicted_class": 5,
    "predicted_label": "Dog",
    "confidence": 94.87,
    "probabilities": {
      "Airplane": 0.20,
      "Automobile": 0.10,
      "Bird": 0.15,
      "Cat": 1.20,
      "Deer": 0.30,
      "Dog": 94.87,
      "Frog": 0.50,
      "Horse": 0.40,
      "Ship": 1.00,
      "Truck": 1.28
    },
    "execution_time_ms": 32
  }
}

The example is for response structure only.

The actual probability distribution must come from my model.

# 12. FOLDER STRUCTURE

Create a production-ready folder structure.

Separate:

- Flask application.
- Configuration.
- Model integration.
- Image preprocessing.
- API routes.
- Constants.
- Templates.
- Static CSS.
- Static JavaScript.
- Tests.
- Deployment files.
- Documentation.

Keep cifar10_predict.h5 inside the dedicated models directory.

Do not put all application logic inside a single app.py file.

# 13. SECURITY

Implement:

- File size validation.
- Image dimensions validation.
- Extension validation.
- Image content validation.
- Safe image processing.
- Secure filename handling if saving files.
- No unnecessary permanent upload storage.
- Appropriate HTTP security headers.
- Environment configuration.
- Production DEBUG=False.
- Safe exception handling.
- Logging.

Do not expose internal server errors or filesystem paths in API responses.

Do not use uploaded filenames as filesystem paths.

# 14. PERFORMANCE

- Load the model once.
- Avoid unnecessary file writes.
- Avoid repeated model initialization.
- Efficient image preprocessing.
- Efficient inference.
- Prevent duplicate API submissions.
- Optimize resource usage.
- Configure Gunicorn appropriately for TensorFlow and server memory.

Do not blindly configure multiple Gunicorn workers if model memory consumption is high.

Explain the deployment memory implications.

# 15. AUTOMATED TESTING

Use pytest.

Test:

- Application initialization.
- Model loading.
- Model input shape.
- Model output shape.
- Valid image upload.
- Valid prediction.
- Unsupported image format.
- Corrupted image.
- Missing image.
- Oversized image.
- Invalid dimensions.
- Health API.
- Model information API.
- Prediction API.
- Frontend clear functionality.

Do not use fake model predictions to claim production testing.

Mock the model only for isolated unit tests where appropriate, and separately verify real inference using the actual model file.

# 16. PRODUCTION DEPLOYMENT

Prepare the application for Ubuntu Linux and AWS EC2.

Include:

- requirements.txt.
- .env.example.
- .gitignore.
- Gunicorn configuration.
- Nginx configuration.
- Systemd service.
- README.md.
- Installation instructions.
- Virtual environment instructions.
- Production deployment instructions.
- SSL configuration guidance.
- Application restart instructions.
- Logs and troubleshooting instructions.

Never commit the model file or environment secrets to Git without explicitly considering the repository and deployment requirements.

# 17. DEVELOPMENT WORKFLOW

Follow the workflow.

PHASE 1:
Inspect my project directory and existing model.

PHASE 2:
Analyze the model file and preprocessing requirements.

PHASE 3:
Explain the proposed architecture.

PHASE 4:
Create the production-ready folder structure.

PHASE 5:
Implement Flask backend.

PHASE 6:
Implement model integration and image preprocessing.

PHASE 7:
Implement premium UI/UX.

PHASE 8:
Integrate frontend and APIs.

PHASE 9:
Implement security and validation.

PHASE 10:
Run automated tests.

PHASE 11:
Run the application and verify actual predictions.

PHASE 12:
Prepare production deployment documentation.

# 18. STRICT DEVELOPMENT RULES

- Work directly in my selected project directory.
- Create actual files.
- Reuse cifar10_predict.h5.
- Do not train a new model.
- Do not build a CNN.
- Do not replace the model.
- Do not generate fake predictions.
- Do not hardcode prediction results.
- Do not skip core functionality.
- Do not leave TODOs for core functionality.
- Verify all imports.
- Verify dependency compatibility.
- Test actual inference.
- Fix implementation errors.
- Ensure Predict and Clear buttons work.
- Ensure mobile responsiveness.
- Keep code modular.
- Provide documentation.

FIRST inspect my existing project and model file.

Then provide the proposed implementation plan.

After I approve or when the project context clearly permits proceeding, implement the complete application directly in the project directory.

Do not stop at generating code snippets. Build the actual application.