# ROLE

Act as a Senior Python Flask Developer, Machine Learning Engineer, Software Architect, Product Designer, Frontend Developer, QA Engineer, and DevOps Engineer.

I want you to build a complete production-ready Home Price Prediction web application using Flask and my existing trained model.

The application must allow users to enter property-related feature values through a form, submit those values, and receive a predicted home price from my existing model.

# 1. PROJECT OBJECTIVE

Build a modern, premium, form-based Home Price Prediction application.

Users must be able to:

1. Enter the required property features.
2. Validate input values.
3. Submit the form.
4. Predict the home price using my existing trained model.
5. Display the predicted price prominently.
6. Display a formatted monetary value.
7. Display a prediction result visualization.
8. Clear all form values.
9. Clear previous prediction results.
10. Enter new property details and predict again.

The application must have a completely different UI/UX from my CIFAR-10 Image Classification application.

Do not reuse the dark navy AI dashboard design.

Create a premium real-estate valuation interface with a light theme.

# 2. EXISTING MODEL

My existing trained model file is:

boston_home_price_prediction.h5

IMPORTANT:

- Use my existing trained model.
- Do not train a new model.
- Do not create a new ML model.
- Do not replace my existing model.
- Do not generate fake predictions.
- Do not hardcode predicted prices.
- Do not assume the model input features without inspecting the model.
- Do not modify the model file.

Before implementing:

1. Inspect the model file.
2. Identify its framework and loading requirements.
3. Inspect its input shape.
4. Inspect its output shape.
5. Identify the input feature count.
6. Determine whether it expects a single feature vector or another input structure.
7. Determine its input dtype.
8. Identify the preprocessing applied during training if available.
9. Identify whether normalization, scaling, standardization, encoding, or transformation was used.
10. Determine the output scale and interpretation.

If the model file does not contain preprocessing information, inspect available project files and documentation.

If preprocessing cannot be determined, clearly explain the limitation and request the relevant training information instead of inventing preprocessing.

# 3. BOSTON HOUSING FEATURE SCHEMA

The original Boston Housing dataset is commonly associated with these 13 features:

CRIM
ZN
INDUS
CHAS
NOX
RM
AGE
DIS
RAD
TAX
PTRATIO
B
LSTAT

However, these features must not automatically be assumed to be the exact model inputs.

Inspect the trained model and use the exact feature order and preprocessing expected by the model.

If the model expects the original 13 features, use the corresponding input form.

If the model expects fewer, more, transformed, or differently ordered features, implement the actual expected schema and document it.

Create a central feature configuration that contains:

- Feature name.
- Feature label.
- Description.
- Data type.
- Default value.
- Minimum value.
- Maximum value.
- Step.
- Unit.
- Validation rule.
- Input control type.
- Display grouping.

Do not duplicate feature definitions throughout the code.

# 4. ETHICAL AND DATASET LIMITATIONS

The original Boston Housing dataset has documented ethical and methodological limitations.

It includes historically problematic variables such as B and LSTAT.

The application must be treated as an educational machine learning demonstration.

Do not present the prediction as:

- A real estate appraisal.
- A lending decision.
- A reliable market valuation.
- A prediction of an actual property's current market price.

Add a clear disclaimer in the application explaining that:

- The model is educational.
- The dataset is historical.
- Predictions are not actual property valuations.
- The dataset may contain historical bias.
- Predictions should not be used for financial, housing, lending, or investment decisions.

Do not silently remove, modify, or replace model inputs without explaining the impact on model compatibility.

# 5. TECHNOLOGY STACK

Backend:

- Python 3.12
- Flask
- TensorFlow/Keras, if compatible with the existing model
- NumPy
- Python-dotenv

Frontend:

- HTML5
- CSS3
- JavaScript
- Bootstrap 5 or custom CSS

Visualization:

- Chart.js, where appropriate

Production:

- Gunicorn
- Nginx
- Linux
- Environment-based configuration
- Python logging

# 6. PREMIUM UI/UX DESIGN

Create a premium real-estate analytics interface.

Do not reuse the dark UI from my CIFAR-10 project.

Design direction:

- Light theme.
- Premium property analytics.
- Editorial-style interface.
- Minimal.
- Modern.
- Elegant.
- Responsive.
- Clean layout.
- Professional typography.
- Soft shadows.
- Rounded cards.
- Spacious layout.
- Subtle animations.

Color palette:

Background: #F7F5F0
Surface: #FFFFFF
Primary text: #1C2523
Secondary text: #738078
Primary accent: #176B45
Secondary accent: #A8C3B0
Accent highlight: #D6AD65
Border: #E6E8E2
Input background: #FAFAF7

Typography:

Use Inter or Manrope.

Use a premium real-estate dashboard aesthetic.

Avoid:

- Dark navy AI dashboards.
- Neon colors.
- Excessive gradients.
- Overuse of glassmorphism.
- Excessive animations.
- Cluttered interfaces.

# 7. APPLICATION LAYOUT

## A. Header

Create a modern header containing:

- Application logo.
- Home Value Estimator.
- Property Intelligence subtitle.
- Model status indicator.
- Responsive navigation.

## B. Hero Section

Heading:

"Understand Your Property's Estimated Value"

Description:

"Enter the property characteristics to generate an estimated value using a machine learning model."

Add a subtle educational ML badge.

## C. Property Input Form

Create a structured, premium form.

Use grouped sections.

Potential groups if the model expects the original Boston features:

### Property & Neighborhood

CRIM
ZN
INDUS
CHAS

### Environmental & Structural

NOX
RM
AGE
DIS

### Accessibility & Tax

RAD
TAX
PTRATIO

### Historical Dataset Variables

B
LSTAT

These groupings are UI organization only. Do not change the model feature order when assembling the prediction input.

Use a responsive layout.

Desktop:

- Two-column or three-column input layout where appropriate.

Mobile:

- Single-column layout.

Input controls must be clearly labelled.

Use number inputs with appropriate step values.

For CHAS, use a toggle or select:

- No river adjacency: 0
- River adjacency: 1

If the model expects a different schema, create form groups according to the actual model input requirements.

## D. Form Controls

For each feature:

- Display a human-readable label.
- Display a short description.
- Display its unit where relevant.
- Use a suitable numeric input.
- Apply minimum and maximum validation where justified.
- Use an appropriate step value.
- Display validation messages.
- Avoid unrealistic defaults.
- Avoid arbitrary values where they may create misleading predictions.

Use default values only when they are justified by the model training data or clearly identified as demonstration defaults.

Do not silently use default values for missing user inputs.

## E. Predict Button

Create a premium primary button:

"Estimate Home Value"

Behavior:

- Disabled while invalid.
- Display loading state.
- Prevent duplicate submissions.
- Validate all inputs.
- Submit the form to the prediction API.
- Display meaningful error messages.
- Display the result after successful prediction.

## F. Clear Button

Create a secondary button:

"Clear Form"

Behavior:

- Clear all input fields.
- Clear previous prediction.
- Clear result visualization.
- Reset validation errors.
- Reset loading state.
- Reset prediction result.
- Return the application to the initial state.

The Clear button must work without reloading the page.

## G. Prediction Result Section

Create a premium result panel.

Display:

- Predicted home price.
- Currency formatting.
- Model output value.
- Prediction execution time.
- Summary of submitted feature values.
- A clean result visualization.

Do not invent a currency or financial interpretation that the training dataset does not support.

Inspect the model's output scale and document how it is displayed.

If the original Boston target is represented in units of $1,000, only use that interpretation if verified from the model's training context.

If the output scale is uncertain, display the raw model output and clearly explain the uncertainty.

## H. Result Visualization

Create a useful visualization using Chart.js or CSS.

Possible visualization:

- Predicted value card.
- Horizontal value indicator.
- Feature summary.
- Model result summary.

Do not invent market comparisons, property price ranges, historical trends, neighborhood comparisons, or appraisal estimates unless the data is actually available.

## I. Input Summary

After prediction, display the submitted input values in a compact summary.

Allow the user to review the inputs.

Do not store personal or property information unnecessarily.

## J. Footer

Display:

"Home Value Estimator | Machine Learning Demonstration"

Add the dataset limitation disclaimer.

# 8. MODEL INTEGRATION

Create a dedicated model-loading service.

Requirements:

- Load the existing model once per application process.
- Do not load the model on every request.
- Validate model input shape.
- Validate model output shape.
- Validate feature count.
- Validate feature ordering.
- Handle prediction exceptions.
- Handle model-loading errors.
- Keep prediction logic reusable.

Do not train or retrain the model.

Do not alter the model weights.

Do not hardcode predicted results.

# 9. INPUT PREPROCESSING

Preprocessing must match the model's training pipeline.

Inspect:

- Feature order.
- Data types.
- Scaling.
- Standardization.
- Normalization.
- Transformation.
- Missing-value handling.
- Input tensor dimensions.

If the model was trained with a scaler such as StandardScaler or MinMaxScaler, use the original scaler if available.

If the scaler is not available and the model expects scaled input, do not invent scaling parameters.

Explain the missing preprocessing information.

Never independently scale features using arbitrary values.

Never change the input feature order.

Create a dedicated preprocessing service.

# 10. BACKEND API

Implement:

POST /api/predict/

Accept JSON input containing the required feature values.

Validate:

- Missing values.
- Invalid data types.
- Invalid numerical values.
- Out-of-range values where justified.
- Model input shape.
- Model availability.

Return:

- Predicted value.
- Formatted result where appropriate.
- Execution time.
- Submitted feature summary.

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
- Feature count.
- Feature names where verified.
- Framework information where available.

Use consistent JSON response formats.

# 11. API RESPONSE FORMAT

Example:

{
  "success": true,
  "message": "Prediction completed successfully",
  "data": {
    "predicted_value": 245.67,
    "display_value": "245.67",
    "execution_time_ms": 32,
    "feature_summary": {
      "RM": 6.5,
      "NOX": 0.5
    }
  }
}

The example is for response structure only.

Do not hardcode predicted values.

Do not invent currency interpretation.

# 12. FOLDER STRUCTURE

Create a production-ready folder structure.

Separate:

- Flask application.
- Configuration.
- Model loading.
- Prediction service.
- Preprocessing.
- API routes.
- Feature schema.
- Templates.
- CSS.
- JavaScript.
- Tests.
- Deployment.
- Documentation.

Keep boston_home_price_prediction.h5 inside a dedicated models directory.

Do not put all application logic inside one app.py file.

# 13. SECURITY

Implement:

- Input validation.
- Safe JSON parsing.
- Appropriate request size limits.
- Security headers.
- Environment-based configuration.
- Safe exception handling.
- Logging.
- Production DEBUG=False.
- No hardcoded secrets.
- Appropriate API error responses.

Do not expose filesystem paths or internal exception details.

# 14. PERFORMANCE

- Load model once.
- Avoid unnecessary model initialization.
- Efficient prediction.
- Handle requests safely.
- Configure Gunicorn workers appropriately.
- Consider model memory consumption.
- Avoid unnecessary database or storage dependencies.

# 15. AUTOMATED TESTING

Use pytest.

Test:

- Flask application initialization.
- Model loading.
- Model input shape.
- Valid prediction.
- Invalid inputs.
- Missing inputs.
- Wrong input types.
- Out-of-range values.
- API health.
- Model information.
- Prediction API.
- Clear form functionality.
- Result UI behavior.

Use mock model predictions only in isolated unit tests.

Separately verify actual inference using the real model file.

Do not claim actual prediction tests passed if they were not executed.

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
- Restart instructions.
- Logs.
- Troubleshooting.

Never commit secrets.

# 17. DEVELOPMENT WORKFLOW

PHASE 1:
Inspect project directory and existing model.

PHASE 2:
Inspect model architecture, input shape, output shape and preprocessing requirements.

PHASE 3:
Explain the proposed architecture and actual feature schema.

PHASE 4:
Create the production folder structure.

PHASE 5:
Implement Flask backend.

PHASE 6:
Implement model integration and preprocessing.

PHASE 7:
Implement the premium real-estate UI.

PHASE 8:
Implement the form and API integration.

PHASE 9:
Implement validation and security.

PHASE 10:
Implement Clear Form functionality.

PHASE 11:
Run automated tests.

PHASE 12:
Run actual model inference.

PHASE 13:
Prepare production deployment documentation.

# 18. STRICT DEVELOPMENT RULES

- Work directly in my selected project directory.
- Create actual files.
- Reuse boston_home_price_prediction.h5.
- Do not train a new model.
- Do not replace the model.
- Do not generate fake predictions.
- Do not hardcode predicted prices.
- Do not assume the feature schema.
- Do not assume the output scale.
- Do not invent preprocessing.
- Do not silently change feature order.
- Do not leave TODOs for core functionality.
- Keep code modular.
- Verify imports.
- Verify dependency compatibility.
- Run tests.
- Run actual inference where possible.
- Ensure the form works.
- Ensure the Clear Form button works.
- Ensure responsive UI.
- Provide complete deployment documentation.

FIRST inspect my existing project and trained model file.

Then explain the implementation plan and model compatibility.

Proceed with implementation directly in my project directory.

Do not stop at code snippets. Build the actual application and test it.