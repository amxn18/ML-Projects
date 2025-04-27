 🧠 Parkinson's Disease Prediction (SVM Classifier)
 This Bash-style README explains how to set up and run a machine learning model to predict Parkinson's disease using a dataset of voice measurements.
 
 📁 Step 1: Place Your Files Properly
 Make sure the following files are in the same directory:
  - parkinson.csv         # Dataset file
  - disease.py            # Python script with model code

# ---------------------------------------
 ⚙️ Step 2: Create a Virtual Environment (Optional but Recommended)
# ---------------------------------------
 python3 -m venv parkinsons_env
 source parkinsons_env/bin/activate   # For Linux/macOS
 parkinsons_env\Scripts\activate      # For Windows

# ---------------------------------------
 📦 Step 3: Install Required Python Packages
# ---------------------------------------
 Run the following command to install dependencies:

pip install pandas numpy scikit-learn

# ---------------------------------------
 🚀 Step 4: Run the Prediction Script
 ---------------------------------------
 This script:
  - Reads and preprocesses the data
  - Splits into training and test sets
  - Trains an SVM model
  - Evaluates accuracy
  - Predicts Parkinson's using example input

python disease.py

 You should see output like:
 Training Accuracy Score: 0.89
 Testing Accuracy Score: 0.87
 The person has Parkinson's Disease.

 ---------------------------------------
 🧪 Step 5: Try with Your Own Input Data
 ---------------------------------------
 Open parkinson.csv and pick a row (exclude 'name' and 'status' columns)
 Example row from CSV:
 phon_R01_S01_1,119.99200,157.30200,...,0.123306,1

 Use this row in Python like:
 input_data = (119.99200, 157.30200, ..., 0.123306)

 Replace the example input_data tuple in disease.py with your own

# ---------------------------------------
 📈 Step 6: Understand the Dataset
 ---------------------------------------
 Columns:
 - Total of 22 biomedical voice features like:
   MDVP:Fo(Hz), Jitter(%), Shimmer(dB), etc.
 Target:
 - status: 1 (Parkinson's), 0 (Healthy)

# ---------------------------------------
 🧼 Optional Cleanup
# ---------------------------------------
 If you used a virtual environment and want to exit:
 deactivate

 ---------------------------------------
 📝 Tips
 ---------------------------------------
 - You can wrap this into a Streamlit or Flask app later
 - Try saving the model using joblib or pickle
 - You can improve accuracy by hyperparameter tuning (e.g. C, gamma in SVM)

# ---------------------------------------
 💡 Summary
# ---------------------------------------
 ✅ SVM Classifier
 ✅ Standardized Input
 ✅ High Accuracy
 ✅ Simple and Fast to Run

 Done! You’re ready to predict Parkinson’s Disease using ML! 💥
