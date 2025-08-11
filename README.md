# Predicting Student At-Risk Status Using Early-Semester Performance Indicators

**Overview:**

This project aims to develop a predictive model for identifying students at risk of academic failure early in the semester.  By analyzing early-semester performance indicators, the model enables proactive interventions to improve student retention and success rates.  The analysis involves data preprocessing, exploratory data analysis (EDA), model training, and evaluation.  The final model provides a probability score indicating the likelihood of a student being at-risk.


**Technologies Used:**

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn


**How to Run:**

1. **Install Dependencies:**  Ensure you have Python 3 installed.  Then, install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

**Example Output:**

The script will print key findings from the exploratory data analysis to the console, including summary statistics and insights into the relationship between different performance indicators and at-risk status.  Additionally, the script will generate several visualization files (e.g., `.png` images) in the `output` directory, illustrating important aspects of the data and model performance.  These visualizations may include histograms of key features, correlation matrices, and model performance metrics.  Finally, the script will output the trained model, which can be used to predict the at-risk status of new students.


**Data:**

The project requires a dataset containing student performance data.  Please ensure you have the necessary data file (e.g., `student_data.csv`) in the project's root directory. The data should include relevant features like assignment scores, attendance, and previous academic performance.  The format of the data file should be specified within the code.

**Further Development:**

This project can be extended by incorporating additional features, exploring different machine learning models, and performing more rigorous model evaluation techniques.  The model could also be integrated into a larger student information system for automated at-risk identification.