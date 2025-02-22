# CNN-Based-Waste-Classification-Using-BDWaste-Dataset
📌 Project Overview

This project aims to develop a Convolutional Neural Network (CNN) model for classifying waste into biodegradable and non-biodegradable categories using the BDWaste dataset. The dataset consists of 2,625 real-world waste images categorized into 21 distinct classes. By training and evaluating multiple CNN architectures, this project identifies the most effective model for waste classification and optimizes its performance using hyperparameter tuning.

🎯 Research Question

How accurately can different CNN architectures classify biodegradable and non-biodegradable waste, and does hyperparameter tuning significantly improve their performance?

🚀 Project Objectives
	•	Preprocess the BDWaste dataset and apply data augmentation techniques.
	•	Train and compare multiple CNN models:
	•	Basic CNN
	•	VGG16
	•	ResNet50
	•	EfficientNet
	•	MobileNetV2
	•	Optimize the best-performing model using hyperparameter tuning.
	•	Evaluate model performance using accuracy, precision, recall, and F1-score.
	•	Store and manage project code and data securely using GitHub for version control.

📂 Dataset Overview
	•	Dataset Source: Mendeley Data Repository
	•	Total Images: 2,625
	•	Number of Classes: 21
	•	Image Format: JPEG/PNG
	•	Dataset Size: ~6.65GB
	•	Data Collection Method: Captured using an Android device in indoor and outdoor settings.

📊 Model Training & Evaluation

The project compares various CNN architectures to determine the most effective model for waste classification. Key steps include:
	1.	Data Preprocessing & Augmentation (Resizing, Normalization, Augmentation)
	2.	Training Different CNN Architectures
	3.	Hyperparameter Tuning (Optimizing Learning Rate, Batch Size, Dropout, etc.)
	4.	Model Evaluation using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-score

⚙️ Implementation Details

🏗 Dependencies

Before running the project, ensure you have the following dependencies installed:

pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python scikit-learn

🚀 Running the Model

	1.	Clone the repository:

git clone https://github.com/AVINASH-ANGILIKAM/CNN-Based-Waste-Classification-Using-BDWaste-Dataset.git
cd CNN-Based-Waste-Classification


	2.	Download the dataset from Mendeley Data and place it in the dataset/ directory.
	3.	Preprocess the data:

python dataset_preprocessing.py


	4.	Train the model:

python model_training.py


	5.	Evaluate the model:

python model_evaluation.py



📊 Results

The project provides comparative analysis of different CNN architectures. The final model selection is based on:
	•	Accuracy: How often the model correctly classifies waste items.
	•	Precision & Recall: The trade-off between false positives and false negatives.
	•	F1-score: A balanced measure combining precision and recall.

🔐 Data Management & Ethics
	•	GitHub Repository: Project Link
	•	Version Control: Weekly commits with clear documentation.
	•	Storage & Security:
	•	Code stored in GitHub.
	•	Dataset stored in Google Drive and OneDrive.
	•	Ethical Considerations:
	•	Dataset does not contain personal or sensitive information.
	•	Complies with university ethical research policies.
	•	Publicly available dataset for academic and research purposes.

