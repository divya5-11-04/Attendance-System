# 👁️‍🗨️ Face Recognition Based Attendance System

This is a Python-based face recognition attendance system using OpenCV. It allows you to collect facial data, train a face recognition model using the LBPH algorithm, and automate the attendance logging process using live webcam input.

---

## 📌 Features

- 🧠 Face detection and recognition using OpenCV
- 📷 Real-time dataset collection using webcam
- 🧑‍🎓 Model training with LBPH face recognizer
- ✅ Automatic attendance marking
- 🕒 Time-stamped IN/OUT tracking
- 💾 CSV-based attendance logging
- 🔐 Unique recognition based on ID & Name
- 🧪 Modular design (dataset collection, model training, and attendance capture separated)

---

## 🛠 Tech Stack

- **Python 3.x**
- **OpenCV (opencv-contrib-python)** – for face detection & recognition
- **NumPy** – for matrix operations
- **CSV** – for attendance storage
- **Haar Cascade Classifier** – for face detection

---


```markdown
## 📁 Project Structure

- `dataset/` → Stores collected face images  
- `trainer.yml` → Stores the trained face recognition model  
- `attendance.csv` → Logs attendance records  
- `dataset_collection.py` → Script to collect and store face images  
- `train_model.py` → Script to train the face recognition model  
- `recognize_attendance.py` → Real-time attendance recognition script  
- `README.md` → Documentation of the project  




---

## ⚙️ Installation

1. **Clone the repository**

git clone https://github.com/your-username/face-attendance.git
cd face-attendance
Install dependencies


pip install opencv-contrib-python numpy
Ensure webcam is connected and working.

🚀 Usage
Step 1: Collect Face Dataset
dataset_collection.py
You'll be prompted to enter a name. The webcam will capture 30 images of your face.

Step 2: Train the Recognizer
train_model.py
This trains the LBPH model and saves trainer.yml.

Step 3: Run the Attendance System
recognize_attendance.py
This starts webcam recognition. When a registered face is recognized, attendance is marked with a timestamp in attendance.csv.

📊 Output
attendance.csv will look like:

ID,Name,Date,Time,Status
1,Amit,2025-07-20,10:42:10,IN
1,Amit,2025-07-20,17:15:30,OUT

🛡️ Requirements
Python 3.7+

Webcam

opencv-contrib-python (for cv2.face)

haarcascade_frontalface_default.xml (included)

🚧 Future Improvements
GUI with Tkinter or PyQt

Face registration via form input

Database (SQLite/MySQL) support

Auto email/SMS attendance notification

Mask/no-mask detection

Anti-spoofing with liveness detection

🤝 Contributing
Pull requests are welcome! Feel free to open an issue if you find a bug or want to suggest a feature.

🙏 Acknowledgements
OpenCV community

PyImageSearch

Open Source Haarcascades by Intel


