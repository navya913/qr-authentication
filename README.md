 **QR Code Authentication**  

This project detects and classifies QR codes using a Roboflow-hosted machine learning model



 🚀 How to Use This Project 

Clone the Repository 
Open a terminal and run:  


git clone https://github.com/navya913/QR_Authentication.git
cd qr-authentication


Access the Provided Secret Key 

This project requires a **Roboflow API key** to function. The **secret key has been provided** separately.  

 🔑 Steps to Use the Secret Key: 
1. Locate the provided `secret.txt` file (shared separately with this repository).  
2. Place `secret.txt` inside the project directory (next to `qr_detection.py`).  
3. Do not modify its contents.The script will automatically read the key from this file.  



Install Dependencies
Ensure you have **Python 3.7+** installed, then run:  


pip install -r requirements.txt


Run the Script**  
Execute the detection script:  

python qr_detection.py


The script will:  
✅ Load the QR code image.  
✅ Use the **Roboflow API** to classify it.  
✅ Draw bounding boxes around detected QR codes.  
✅ Save the result as `detection_result.png`.  


📂 Project Structure


QR_Authentication/
│── First Print/                  # Original QR codes
│── Second Print/                 # Counterfeit QR codes
│── qr_detection.py               # Main script
│── secret.txt                    # (Instructor must place this file here)
│── requirements.txt               # Dependencies
│── README.md                      # Documentation
│── detection_result.png           # Output image
│── .gitignore                     # Excludes secret.txt from Git




📌 Important Notes

- **The `secret.txt` file is necessary for this project to run.**  
- The instructor must access the provided `secret.txt` file and place it in the project directory.  
- **There is no need to generate a new API key.**  
- The key is kept secret and is **not uploaded to GitHub** for security.  



👨‍💻 Author 
- Name: Navya Jainayak  
- GitHub: [navya913](https://github.com/navya913)  



