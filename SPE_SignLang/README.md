# Sign Language Recognition

A web application that recognizes sign language gestures using deep learning. The application uses a convolutional neural network (CNN) to classify hand gestures representing letters A-Z.

## Features

- Real-time sign language recognition
- User-friendly web interface
- Support for image upload
- Drag and drop functionality
- Responsive design

## Tech Stack

- Backend: FastAPI, TensorFlow, Python
- Frontend: HTML, CSS, JavaScript
- Deep Learning: CNN for image classification

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SIGN_LANG.git
cd SIGN_LANG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
uvicorn signlang:app --reload
```

4. Start the frontend server:
```bash
python -m http.server 3000
```

5. Open your browser and navigate to:
```
http://localhost:3000
```

## Usage

1. Click the upload area or drag and drop an image
2. The system will process the image and display the predicted letter
3. Results are shown in real-time

## Project Structure

```
SIGN_LANG/
├── signlang.py          # FastAPI backend
├── index.html           # Frontend interface
├── model_weights.h5     # Trained model weights
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
