# Birdscribe AI

Birdscribe AI is a Streamlit-based application that allows users to detect birds in images and videos using a YOLO model. It then retrieves additional information about the detected bird species using a large language model (LLM) hosted on Hugging Face.

## Features

- **Bird Detection**: Uses YOLO object detection to identify birds in images and videos.
- **Bird Information Retrieval**: Fetches details like scientific name, geographical distribution, size, weight, and lifespan from an LLM.
- **User Interaction**: Users can upload images or videos, ask additional questions, and receive AI-generated responses.
- **Custom Styling**: Aesthetic UI elements for a better user experience.

## Technologies Used

- **Streamlit**: For building the web application.
- **YOLO (You Only Look Once)**: For bird detection in images and videos.
- **Hugging Face Inference API**: For retrieving bird-related information using the Mistral-7B-Instruct-v0.2 model.
- **OpenCV**: For image and video processing.
- **PIL (Pillow)**: For handling image files.
- **NumPy**: For efficient array operations.

## Installation

To run this project locally, follow these steps:

### Prerequisites

Ensure you have Python installed (preferably Python 3.8 or later).

### Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/birdscribe-ai.git
   cd birdscribe-ai
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## How to Use

1. Choose a file type (Image or Video).
2. Upload an image (JPG, JPEG, PNG) or a video (MP4, AVI, MPEG4).
3. The app will detect birds and display their names.
4. Additional bird information will be retrieved and displayed.
5. Users can ask further questions via the sidebar chatbot.

## API Usage

The application interacts with the Hugging Face API using the following method:

- **Bird Information Retrieval:**
  ```python
  response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
  ```
- **Querying the LLM for Additional Information:**
  ```python
  payload = {"inputs": system_prompt, "parameters": {"temperature": 0.8, "max_length": 100}}
  response = requests.post(API_URL, headers=headers, json=payload)
  ```

## Acknowledgments

- **Ultralytics YOLO** for object detection.
- **Hugging Face** for hosting the Mistral LLM.
- **Streamlit** for providing an interactive web interface.

## License

This project is licensed under the MIT License.

## Author

Your Name - [Your GitHub Profile](https://github.com/your-github-profile)

