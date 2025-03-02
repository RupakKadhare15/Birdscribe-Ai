# Birdscribe AI
![logo](assets/final_logo.png)

Birdscribe AI is a Streamlit-based application that allows users to detect birds in images and videos using a YOLO model. It then retrieves additional information about the detected bird species using a large language model (LLM) hosted on Hugging Face.

### Overview ♾️
## Key Features
- AI-Powered Bird Detection – Uses YOLO-based Computer Vision to identify 200+ bird species from images and videos.
-  Conversational Bird Insights – Integrates Mistral-7B LLM to provide species details, habitat information, and conservation status.
-  Real-Time Identification – Quickly detects birds and displays relevant information to aid researchers, birdwatchers, and conservationists.
-  User-Friendly Interface – Developed with Streamlit, ensuring a smooth and accessible experience for all users.
- Community-Driven Conservation – Encourages citizen scientists to contribute data, fostering global engagement in bird conservation.

## Inspiration 🌄
1. **Passion for Wildlife Conservation** – Inspired by the urgent need to protect bird species from habitat loss, climate change, and declining populations.
2. **Bridging AI & Nature** – Leveraging Computer Vision and LLMs to make bird identification and query accessible, fast, and intelligent.
3. **Empowering Bird Enthusiasts & Researchers** – Creating a free, AI-driven tool for birdwatchers, students, and conservationists to identify and learn about birds effortlessly.
4. **Fostering Environmental Awareness** – Encouraging people to explore, appreciate, and protect biodiversity through technology-driven education.
   
BirdScribe AI merges innovation with impact, promoting wildlife conservation and awareness while aligning with UN SDGs 4 (Quality Education), 13 (Climate Action), and 15 (Life on Land).

References -  
1. https://sites.google.com/xtec.cat/sdg-15-life-on-land-birds/home
2. https://www.birds.cornell.edu/home/bring-birds-back/

## Features 

- **Bird Detection**: Uses YOLO object detection to identify birds in images and videos.
- **Bird Information Retrieval**: Fetches details like scientific name, geographical distribution, size, weight, and lifespan from an LLM.
- **User Interaction**: Users can upload images or videos, ask additional questions, and receive AI-generated responses.

## Technologies Used 💻

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

## How to Use 📖

1. Choose a file type (Image or Video).
2. Upload an image (JPG, JPEG, PNG) or a video (MP4, AVI, MPEG4).
3. The app will detect birds and display their names.
4. Additional bird information will be retrieved and displayed.
5. Users can ask further questions via the sidebar chatbot.

## Detection Results 🔍

![Downy Woodpecker](assets/Downy_woodpecker.gif)

![Clark Nutcracker](assets/clark_nutcracker.png)


## Testing 🧪
We strongly suggest checking out the **Test Folder**, which contains various classes of birds, providing images and videos to enhance your testing and UI experience.

## Dataset 📊
The origincal dataset can be accessed using this link:
https://www.vision.caltech.edu/datasets/cub_200_2011/

## Authors 🧑🏻‍💻 

[Priyesh Gawali](https://github.com/Roronoa-17)
[Abhijit Dhande](https://github.com/abhijit-8688)
[Aniket Ambatkar](https://github.com/AniketAmbatkar)
[Rupak Kadhare](https://github.com/RupakKadhare15)
