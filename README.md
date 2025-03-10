# Birdscribe AI
![logo](assets/final_logo.png)

Detect, Query, and Learn: AI-powered bird identification with intelligent insights.

## Overview ♾️
BirdScribe AI is a Streamlit-powered app that allows users to effortlessly detect birds in images and videos. It provides detailed information about each species, all at your fingertips, powered by a large language model (LLM) 

### Inspiration 🌄
- **Passion for Wildlife Conservation** – Inspired by the urgent need to protect bird species from habitat loss, climate change, and declining populations.
- **Bridging AI & Nature** – Leveraging Computer Vision and LLMs to make bird identification and query accessible, fast, and intelligent.
- **Empowering Bird Enthusiasts & Researchers** – Creating a free, AI-driven tool for birdwatchers, students, and conservationists to identify and learn about birds effortlessly.
- **Fostering Environmental Awareness** – Encouraging people to explore, appreciate, and protect biodiversity through technology-driven education.
   
BirdScribe AI merges innovation with impact, promoting wildlife conservation and awareness while aligning with UN SDGs 4 (Quality Education), 13 (Climate Action), and 15 (Life on Land).

References -  
1. https://sites.google.com/xtec.cat/sdg-15-life-on-land-birds/home
2. https://www.birds.cornell.edu/home/bring-birds-back/

### Features 

- **Bird Detection** – Uses YOLO-based Computer Vision to identify 200+ bird species from images and videos.
- **Conversational Bird Insights** – Integrates Mistral-7B LLM to provide species specific details like scientific name, geographical distribution, size, weight, and lifespan.
- **Real-Time Identification** – Quickly detects birds and displays relevant information to aid researchers, birdwatchers and conservationists.
- **User-Friendly Interface** – Developed with Streamlit, Users can upload images or videos, ask additional questions, and receive AI-generated responses, ensuring a smooth and accessible experience for all users.

### Technologies Used 💻

Python, YOLO, Streamlit, OpenCV, Requests, Pillow, NumPy, Hugging Face API, Huggingspace

### Installation 

To run this project locally, follow these steps:

#### Prerequisites

Ensure you have Python installed (preferably Python 3.8 or later).

#### Setup

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
