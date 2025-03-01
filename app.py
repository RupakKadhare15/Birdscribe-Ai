import streamlit as st
import cv2
import requests
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Hugging Face API Key
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

final_birds = None

# Load YOLO Model
model = YOLO('best_v11-60epochs.pt')
bird_name = None
# Styling
st.markdown(
    """
    <style>
        .stApp {
            color: var(--text-color);
        }
        
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #4B9FE1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-bottom: 5px;
        }
        
        .subtext {
            font-size: 16px;
            text-align: center;
            opacity: 0.8;
            margin-bottom: 10px;
        }
        
        .bird-count {
            font-size: 22px;
            font-weight: bold;
            color: #FF7F50;
            text-align: center;
            margin: 20px 0;
        }
        
        .info-box {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .bird-info {
            margin: 8px 0;
            padding: 8px;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            color: inherit;
        }
        
        .info-label {
            font-weight: bold;
            color: #4B9FE1;
            margin-right: 10px;
        }
        
        .bird-name {
            color: #4B9FE1;
            font-size: 1.5em;
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Header
st.markdown('<p class="title">🦜 Birdscribe AI</p><p class="subtext">Detect birds in images and videos using AI-powered vision.</p>', unsafe_allow_html=True)

def extract_clean_info(text):
    """Extract and clean bird information from LLM response."""
    # Define the expected fields
    fields = [
        "Scientific Name",
        "Common Names",
        "Geographical Distribution",
        "Size",
        "Weight",
        "Feet Type",
        "Lifespan"
    ]
    
    # Initialize dictionary to store the information
    info_dict = {}
    
    # Process each line
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        
        # Check if this line contains field information
        for field in fields:
            if field.lower() in line.lower() and ":" in line:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    info_dict[field] = parts[1].strip()
                    break
            # Handle numbered format: "1. Scientific Name: Icterus parisorum"
            elif line.startswith(f"{fields.index(field) + 1}.") and field.lower() in line.lower() and ":" in line:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    info_dict[field] = parts[1].strip()
                    break
    
    # Return formatted string with collected information
    result = []
    for field in fields:
        if field in info_dict and info_dict[field]:
            result.append(f"{field}: {info_dict[field]}")
            
    return '\n'.join(result)

def query_bird_info(bird_name):
    """Query bird information from Mistral API with clear formatting instructions."""
    prompt = f"""Provide detailed information about the bird species '{bird_name}' using the exact format below:

Scientific Name: [scientific name]
Common Names: [common names]
Geographical Distribution: [distribution info]
Size: [size measurements]
Weight: [weight range]
Feet Type: [feet description]
Lifespan: [lifespan info]

Important: Include the labels exactly as shown above, followed by a colon and the information.
"""
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code}")
            return None
            
        result = response.json()[0]["generated_text"]
        return extract_clean_info(result)
    except Exception as e:
        st.error(f"Error querying bird information: {str(e)}")
        return None

def format_bird_info(info_text):
    """Format bird information for HTML display."""
    if not info_text:
        return "<div class='bird-info'>Information not available</div>"
        
    formatted_html = ""
    for line in info_text.split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                label, value = parts
                formatted_html += f"<div class='bird-info'><span class='info-label'>{label.strip()}</span>{value.strip()}</div>"
    
    return formatted_html

# File upload section
upload_type = st.selectbox(
    "Choose file type",
    ["Image", "Video"],
    key="file_type"
)

# Set allowed file types
if upload_type == "Image":
    allowed_types = ["jpg", "jpeg", "png"]
    upload_message = "Upload an image (JPG, JPEG, PNG)"
else:
    allowed_types = ["mp4", "avi", "mpeg4"]
    upload_message = "Upload a video (MP4, AVI, MPEG4)"

uploaded_file = st.file_uploader(upload_message, type=allowed_types)

if uploaded_file is not None:
    bird_info = {}

    if upload_type == "Image":
        # Process image
        image = Image.open(uploaded_file)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert PIL to Numpy array for YOLO
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[-1] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # Run detection
        results = model(image_np)
        annotated_frame = results[0].plot()

        # Get detected bird names
        detected_names = []
        if results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls)
                if class_id in model.names:
                    detected_names.append(model.names[class_id])
        
        if not detected_names:
            detected_names = ["No birds detected"]

        # Display annotated image
        st.image(annotated_frame, caption="Detected Objects", use_container_width=True)

        # Display detected birds count
        st.markdown(f'<p class="bird-count">Detected Bird(s): {", ".join(detected_names)}</p>', unsafe_allow_html=True)

        # Get and display bird information
        if "No birds detected" not in detected_names:
            with st.spinner("Retrieving bird information..."):
                for bird_name in detected_names:
                    if bird_name not in bird_info:
                        info = query_bird_info(bird_name)
                        if info:
                            st.markdown(
                                f'<div class="info-box">'
                                f'<h3 class="bird-name">{bird_name}</h3>'
                                f'{format_bird_info(info)}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning(f"Could not retrieve information for {bird_name}")

    else:
        # Process video
        with st.spinner("Processing video..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            detected_birds = set()
            bird_info_dict = {}

            # Display detected birds list at the top (dynamic updating)
            birds_placeholder = st.empty()

            # Video processing progress bar
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Update progress
                current_frame += 1
                progress_value = min(current_frame / total_frames, 1.0)
                progress_bar.progress(progress_value)

                # Only process every nth frame for efficiency
                if current_frame % 10 == 0:  # Process every 10th frame
                    results = model(frame)
                    annotated_frame = results[0].plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Display the processed frame
                    stframe.image(annotated_frame, caption="Processing Video...", use_container_width=True)

                    # Detect new birds in this frame
                    new_birds = set()
                    if results[0].boxes:
                        for box in results[0].boxes:
                            class_id = int(box.cls)
                            if class_id in model.names:
                                bird_name = model.names[class_id]
                                if bird_name not in detected_birds:
                                    detected_birds.add(bird_name)
                                    new_birds.add(bird_name)  # Track newly detected birds

                    # Update bird list dynamically
                    birds_placeholder.markdown(f'<p class="bird-count">Detected Bird(s): {", ".join(detected_birds)}</p>',
                                            unsafe_allow_html=True)

                    # Fetch & display info for newly detected birds immediately
                    for bird_name in new_birds:
                        info = query_bird_info(bird_name)
                        bird_info_dict[bird_name] = info

                        if info:
                            st.markdown(
                                f'<div class="info-box">'
                                f'<h3 class="bird-name">{bird_name}</h3>'
                                f'{format_bird_info(info)}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning(f"Could not retrieve information for {bird_name}")

            cap.release()
            progress_bar.empty()


class_type = st.selectbox(
    "If you want to know more about a specific bird, select the name of bird:",
    [None,'Black footed Albatros',
 'Laysan Albatros',
 'Sooty Albatros',
 'Groove billed Ani',
 'Crested Aukle',
 'Least Aukle',
 'Parakeet Aukle',
 'Rhinoceros Aukle',
 'Brewer Blackbird',
 'Red winged Blackbird',
 'Rusty Blackbird',
 'Yellow headed Blackbird',
 'Bobolink',
 'Indigo Bunting',
 'Lazuli Bunting',
 'Painted Bunting',
 'Cardinal',
 'Spotted Catbird',
 'Gray Catbird',
 'Yellow breasted Chat',
 'Eastern Towhee',
 'Chuck will Widow',
 'Brandt Cormorant',
 'Red faced Cormorant',
 'Pelagic Cormorant',
 'Bronzed Cowbird',
 'Shiny Cowbird',
 'Brown Creeper',
 'American Crow',
 'Fish Crow',
 'Black billed Cuckoo',
 'Mangrove Cuckoo',
 'Yellow billed Cuckoo',
 'Gray crowned Rosy Finch',
 'Purple Finch',
 'Northern Flicker',
 'Acadian Flycatcher',
 'Great Crested Flycatcher',
 'Least Flycatcher',
 'Olive sided Flycatcher',
 'Scissor tailed Flycatcher',
 'Vermilion Flycatcher',
 'Yellow bellied Flycatcher',
 'Frigatebird',
 'Northern Fulmar',
 'Gadwall',
 'American Goldfinch',
 'European Goldfinch',
 'Boat tailed Grackle',
 'Eared Grebe',
 'Horned Grebe',
 'Pied billed Grebe',
 'Western Grebe',
 'Blue Grosbeak',
 'Evening Grosbeak',
 'Pine Grosbeak',
 'Rose breasted Grosbeak',
 'Pigeon Guillemot',
 'California Gull',
 'Glaucous winged Gull',
 'Heermann Gull',
 'Herring Gull',
 'Ivory Gull',
 'Ring billed Gull',
 'Slaty backed Gull',
 'Western Gull',
 'Anna Hummingbird',
 'Ruby throated Hummingbird',
 'Rufous Hummingbird',
 'Green Violetear',
 'Long tailed Jaeger',
 'Pomarine Jaeger',
 'Blue Jay',
 'Florida Jay',
 'Green Jay',
 'Dark eyed Junco',
 'Tropical Kingbird',
 'Gray Kingbird',
 'Belted Kingfisher',
 'Green Kingfisher',
 'Pied Kingfisher',
 'Ringed Kingfisher',
 'White breasted Kingfisher',
 'Red legged Kittiwake',
 'Horned Lark',
 'Pacific Loon',
 'Mallard',
 'Western Meadowlark',
 'Hooded Merganser',
 'Red breasted Merganser',
 'Mockingbird',
 'Nighthawk',
 'Clark Nutcracker',
 'White breasted Nuthatch',
 'Baltimore Oriole',
 'Hooded Oriole',
 'Orchard Oriole',
 'Scott Oriole',
 'Ovenbird',
 'Brown Pelican',
 'White Pelican',
 'Western Wood Pewee',
 'Sayornis',
 'American Pipit',
 'Whip poor Will',
 'Horned Puffin',
 'Common Raven',
 'White necked Raven',
 'American Redstart',
 'Geococcyx',
 'Loggerhead Shrike',
 'Great Grey Shrike',
 'Baird Sparrow',
 'Black throated Sparrow',
 'Brewer Sparrow',
 'Chipping Sparrow',
 'Clay colored Sparrow',
 'House Sparrow',
 'Field Sparrow',
 'Fox Sparrow',
 'Grasshopper Sparrow',
 'Harris Sparrow',
 'Henslow Sparrow',
 'Le Conte Sparrow',
 'Lincoln Sparrow',
 'Nelson Sharp tailed Sparrow',
 'Savannah Sparrow',
 'Seaside Sparrow',
 'Song Sparrow',
 'Tree Sparrow',
 'Vesper Sparrow',
 'White crowned Sparrow',
 'White throated Sparrow',
 'Cape Glossy Starling',
 'Bank Swallow',
 'Barn Swallow',
 'Cliff Swallow',
 'Tree Swallow',
 'Scarlet Tanager',
 'Summer Tanager',
 'Artic Tern',
 'Black Tern',
 'Caspian Tern',
 'Common Tern',
 'Elegant Tern',
 'Forsters Tern',
 'Least Tern',
 'Green tailed Towhee',
 'Brown Thrasher',
 'Sage Thrasher',
 'Black capped Vireo',
 'Blue headed Vireo',
 'Philadelphia Vireo',
 'Red eyed Vireo',
 'Warbling Vireo',
 'White eyed Vireo',
 'Yellow throated Vireo',
 'Bay breasted Warbler',
 'Black and white Warbler',
 'Black throated Blue Warbler',
 'Blue winged Warbler',
 'Canada Warbler',
 'Cape May Warbler',
 'Cerulean Warbler',
 'Chestnut sided Warbler',
 'Golden winged Warbler',
 'Hooded Warbler',
 'Kentucky Warbler',
 'Magnolia Warbler',
 'Mourning Warbler',
 'Myrtle Warbler',
 'Nashville Warbler',
 'Orange crowned Warbler',
 'Palm Warbler',
 'Pine Warbler',
 'Prairie Warbler',
 'Prothonotary Warbler',
 'Swainson Warbler',
 'Tennessee Warbler',
 'Wilson Warbler',
 'Worm eating Warbler',
 'Yellow Warbler',
 'Northern Waterthrush',
 'Louisiana Waterthrush',
 'Bohemian Waxwing',
 'Cedar Waxwing',
 'American Three toed Woodpecker',
 'Pileated Woodpecker',
 'Red bellied Woodpecker',
 'Red cockaded Woodpecker',
 'Red headed Woodpecker',
 'Downy Woodpecker',
 'Bewick Wren',
 'Cactus Wren',
 'Carolina Wren',
 'House Wren',
 'Marsh Wren',
 'Rock Wren',
 'Winter Wren',
 'Common Yell'],
    key="Class Type"
)

def query_huggingface(prompt, bird_name):
    system_prompt = (
        f"You will only answer questions related to the birds listed below. "
        f"If the question is about a bird not in the list, respond with 'I can only provide information about the detected birds.'\n\n"
        f"List of detected birds: {bird_name}\n\n"
        f"Now, answer the following question based only on this list: {prompt}"
    )
    payload = {"inputs": system_prompt, "parameters": {"temperature": 0.8, "max_length": 100}}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        generated_text = response.json()[0]["generated_text"]
        return generated_text.replace(system_prompt, "").strip()
    else:
        return "Error: Unable to fetch response."
    
# Sidebar
st.sidebar.title("Have any question? Ask me!")
user_input = st.sidebar.text_input("You:", key="user_input")
if st.sidebar.button("Send") and user_input:
    if bird_name or class_type is not None:
        response = query_huggingface(user_input, class_type or bird_name)
        st.sidebar.write("**Bot:**", response)
    else:
        print("Please select a class ")

st.markdown("""
            ### Help:
            - Upload an image or videos for detecting birds. After detection, information will be generated.
            - Have further questions, select the bird name from drop down menu and ask the bot from sidebar.
            """)