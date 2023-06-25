import os
import csv
import torch
import time
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from collections import Counter
from retrying import retry

# Load .env file
load_dotenv()

# Get OpenAI API key from .env file
api_key = os.getenv('OPENAI_API_KEY')

# Set the OpenAI API key
openai.api_key = api_key

# Set up image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 32  # Adjusted max_length parameter
num_beams = 8    # Adjusted num_beams parameter
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def get_image_description(image_path):
    # Open the image
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    # Generate image features
    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate the description
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    return preds[0]

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def generate_title(description):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Generate a short title for an artwork described as: {description}"},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=10,
            temperature=0.5
        )
        title = response['choices'][0]['message']['content'].strip()
        title = title.replace('"', '')
        title = ' '.join(title.split()[:5])
        return title
    except openai.error.ServiceUnavailableError:
        print("OpenAI API rate limit reached or server error. Retrying...")
        time.sleep(60)
        raise
    
def read_existing_titles(csv_file_path):
    if not os.path.exists(csv_file_path):
        return []
    
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        return [row[0] for row in reader]  # return the list of filenames

def generate_csv():
    # Ask the user for the directory path
    dir_path = input('Please enter the path of your art folder: ')

    # Ask the user for the number of images to process
    num_images = int(input('Please enter the number of images you want to name: '))

    # Define the CSV file path
    csv_file_path = os.path.join(dir_path, 'art_titles.csv')

    # Get the list of existing titles
    existing_titles = read_existing_titles(csv_file_path)

    # Initialize a counter for the titles
    titles_counter = Counter(existing_titles)

    # Create or open the CSV file in write mode
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not existing_titles:
            writer.writerow(["File Name", "Title"])

        # Get the list of images in the directory
        images = [filename for filename in os.listdir(dir_path) if filename.endswith(".jpg") or filename.endswith(".png")]
        images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # Filter out images that have already been processed
        images_to_process = [image for image in images if image not in existing_titles]

        if not images_to_process:
            print("All images have already been named.")
            return

        # Process the images
        for filename in tqdm(images_to_process[:num_images], desc="Processing images"):
            file_path = os.path.join(dir_path, filename)
            description = get_image_description(file_path)
            title = generate_title(description)

            # Check for duplicate titles and generate a new one if a duplicate is found
            while titles_counter[title] > 0:
                title = generate_title(description)

            # Update the counter and write to the CSV
            titles_counter[title] += 1
            writer.writerow([filename, title])

        print("Finished naming images.")

# Run the function
generate_csv()
