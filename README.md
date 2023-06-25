Artwork Naming Script
This Python script, named namegen.py, helps you to automatically generate names for your artwork based on their visual content using the OpenAI GPT-3 model.

Setup
Install Python
Before you can run this script, you need to have Python installed on your system. If you don't have Python installed, you can download it from here. We recommend installing Python 3.8 or later.

After installing Python, verify the installation by opening a terminal and running:

css
Copy code
python --version
You should see Python's version number in the response.

Install Dependencies
This script requires several Python libraries. You can install these dependencies by running the following command in the terminal:

Copy code
pip install torch transformers tqdm pillow python-dotenv openai
If you're using a Python version that's older than 3.4, you might need to use pip3 instead of pip.

Obtain OpenAI API Key
This script requires an OpenAI API key to generate the artwork titles. If you don't have an OpenAI API key, you can obtain one by signing up on the OpenAI website.

Once you have your API key, create a file named .env in the same directory as the script and add your API key to it like this:

makefile
Copy code
OPENAI_API_KEY=your-api-key-here
Replace your-api-key-here with your actual OpenAI API key.

Running the Script
In the terminal, navigate to the directory containing the script and run:

Copy code
python namegen.py
The script will ask for the path to your art folder and the number of images you want to process. The image names will be generated and stored in a CSV file in the same directory as your images. The file will be named art_titles.csv.

If any images have already been named, the script will skip them and continue to the next image. The order of the images in the CSV file will be sorted in ascending numerical order based on the image file names.

For any further queries or assistance, feel free to reach out. Happy naming!
