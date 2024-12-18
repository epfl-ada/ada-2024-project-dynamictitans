from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os
import requests
import pandas as pd

# Function to download an image from a given URL
def download_image(image_url, save_path):
    """
    Download an image from a given URL and save it to a specified path.
    
    Parameters:
        image_url (str): URL of the image to download.
        save_path (str): Local path to save the downloaded image.
    """
    try:
        response = requests.get(image_url, stream=True)  # Request the image from the URL
        if response.status_code == 200:  # Check if the request was successful
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):  # Save the image in chunks
                    file.write(chunk)
            print(f"Image saved to {save_path}")
        else:
            print(f"Failed to fetch image from {image_url}")
    except Exception as e:
        print(f"Error downloading image: {e}")

# Function to fetch actor images from IMDB using Selenium
def fetch_imdb_actor_images(actors, output_dir="../data/actor_images"):
    """
    Fetch actor profile images from IMDB based on a list of actor names.
    
    Parameters:
        actors (list): List of actor names to search for on IMDB.
        output_dir (str): Directory to save downloaded actor images.
    """
    # Set up the Selenium WebDriver using Chrome
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    # Base IMDB search URL
    base_url = "https://www.imdb.com/find?q="
    
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    for actor in actors:
        # Construct the search URL for the actor
        search_url = base_url + "+".join(actor.split())
        driver.get(search_url)  # Navigate to the search page

        try:
            # Wait for the search results to load and locate the first result element
            actor_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".ipc-metadata-list-summary-item"))
            )

            # Extract the actor's name and link to their IMDB profile
            name_link = actor_element.find_element(By.CSS_SELECTOR, ".ipc-metadata-list-summary-item__t")
            actor_name = name_link.text
            actor_link = name_link.get_attribute("href")
            print(f"Actor: {actor_name}, Link: {actor_link}")

            # Extract the actor's profile image URL
            img_tag = actor_element.find_element(By.CSS_SELECTOR, ".ipc-image")
            # Get the high-resolution image URL
            img_url = img_tag.get_attribute("srcset").split(",")[-1].split(" ")[0]
            if not img_url.startswith("https"):  # Fallback to an alternative URL if necessary
                img_url = img_tag.get_attribute("src")
            print(f"Image URL: {img_url}")

            # Save the image locally with the actor's name as the filename
            save_path = os.path.join(output_dir, f"{actor_name.replace(' ', '_')}.jpg")
            download_image(img_url, save_path)
        except Exception as e:
            print(f"Failed to process actor {actor}: {e}")

    # Close the WebDriver
    driver.quit()
First_20_higher_actor = pd.read_csv('../data/HigherActor.csv')
First_20_lower_actor = pd.read_csv('../data/LowerActor.csv')
fetch_imdb_actor_images(First_20_higher_actor['Actor name'].tolist())
fetch_imdb_actor_images(First_20_lower_actor['Actor name'].tolist())