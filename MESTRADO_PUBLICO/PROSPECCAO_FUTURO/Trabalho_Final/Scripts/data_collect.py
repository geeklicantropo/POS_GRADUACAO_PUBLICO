import os
import requests
from bs4 import BeautifulSoup

# Define the URL of the page to scrape
url = "https://portal.inmet.gov.br/dadoshistoricos"

# Define the folder where the downloaded files will be saved
output_folder = "dados"
os.makedirs(output_folder, exist_ok=True)

# Define a headers dictionary to include a User-Agent
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

def download_zip_files(start_year=None, end_year=None):
    # Send a GET request to the URL with the headers
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Check if the request was successful

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all links to .zip files
    links = soup.find_all("a", href=True)
    zip_links = [link["href"] for link in links if link["href"].endswith(".zip")]

    for link in zip_links:
        # Extract the year from the link (assuming the year is in the filename)
        year = link.split("/")[-1].split(".")[0]

        # Convert the year to integer for comparison
        year_int = int(year)

        # Check if the year matches the specified range
        if start_year and end_year:
            if not (start_year <= year_int <= end_year):
                continue
        elif start_year:  # If only a single year is specified
            if year_int != start_year:
                continue

        # Create the full URL based on whether the link is absolute or relative
        if link.startswith("https://"):
            full_url = link
        else:
            full_url = f"https://portal.inmet.gov.br{link}"

        file_name = os.path.join(output_folder, f"{year}.zip")

        print(f"Downloading {file_name}...")

        # Download the file with headers and save it
        with requests.get(full_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(file_name, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Downloaded {file_name}")

#Call the function to download files from a range, a single year, or all available files
download_zip_files()  # This will download everything


#download_zip_files(start_year=2005)  # Only downloads the year 2005
#Example usage: download_zip_files(start_year=2000, end_year=2005)  # Downloads from 2000 to 2005
