
Gemini
👋 Hi! Gemini in Drive is here to help you summarize documents and answer questions about your projects.


Gemini may display inaccurate information and does not represent Google's views. Double check responses. Learn more
import requests
from bs4 import BeautifulSoup
import os

def fetch_episode_links(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')
    episode_links = {}
    base_url = base_url.replace('/friends', '')  # Remove trailing slash
    for link in links:
        href = link.get('href')
        if href and 'season' in href:
            full_url = f"{base_url.rstrip('/')}{href}"  # Correctly form the full URL
            episode_number = href.split('/')[-1].replace('.html', '')
            episode_links[episode_number] = full_url
    return episode_links

def download_episode_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # To handle HTTP request errors
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        return text
    except requests.RequestException as e:
        print(f"Failed to download {url}: {str(e)}")
        return None

def save_text(filename, text, directory='transcripts'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, f"{filename}.txt")
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Saved {filename}.txt")

def main():
    base_url = 'https://edersoncorbari.github.io/friends/'
    episode_links = fetch_episode_links(base_url)

    for episode, link in episode_links.items():
        text = download_episode_text(link)
        if text:
            season, episode_number = episode[:2], episode[2:]
            filename = f"{season}{episode_number}"
            save_text(filename, text)

if __name__ == "__main__":
    main()