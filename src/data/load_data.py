import argparse
import enum
import os
from pathlib import Path
import urllib.parse
import tarfile
import zipfile
import requests
import xml.etree.ElementTree as ET

BASE_URL = "https://nedlasting.geonorge.no/geonorge/ATOM-feeds"

def _extract_archive(filepath: str, extract_dir: str) -> bool:
    """_summary_
    Extract zip or tar archives. Returns True if extracted, False otherwise.
    """
    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, "r") as z:
            z.extractall(extract_dir)
        return True
    if tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, "r:*") as t:
            t.extractall(extract_dir)
        return True
    return False

def extract_folder(filename, download_dir, filepath):
    """_summary_
    Unzip files from download directory. 
    """
    # build a sensible extraction folder name by stripping all archive suffixes
    name_no_ext = filename
    while True:
        base, ext = os.path.splitext(name_no_ext)
        if ext.lower() in ('.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz'):
            name_no_ext = base
        else:
            break
    extract_dir = os.path.join(download_dir, name_no_ext)
    os.makedirs(extract_dir, exist_ok=True)

    try:
        extracted = _extract_archive(filepath, extract_dir)
        if extracted:
            print(f"✔ Extracted to {extract_dir}")
        else:
            print(f"ℹ Not an archive or unsupported format: {filename}")
    except Exception as e:
        print(f"✖ Failed to extract {filename}: {e}")

def get_sentinel(url, download_dir, extract_archives: bool = True, folder_limit = 100):
    """_summary_
    Load images from Kartverket and store them as images. 
    """
    # Folder to save files
    os.makedirs(download_dir, exist_ok=True)

    # Download the Atom feed
    response = requests.get(url)
    response.raise_for_status()

    # Parse XML 
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(response.content)

    folders_visited = []
    for entry in root.findall('atom:entry', ns):
        link_elem = entry.find('atom:link[@rel="alternate"]', ns)
        if link_elem is not None:
            file_url = link_elem.attrib['href']
            filename = os.path.basename(file_url)

            print(f"Downloading: {filename}")
            file_response = requests.get(file_url, stream=True)
            file_response.raise_for_status()

            filepath = os.path.join(download_dir, filename)
            with open(filepath, "wb") as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"✔ Saved to {filepath}")

            if extract_archives:
                extract_folder(filename, download_dir, filepath)
            
            # --- BREAK IF LIMIT REACHED ---
            # TODO: find cleaner way to break out of the loop
            folders_visited.append(filepath)
            if len(folders_visited) >= folder_limit:
                break

    print("All downloads completed.")
    return folders_visited

class SentinelRGB(enum.Enum):
    YEAR_2018 = "SatellittdataSentinelSkyfritt2018_AtomFeedTIFF.xml"
    YEAR_2019 = "SatellittdataSentinelSkyfritt2019int16_AtomFeedTIFF.xml"
    YEAR_2020 = "SatellittdataSentinelSkyfritt2020int16_AtomFeedTIFF.xml"
    YEAR_2021 = "SatellittdataSentinelSkyfritt2021int16_AtomFeedTIFF.xml"

    @classmethod
    def get_full_path(cls, year):
        member_name = f"YEAR_{year}"
        try:
            return urllib.parse.urljoin(BASE_URL.rstrip('/') + '/', cls[member_name].value)
        except KeyError:
            raise ValueError(f"Year {year} is not in the enum class")

def main():
    parser = argparse.ArgumentParser(description="Load sentinel2 images")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--full_url", type=str, default=None, help="Allows to input the full URL")
    parser.add_argument("--download_dir", type=str, default="downloads", help="Folder to save downloads")
    parser.add_argument("--folder_limit", type=int, default=100)
    parser.add_argument("--no-extract", dest="extract", action="store_false", help="Do not extract archives")
    parser.set_defaults(extract=True)
    args = parser.parse_args()
    
    if args.full_url is None:
        feed_url = SentinelRGB.get_full_path(args.year)
    else:
        feed_url = args.full_url
    
    print(f"Using feed URL: {feed_url}")
    get_sentinel(str(feed_url), args.download_dir, args.extract, args.folder_limit)

if __name__ == "__main__":
    main()
