#!/usr/bin/env python3
import os
import argparse
import zipfile
import urllib.request
import shutil
from tqdm import tqdm


def download_file(url, dest):
    """Download a file with progress bar.
    
    Args:
        url (str): URL of the file to download
        dest (str): Destination path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # Download the file with progress bar
    with urllib.request.urlopen(url) as response:
        total_size = int(response.info().get('Content-Length', 0))
        with open(dest, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest)) as pbar:
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path, extract_path):
    """Extract a zip file.
    
    Args:
        zip_path (str): Path to the zip file
        extract_path (str): Path to extract to
    """
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting'):
            zip_ref.extract(member, extract_path)


def download_cyclegan_dataset(dataset_name, data_dir='./datasets'):
    """Download a CycleGAN dataset.
    
    Args:
        dataset_name (str): Name of the dataset to download
            (e.g., 'monet2photo', 'apple2orange', etc.)
        data_dir (str): Directory to save the dataset
    """
    # CycleGAN datasets URL
    url = f'http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset_name}.zip'
    
    # Download and extract the dataset
    zip_path = os.path.join(data_dir, f'{dataset_name}.zip')
    extract_path = os.path.join(data_dir, dataset_name)
    
    print(f'Downloading {dataset_name} dataset...')
    download_file(url, zip_path)
    
    print(f'Extracting {dataset_name} dataset...')
    extract_zip(zip_path, extract_path)
    
    # Remove the zip file
    os.remove(zip_path)
    
    print(f'Dataset downloaded and extracted to {extract_path}')


def download_wikiart_styles(data_dir='./datasets'):
    """Download a subset of WikiArt styles for training.
    
    Args:
        data_dir (str): Directory to save the dataset
    """
    # WikiArt styles subset URL (using a public subset)
    url = 'https://github.com/xunhuang1995/AdaIN-style/releases/download/data/style_data.zip'
    
    # Download and extract the dataset
    zip_path = os.path.join(data_dir, 'wikiart_styles.zip')
    extract_path = os.path.join(data_dir, 'wikiart_styles')
    style_dir = os.path.join(data_dir, 'style')
    
    print('Downloading WikiArt styles dataset...')
    download_file(url, zip_path)
    
    print('Extracting WikiArt styles dataset...')
    extract_zip(zip_path, extract_path)
    
    # Move the style images to the style directory
    if os.path.exists(os.path.join(extract_path, 'style')):
        os.makedirs(style_dir, exist_ok=True)
        for style_img in os.listdir(os.path.join(extract_path, 'style')):
            src = os.path.join(extract_path, 'style', style_img)
            dst = os.path.join(style_dir, style_img)
            shutil.copy(src, dst)
    
    # Remove the zip file and extracted directory
    os.remove(zip_path)
    shutil.rmtree(extract_path)
    
    print(f'Style images downloaded and extracted to {style_dir}')


def create_dataset_structure(dataset_name, data_dir='./datasets'):
    """Create the dataset directory structure.
    
    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Directory to save the dataset
    """
    # Create directories
    dirs = ['trainA', 'trainB', 'testA', 'testB', 'style']
    for d in dirs:
        os.makedirs(os.path.join(data_dir, dataset_name, d), exist_ok=True)
    
    print(f'Dataset directory structure created at {os.path.join(data_dir, dataset_name)}')


def main():
    """Main function to download datasets."""
    parser = argparse.ArgumentParser(description='Download datasets for AdaIN-CycleGAN')
    parser.add_argument('--dataset', type=str, default='monet2photo',
                        help='name of the dataset to download (default: monet2photo)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='directory to save the datasets (default: ./datasets)')
    parser.add_argument('--download_styles', action='store_true',
                        help='download WikiArt styles dataset')
    parser.add_argument('--create_structure', action='store_true',
                        help='create empty dataset directory structure')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Download CycleGAN dataset
    if args.dataset:
        download_cyclegan_dataset(args.dataset, args.data_dir)
    
    # Download WikiArt styles dataset
    if args.download_styles:
        download_wikiart_styles(args.data_dir)
    
    # Create empty dataset directory structure
    if args.create_structure:
        create_dataset_structure(args.dataset, args.data_dir)


if __name__ == '__main__':
    main() 