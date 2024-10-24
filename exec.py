import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set paths using environment variables
data_dir = os.getenv('DATA_DIR')

def main():
   print(data_dir)
   print('Works')

if __name__ == '__main__':
   main()

