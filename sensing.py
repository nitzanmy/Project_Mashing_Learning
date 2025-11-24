
import pandas as pd
from pathlib import Path

# Constants
DATA_DIR = Path("data")
OUTPUT_FILE = "sensed_data.csv"
HEADER_SEPARATOR = "-" * 60 # Separator used to distinguish the header from the body in TXT files


def extract_data_from_file(file_path):
    """
    Opens a TXT file, separates the header from the body, and extracts the fields.
    """
    try:
        # Read the entire file content
        content = file_path.read_text(encoding="utf-8")
        
        # Separate the article body from the header
        parts = content.split(HEADER_SEPARATOR + '\n', 1)
        if len(parts) < 2:
            print(f"Skipping file {file_path}: Header separator not found.")
            return None

        header_str = parts[0].strip()
        body_text = parts[1].strip()
        
        # Extract data from the header (line by line)
        header_lines = header_str.split('\n')
        
        # Line 1: The Guardian | Section | Date
        metadata_parts = header_lines[0].split(' | ')
        section = metadata_parts[1].strip() if len(metadata_parts) > 1 else 'N/A'
        
        # Line 2: By Author
        byline = header_lines[1].replace('By ', '').strip() if len(header_lines) > 1 else 'Unknown Author'
        
        # Line 3: URL
        url = header_lines[2].strip() if len(header_lines) > 2 else 'N/A'
        
        # Line 5: Title (Line 4 is usually blank)
        title = header_lines[4].strip() if len(header_lines) > 4 else 'Untitled'

        # Note: The Label is derived from the directory name itself
        article_label = file_path.parent.name
        
        return {
            "Label": article_label,
            # "Section": section,
            "Title": title,
            "Byline": byline,
            "URL": url,
            "BodyText": body_text
        }
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def main():
    all_data = []
    
    # Loop over all first-level directories within DATA_DIR (news, sport, culture, etc.)
    for section_dir in DATA_DIR.iterdir():
        if section_dir.is_dir():
            print(f"Processing directory: {section_dir.name}")
            
            # Loop over all TXT files inside the directory
            for file_path in section_dir.glob("*.txt"):
                article_data = extract_data_from_file(file_path)
                if article_data:
                    all_data.append(article_data)

    if not all_data:
        print("No articles were successfully processed.")
        return

    # Create the DataFrame (data table)
    df = pd.DataFrame(all_data)
    
    # Save the data to a CSV file
    output_path = Path(OUTPUT_FILE)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nSuccessfully created {output_path.name} with {len(df)} rows.")
    print("Columns included: Label, Section, Title, Byline, URL, BodyText.")


if __name__ == "__main__":
    # Check for the required pandas library installation
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is not installed. Please install it using: pip install pandas")
        
    main()