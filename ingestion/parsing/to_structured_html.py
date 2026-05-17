from bs4 import BeautifulSoup
from pathlib import Path


class HTMLCleaner:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- CLEAN CORE ----------------
    def clean_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. Remove head, style, script
        if soup.head:
            soup.head.decompose()

        for style in soup.find_all("style"):
            style.decompose()

        for script in soup.find_all("script"):
            script.decompose()

        # 2. Keep only important attributes
        for tag in soup.find_all(True):
            tag.attrs = {
                k: v for k, v in tag.attrs.items()
                if k in ['colspan', 'rowspan']
            }

        # 3. Extract body
        if soup.body:
             return soup.body.prettify()

        return soup.prettify().strip()

    # ---------------- FILE PROCESS ----------------
    def process_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_html = f.read()

            cleaned_html = self.clean_html(raw_html)

            output_file = self.output_dir / file_path.name

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_html)

            print(f" Processed: {file_path.name}")

        except Exception as e:
            print(f" Error processing {file_path.name}: {e}")

    # ---------------- BULK PROCESS ----------------
    def process_all(self, ordered=True):
        html_files = list(self.input_dir.glob("*.html"))

        if ordered:
            # Sort by page number (page_1, page_2, ...)
            html_files = sorted(
                html_files,
                key=lambda x: int(x.stem.split("_")[1])
            )

        print(f" Found {len(html_files)} files")

        for file_path in html_files:
            self.process_file(file_path)

        print("\n All files processed successfully!")

# if __name__ == "__main__":
#     cleaner = HTMLCleaner(
#         input_dir="Results/html",             
#         output_dir="Results/structured/html"   
#     )

#     cleaner.process_all()