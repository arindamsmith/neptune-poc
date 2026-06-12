import requests
import json
import os
from pathlib import Path

# ── CONFIG — replace these with your actual values ────────────────────────────

API_URL        = "http://localhost:8000/your-endpoint"   # replace with your actual FastAPI POST endpoint
INPUT_FOLDER   = r"C:\Users\aghosh\Downloads\input"      # folder containing your files
OUTPUT_FOLDER  = r"C:\Users\aghosh\Downloads\output"     # folder where chunk JSONs will be saved

CHUNK_SIZE     = 2000   # adjust as needed
CHUNK_OVERLAP  = 200    # adjust as needed

# Set to None to process ALL file types, or specify a list of extensions to filter
ALLOWED_EXTENSIONS = [".pdf"]                        # only PDFs
# ALLOWED_EXTENSIONS = [".docx"]                    # only Word docs
# ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"]    # multiple types
# ALLOWED_EXTENSIONS = None                         # no filter — process everything

# ─────────────────────────────────────────────────────────────────────────────


def call_ingestion_api(file_path: str) -> list[dict]:
    """Call the data ingestion API for a single file and return the list of chunks."""
    payload = {
        "file_path": file_path,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }
    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def save_chunks(chunks: list[dict], output_path: str):
    # """Save chunks to a JSON file."""
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    """Save chunks to a TXT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"--- Chunk {i} ---\n")
            f.write(chunk.get("page_content", "[No page_content found]"))
            f.write("\n\n")


def print_chunks(file_name: str, chunks: list[dict]):
    """Print page_content of each chunk to console."""
    print(f"\n{'='*70}")
    print(f"FILE : {file_name}")
    print(f"TOTAL CHUNKS : {len(chunks)}")
    print(f"{'='*70}")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.get("page_content", "[No page_content found]"))


def process_folder():
    """Main function — iterate over all files in the input folder and process each."""

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Collect all files in the input folder (non-recursive)
    # all_files = [f for f in Path(INPUT_FOLDER).iterdir() if f.is_file()]

    if ALLOWED_EXTENSIONS:
        all_files = [
            f for f in Path(INPUT_FOLDER).iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
        ]
    else:
        all_files = [f for f in Path(INPUT_FOLDER).iterdir() if f.is_file()]

    filter_msg = f"Filter: {ALLOWED_EXTENSIONS}" if ALLOWED_EXTENSIONS else "Filter: None (all file types)"
    print(f"Found {len(all_files)} file(s) in: {INPUT_FOLDER}  |  {filter_msg}")

    if not all_files:
        print(f"No files found in: {INPUT_FOLDER}")
        return

    # print(f"Found {len(all_files)} file(s) in: {INPUT_FOLDER}")

    success_count = 0
    error_count   = 0

    for file_path in all_files:
        print(f"\nProcessing: {file_path.name} ...")

        try:
            # Call the API
            chunks = call_ingestion_api(str(file_path))

            # Print to console
            print_chunks(file_path.name, chunks)

            # Save to individual txt file
            # output_file_name = f"{file_path.stem}_chunks.json"
            output_file_name = f"{file_path.stem}_chunks.txt"
            output_path      = os.path.join(OUTPUT_FOLDER, output_file_name)
            save_chunks(chunks, output_path)

            print(f"\n✅ Saved to: {output_path}")
            success_count += 1

        except requests.exceptions.HTTPError as e:
            print(f"❌ API error for {file_path.name}: {e}")
            error_count += 1

        except Exception as e:
            print(f"❌ Unexpected error for {file_path.name}: {e}")
            error_count += 1

    print(f"\n{'='*70}")
    print(f"DONE — {success_count} succeeded, {error_count} failed.")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"{'='*70}")


def combine_all_chunks():
    """
    Optional — run this separately after process_folder() 
    to merge all individual JSONs into one combined file.
    """
    combined = []
    output_files = list(Path(OUTPUT_FOLDER).glob("*_chunks.json"))

    if not output_files:
        print("No chunk files found to combine.")
        return

    for chunk_file in output_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            # Tag each chunk with its source file name for traceability
            for chunk in chunks:
                chunk["source_file"] = chunk_file.stem.replace("_chunks", "")
            combined.extend(chunks)

    combined_output_path = os.path.join(OUTPUT_FOLDER, "combined_chunks.json")
    with open(combined_output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"✅ Combined {len(combined)} chunks from {len(output_files)} files.")
    print(f"   Saved to: {combined_output_path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 — Process all files and save individual JSONs
    process_folder()

    # Step 2 — Uncomment below when you want to combine all into one file
    # combine_all_chunks()