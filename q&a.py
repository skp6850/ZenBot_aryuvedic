import os
import csv
import time
from langchain_ollama import OllamaLLM

def read_markdown_files(directory: str) -> str:
    """Reads Markdown files from the specified directory and returns their combined text."""
    combined_text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                combined_text += file.read() + "\n"
    return combined_text

def chunk_text(text: str, chunk_size: int = 1000) -> list:
    """Chunks the input text into smaller chunks based on the specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_qa_pairs(text_chunk: str) -> str:
    """Generates question and answer pairs using the LLaMA 3.2 model."""
    prompt = f"Based on the following text, generate 3 question and answer pairs:\n\n{text_chunk}\n\nPlease provide Q&A pairs:"

    # Initialize Ollama with LLaMA 3.2
    llm = OllamaLLM(model="llama3.2")  # Use "llama2" as the model name

    try:
        # Call the model to generate the completion
        response = llm(prompt)
        return response
    except Exception as e:
        print(f"Error generating Q&A pairs: {e}")
        return ""

def parse_qa_pairs(qa_text: str) -> list:
    """Parses the input text into a list of question and answer pairs."""
    qa_list = []
    pairs = qa_text.split("\n\n")  # Assuming each Q&A pair is separated by a double newline
    for pair in pairs:
        if "?" in pair:  # Assuming questions contain a '?'
            question, answer = pair.split("?", 1)
            qa_list.append((question.strip() + "?", answer.strip()))  # Re-add "?" to the question
    return qa_list

def save_to_csv(output_file: str, qa_pairs: list):
    """Saves the Q&A pairs to a CSV file."""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Question", "Answer"])  # Write the header row

            for question, answer in qa_pairs:
                csv_writer.writerow([question, answer])
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def main(markdown_directory: str, output_csv_file: str):
    """Main function that reads Markdown files and generates Q&A pairs."""
    combined_text = read_markdown_files(markdown_directory)
    text_chunks = chunk_text(combined_text)

    start_time = time.time()
    total_chunks = len(text_chunks)

    qa_pairs_list = []

    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i + 1}/{total_chunks} for Q&A generation...")

        try:
            qa_pairs_text = generate_qa_pairs(chunk)
            qa_pairs = parse_qa_pairs(qa_pairs_text)

            qa_pairs_list.extend(qa_pairs)

            elapsed_time = time.time() - start_time
            estimated_time_per_chunk = elapsed_time / (i + 1)
            remaining_chunks = total_chunks - (i + 1)
            remaining_time = estimated_time_per_chunk * remaining_chunks
            print(f"Elapsed time: {elapsed_time:.2f} seconds, Estimated time remaining: {remaining_time:.2f} seconds")
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            continue

    save_to_csv(output_csv_file, qa_pairs_list)

    print(f"Q&A pairs saved to {output_csv_file}")

if __name__ == "__main__":
    markdown_directory ="md"
    output_csv_file = "qa_output.csv"
    main(markdown_directory, output_csv_file)