import os
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


# Function to extract text from PDF files
def extract_text_from_pdfs(pdf_files):
    combined_text = ""

    for pdf_file in pdf_files:
        print(f"Reading PDF: {pdf_file}")  # Print statement to indicate which PDF is being read
        with open(pdf_file, "rb") as file:  # Open the PDF file in binary mode
            reader = PdfReader(file)  # Use PdfReader for newer versions of PyPDF2
            for page in reader.pages:
                combined_text += page.extract_text() + "\n"  # Extract and accumulate text

    return combined_text


# Function to create a new PDF from extracted text
def create_pdf_from_text(text, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    lines = text.split("\n")
    y_position = height - 40  # Start near the top of the page

    for line in lines:
        if y_position < 40:  # If near the bottom of the page
            c.showPage()  # Start a new page
            y_position = height - 40  # Reset to top of the new page

        c.drawString(40, y_position, line)
        y_position -= 15  # Move down by 15 pixels for the next line

    c.save()


# Main function to combine multiple PDFs' text into one PDF
def main(pdf_folder, output_pdf):
    # Check if the folder exists
    if not os.path.exists(pdf_folder):
        print(f"Directory does not exist: {pdf_folder}")
        return

    # Get all PDF files from the folder
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in the folder: {pdf_folder}")
        return

    # Extract text from the PDF files
    combined_text = extract_text_from_pdfs(pdf_files)

    # Create a new PDF from the combined text
    create_pdf_from_text(combined_text, output_pdf)
    print(f"Text extracted and saved into {output_pdf}")


if __name__ == "__main__":
    # Folder containing PDF files
    pdf_folder = r"C:/Users/skp68/Downloads/ayurvedic-doctor-bot-master/ayurvedic-doctor-bot-master/data"

    # Output PDF file
    output_pdf = "combined_output.pdf"

    # Run the main function
    main(pdf_folder, output_pdf)
