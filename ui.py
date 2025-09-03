import tkinter as tk
from tkinter import filedialog, scrolledtext
from embed import embed
from generate import answer_question

import os
print("[DEBUG] Current working directory:", os.getcwd())

class StudyBuddyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("StudyBuddy: PDF Question Answering")

        # --- Top: Upload + Status ---
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)

        self.upload_btn = tk.Button(top_frame, text="Upload PDF", command=self.upload_pdf)
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(top_frame, text="No file uploaded.")
        self.status_label.pack(side=tk.LEFT, padx=10)   

        # --- Middle: LLM Answer Output ---
        output_frame = tk.Frame(root)
        output_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        tk.Label(output_frame, text="Answer:").pack(anchor="w")
        self.answer_box = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.answer_box.pack(fill=tk.BOTH, expand=True)

        # --- Bottom: Question Input ---
        question_frame = tk.Frame(root)
        question_frame.pack(pady=10)

        tk.Label(question_frame, text="Question:").pack(anchor="w")
        self.question_entry = tk.Entry(question_frame, width=80)
        self.question_entry.pack(side=tk.LEFT, padx=5)

        self.ask_btn = tk.Button(question_frame, text="Ask", command=self.ask_question)
        self.ask_btn.pack(side=tk.LEFT)

    def upload_pdf(self):
        """
        User uploads their chosen pdf, and the program chunks and embed the pdf into
        a vector db for quick retrieval to provide context to llama so it can answer
        questions with context.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf")],
            title="Select a PDF file"
        )

        if file_path:
            # load file path
            self.file_path = file_path
            self.status_label.config(text="Loading and chunking...")
            self.root.update()  # Update UI before processing

            try:
                embedded = embed(file_path)
                if not embedded:
                    self.status_label.config(text="Embedding failed ❌")
                    return
                self.chunks = embedded
                self.status_label.config(text=f"{len(self.chunks)} chunks ready")
            except Exception as e:
                # messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")
                self.status_label.config(text="Upload failed")

    # Parse user question
    def ask_question(self):
        query = self.question_entry.get().strip()
        if not query:
            self.answer_box.insert(tk.END, "Please enter a question.\n")
            return

        self.answer_box.delete(1.0, tk.END)  # Clear previous output
        self.answer_box.insert(tk.END, "Thinking...\n")
        self.root.update()  # Force UI update before processing
        try:
            reply = answer_question(query)

            self.answer_box.delete("1.0", tk.END)
            self.answer_box.insert(tk.END, reply)
            self.status_label.config(text="Answer ready ")
        except Exception as e:
            self.answer_box.insert(tk.END, f"Error: {str(e)}\n")
            self.status_label.config(text="Failed to generate answer ")

if __name__ == "__main__":
    root = tk.Tk()
    app = StudyBuddyApp(root)
    root.mainloop()