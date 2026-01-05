import os
from langchain.text_splitters import RecursiveCharacterTextSplitter

INPUT_DIR = "data/cleaned"
OUTPUT_DIR = "data/chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

for file in os.listdir(INPUT_DIR):
    with open(os.path.join(INPUT_DIR, file), encoding="utf-8") as f:
        chunks = splitter.split_text(f.read())

    for i, chunk in enumerate(chunks):
        out = f"{file.replace('.txt','')}_chunk_{i}.txt"
        with open(os.path.join(OUTPUT_DIR, out), "w", encoding="utf-8") as f:
            f.write(chunk)

    print(f"{file}: {len(chunks)} chunks")
