import os
from langchain_community.document_loaders import WikipediaLoader

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAGES = [
    "House Stark", "House Lannister", "Jon Snow",
    "Daenerys Targaryen", "The Red Wedding"
]

for title in PAGES:
    docs = WikipediaLoader(query=title, load_max_docs=1).load()
    for i, doc in enumerate(docs):
        path = os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_')}_{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.page_content)

        print(f"Saved → {path}")
