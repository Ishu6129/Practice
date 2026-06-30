from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader,CSVLoader

pdf_loader = DirectoryLoader(
    path="../FILES",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

txt_loader = DirectoryLoader(
    path="../FILES",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

csv_loader=DirectoryLoader(
    path="../FILES",
    glob="**/*.csv",
    loader_cls=CSVLoader,
)

print("PDF Documents")
c=0
for doc in pdf_loader.lazy_load():
    c+=1
    print(doc.metadata)

print("\nText Documents")
for doc in txt_loader.lazy_load():
    c+=1
    print(doc.metadata)
print("\nCSV Document")
for doc in csv_loader.load():
    c+=1
    print(doc.metadata)
print("total docs: ",c)