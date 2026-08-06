def read_essay(path):
    with open(path, encoding="utf-8") as file:
        text = file.read()
        return text
