def parse_synset_name_file(filepath):
    synsets = []
    names = []
    mapping = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            synset = parts[0]
            name = " ".join(parts[1:])

            synsets.append(synset)
            names.append(name)
            mapping[synset] = name

    return synsets, names, mapping

filepath = "../../data/imagenet-r-names.txt"

synsets, names, mapping = parse_synset_name_file(filepath)

print(names)