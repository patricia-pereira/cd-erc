import json

def load_dailydialog(
    path_sents: str,
    path_labels: str,
):
    with open(path_sents, 'r') as f:
        read_sents = f.readlines()

    sents=[]

    for d in range(len(read_sents)):
        new_sents_add = []
        read_sents[d] = read_sents[d].rstrip("\n")
        new_sents = read_sents[d].split(sep="__eou__")
        for i in range(len(new_sents)):
            if new_sents[i] == '':
                continue
            new_sents_add.append(new_sents[i])
        sents += new_sents_add

    with open(path_labels, 'r') as f:
        read_labels = f.readlines()

    dialog_id = []
    labels=[]

    for d in range(len(read_labels)):
        new_labels_add=[]
        read_labels[d]=read_labels[d].rstrip("\n")
        new_labels = read_labels[d].split(sep=" ")
        for i in range(len(new_labels)):
            if new_labels[i]=='':
                continue
            new_labels_add.append(int(new_labels[i]))
        labels+=new_labels_add
        dialog_id += [d]*len(new_labels_add)

    data = []

    for i in range(len(sents)):
        data.append({'text': sents[i], "label": labels[i], "dialog_id": dialog_id[i]})

    return data

def load_emowoz(
    path: str,
):
    f = open(path+"emowoz")
    d = open(path+"data-split.json")
    g = open(path+"emowoz-dialmage.json")
    data1 = json.load(f)
    data2 = json.load(g)
    pointers = json.load(d)
    train = []
    val = []
    test = []
    
    
    for d_id, dialog in enumerate(pointers["train"]["multiwoz"]):
        for id, turn in enumerate(data1[dialog]["log"]) :
            text=turn["text"]
            if id%2==0:
                label=turn["emotion"][3]["emotion"]
            else:
                label=-1
            dialog_id = d_id

            train.append({'text': text, "label": label, "dialog_id": dialog_id})

    for d_id, dialog in enumerate(pointers["train"]["dialmage"]):
        for id, turn in enumerate(data2[dialog]["log"]) : 
            text=turn["text"]
            if id%2==0:
                label=turn["emotion"][3]["emotion"]
            else:
                label=-1
            dialog_id = d_id

            train.append({'text': text, "label": label, "dialog_id": dialog_id})

    for d_id, dialog in enumerate(pointers["dev"]["multiwoz"]):
        for id, turn in enumerate(data1[dialog]["log"]) :
            text=turn["text"]
            if id%2==0:
                label=turn["emotion"][3]["emotion"]
            else:
                label=-1
            dialog_id = d_id

            val.append({'text': text, "label": label, "dialog_id": dialog_id})

    for d_id, dialog in enumerate(pointers["dev"]["dialmage"]):
        for id, turn in enumerate(data2[dialog]["log"]) : 
            text=turn["text"]
            if id%2==0:
                label=turn["emotion"][3]["emotion"]
            else:
                label=-1
            dialog_id = d_id

            val.append({'text': text, "label": label, "dialog_id": dialog_id})

    for d_id, dialog in enumerate(pointers["test"]["multiwoz"]):
        for id, turn in enumerate(data1[dialog]["log"]) :
            text=turn["text"]
            if id%2==0:
                label=turn["emotion"][3]["emotion"]
            else:
                label=-1
            dialog_id = d_id

            test.append({'text': text, "label": label, "dialog_id": dialog_id})

    for d_id, dialog in enumerate(pointers["test"]["dialmage"]):
        for id, turn in enumerate(data2[dialog]["log"]) : 
            text=turn["text"]
            if id%2==0:
                label=turn["emotion"][3]["emotion"]
            else:
                label=-1
            dialog_id = d_id

            test.append({'text': text, "label": label, "dialog_id": dialog_id})
    
    f.close()
    g.close()
    d.close()
 
    return (train, val, test)
