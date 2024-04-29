train_img_path = "digitdata/trainingimages"
train_label_path = "digitdata/traininglabels"

BLANK_LINE = (" " * 28) + "\n"

with open(train_img_path, "rt") as f:
    train_x = f.readlines()


for line in train_x:
    if line == BLANK_LINE:
        print("line is blank")
