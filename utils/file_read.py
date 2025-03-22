import re
import pandas as pd

def relation_val(text):
    x = re.search(r"^\s*@relation\s+([a-z][a-z0-9-_]*)", text, re.MULTILINE | re.IGNORECASE)
    try:
        return x.group(1)
    except:
        return None

def attribute_val(text):
    x = re.findall(r"^\s*@attribute\s+([a-z][a-z0-9-_]*)\s+{([^}]*)}", text, re.MULTILINE | re.IGNORECASE)
    y = pd.DataFrame(x)
    print(y)

def read_arff(text):
    print(f"Relation: {relation_val(text)}")
    print(f"Attributes: {attribute_val(text)}")

def read_csv(f_location):
    return pd.read_csv(f_location)

if __name__ == "__main__":
    x = "@relation contact-lenses\n\
        @attribute age 			{young, pre-presbyopic, presbyopic}\n\
        @attribute spectacle-prescrip	{myope, hypermetrope}\n\
        @attribute astigmatism		{no, yes}\n\
        @attribute tear-prod-rate	{reduced, normal}\n\
        @attribute contact-lenses	{soft, hard, none}\n"
    attribute_val(x)