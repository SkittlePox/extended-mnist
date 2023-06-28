import nltk
from nltk import CFG
from random import choice

def generate_propositions(digits_in_image):
    grammar = CFG.fromstring("""
        proposition -> digit 'and' proposition | digit 'or' proposition | digit
        digit -> 'not' number | number
        number -> '0' | '1' | '2' | '3' | '4'
    """)

    propositions = []

    for proposition in nltk.parse.generate.generate(grammar, depth=5):
        propositions.append(' '.join(proposition))

    true_propositions = []
    false_propositions = []

    for proposition in propositions:
        truth_value = eval_proposition(proposition, digits_in_image)
        if truth_value:
            true_propositions.append(proposition)
        else:
            false_propositions.append(proposition)

    return true_propositions, false_propositions

def eval_proposition(proposition, digits_in_image):
    # Use dictionary to map string digits to boolean values based on their presence in the image
    digit_dict = {str(i): str(i in digits_in_image) for i in range(5)}
    
    # Replace the digit strings in the proposition with their truth values and 'and', 'or', 'not' with respective Python operators
    for k, v in digit_dict.items():
        proposition = proposition.replace(k, v)
    
    proposition = proposition.replace('and', 'and').replace('or', 'or').replace('not ', 'not ')
    
    # Evaluate the proposition and return the result
    return eval(proposition)

# Test with some image data
digits_in_image = [0, 1, 4]
true_propositions, false_propositions = generate_propositions(digits_in_image)

print("True Propositions:")
for prop in true_propositions:
    print(prop)

print("\nFalse Propositions:")
for prop in false_propositions:
    print(prop)
