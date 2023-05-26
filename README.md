# Tokenizer-for-text-model

Help people who want to build a text based deep learning model with a tokenizer

## Before using

Install the required dependencies using pip:

Run the following command in the terminal or command prompt:
```bash
pip install sentencepiece
```
This command will install the sentencepiece library, which is required for the Tokenizer class.


## How to use it ?

1. Import the Tokenizer class from the module where it is defined. Assuming the code you provided is in a file named tokenizer.py, you can import it as follows:

```python
from tokenizer import Tokenizer
```

2. Create an instance of the Tokenizer class by providing the path to the SentencePiece model file. For example:

```python
tokenizer = Tokenizer("tokenizer.model")
```

3. You can now use the tokenizer to encode and decode text. The encode method takes a string as input and returns the encoded tokens as a list of integers. Here's an example:

```python
text = "Hello, how are you?"
encoded_tokens = tokenizer.encode(text, bos=True, eos=True)
print(encoded_tokens)
```

In the example above, bos=True adds the BOS token to the beginning of the encoded tokens, and eos=True adds the EOS token to the end. You can adjust these parameters based on your requirements.

4. The decode method takes a list of integers (encoded tokens) and returns the decoded string. Here's an example:

```python
tokens = [1, 34, 56, 78, 2]
decoded_text = tokenizer.decode(tokens)
print(decoded_text)
```

In the example above, tokens is a list of integers representing encoded tokens, and decoded_text will contain the decoded string.

Make sure you have the necessary dependencies installed, including the sentencepiece library, which provides the SentencePiece functionality used by the Tokenizer class.
