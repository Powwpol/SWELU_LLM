# Training Data Overview

This directory contains 8 text files with diverse content for training the MambaSWELU language model.

## Files Description

1. **scientific_texts.txt** (2.8 KB, 19 lines)
   - Machine learning concepts
   - Deep learning fundamentals
   - Neural network architectures
   - AI training techniques

2. **programming_concepts.txt** (4.0 KB, 29 lines)
   - Python programming
   - Data structures and algorithms
   - Software development practices
   - Database and API design

3. **general_knowledge.txt** (4.2 KB, 29 lines)
   - Earth science
   - Biology and genetics
   - Physics and chemistry
   - Astronomy and cosmology

4. **mathematics.txt** (4.0 KB, 29 lines)
   - Algebra and calculus
   - Geometry and trigonometry
   - Probability and statistics
   - Linear algebra and logic

5. **conversations.txt** (5.2 KB, 44 lines)
   - Q&A format
   - Technical explanations
   - Conversational style
   - Educational content

6. **stories.txt** (5.3 KB, 29 lines)
   - Narrative writing
   - Descriptive passages
   - Various scenarios and settings
   - Creative storytelling

7. **code_explanations.txt** (5.1 KB, 39 lines)
   - Python code examples
   - Programming patterns
   - Best practices
   - Practical implementations

8. **technical_documentation.txt** (4.2 KB, 157 lines)
   - API documentation
   - Database schemas
   - Configuration guides
   - Installation instructions
   - Troubleshooting tips

## Statistics

- **Total files**: 8
- **Total size**: 56 KB
- **Total lines**: 375 lines
- **Content diversity**: High (technical, scientific, creative, conversational)

## Usage

These files can be used to train the MambaSWELU model using the data_prep.py module:

```python
from src.data_prep import TextDataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Read and tokenize all training files
token_ids = []
for file in glob.glob("input_training_data/*.txt"):
    with open(file, 'r') as f:
        text = f.read()
        tokens = tokenizer.encode(text)
        token_ids.extend(tokens)

# Create dataset
dataset = TextDataset(token_ids, seq_len=2048)
```

## Content Quality

The training data includes:
- ✅ Diverse topics and writing styles
- ✅ Technical and educational content
- ✅ Proper grammar and structure
- ✅ Mix of formal and informal language
- ✅ Code examples and documentation
- ✅ Conversational Q&A patterns

## Next Steps

1. Tokenize all text files using GPT-2 tokenizer
2. Create TextDataset instances
3. Set up DataLoader with appropriate batch size
4. Begin training with train.py
5. Monitor loss and perplexity metrics
