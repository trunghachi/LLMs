from transformers import pipeline

class ContentFilter:
    def __init__(self, classifier_model="unitary/toxic-bert"):
        # Use a pretrained toxicity classifier
        self.classifier = pipeline("text-classification", model=classifier_model)
        self.threshold = 0.7  # Toxicity threshold
    
    def is_safe(self, text):
        # Classify the text
        result = self.classifier(text)[0]
        label = result["label"]
        score = result["score"]
        
        # Return True if the text is safe (not toxic)
        if label == "toxic" and score > self.threshold:
            return False
        return True

def filter_llm_output(llm_output):
    # Initialize content filter
    content_filter = ContentFilter()
    
    # Check if the output is safe
    if content_filter.is_safe(llm_output):
        return llm_output
    else:
        return "Content blocked due to potential toxicity."

if __name__ == "__main__":
    # Example usage
    sample_outputs = [
        "This is a nice message.",
        "You are an idiot and I hate you!",
    ]
    
    for output in sample_outputs:
        filtered_output = filter_llm_output(output)
        print(f"Input: {output}")
        print(f"Output: {filtered_output}\n")
