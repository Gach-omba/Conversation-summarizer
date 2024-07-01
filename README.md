# conversation-summarizer
This is a powerful tool designed to summarize lengthy conversations into coherent summaries. You can use it to quickly grasp key points in conversations without having to read through the entire conversation
## Technical Notes About the model
The model is a finetuned version of the google pegasus model. It is finetuned on the Samsung's samsum conversations dataset. It is a 571 million parameters model. The model uses beam search with a length penalty of 0.8 and has a maximum output length of 128. This can be modified in the gen_kwargs parameters. The model has a slight difference from the original published model in the transformers book. It has been optimized to run in a standard GPU without running out of memory. This is done by clearing the cache memory after each batch.

## Using the model
There are two ways to work with the model.
    i) Running the notebook to train the model locally
    ii) Using loading the model using transformers library

### Running the model locally
You can run the model locally by running the notebook. It can run on any environment but it is highly recommended to use Google Colab as it provides a free 15GB GPU.

### Use a pipeline as a high-level helper ( if you want to get running in the shortest time possible)

from transformers import pipeline

pipe = pipeline("text2text-generation", model="Gachomba/pegasus-samsum")

### Load the model directly ( if you want to do additional transformations on the data)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Gachomba/pegasus-samsum")
model = AutoModelForSeq2SeqLM.from_pretrained("Gachomba/pegasus-samsum")

### Sample usage ( code to use it within your python code)

from transformers import pipeline

pipe = pipeline("text2text-generation", model="Gachomba/pegasus-samsum")
gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

def SummaryGeneration(userMessage):
    response = pipe(userMessage, **gen_kwargs)
    return response[0]['generated_text']  

message="Person A: Hey everyone! How's it going?

Person B: Hi! I'm doing well, just finished a big project at work. What about you?

Person C: Hey! Congrats on finishing your project, B. I've been busy with some personal stuff, but it's all good now. How about you, A?

Person A: Thanks, C! I'm doing great. Just got back from a short trip to the mountains. It was so refreshing. How are you doing, D?

Person D: Hi all! I'm doing fine, just dealing with some house renovations. It's a bit chaotic, but I'm excited to see the final result.

Person B: That sounds exciting, D. What kind of renovations are you doing?

Person D: We're redoing the kitchen and adding a small patio in the backyard. It's a lot of work, but I think it'll be worth it."

message = SummaryGeneration(message)
print(message)
