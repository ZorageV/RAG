# Retrieval-Augmented-Generation

## Description
Welcome to the project, This is a  Retrieval-Augmented-Generation system made in Python and the LLama-index framework. It allows users to chats with a set of pds documents.

## Instructions
1. Fork this repository
2. Clone the forked repository
3. create a python virtual environment and activate it
4. run the requiremnt.txt to install all the required modules
5. Check if you have visual studio tools(build tools with desktop development c++ tools) and cpp compiler installed and running
6. Sign up on Hugging face to get your own token aand paste it in  hft_token to access hugging face modules
7. Add the pdfs to test pdfs folder
8. run the main.py file and wait (Depending on your device it may take some time to generate responses, please be patient)


## Building Process 

### Choosing the framework:
I tried implementing rag using langchain and llamaindex, but found llamaindex to be better when dealing with multiple files and large files. Langchain on the other offers friendly approach and each process can be easily understood in it, whereas llamaindex uses a central llm which performs all the tasks in a pipeline system so customization can get a little heavy when you are not familar.Llama index in much better in terms of perofrmance and for production purposes.

### Open Source Resources Used:
The Large language model used, embeddings used and database used are all open source so anybody can easily run the model on their system.if you wish to use other llms you are free to go thorugh the llama index documentation.


