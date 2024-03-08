# Resonate

#### Current Phase: Sprint 1

## Project Overview
Resonate is a Retrieval-augmented generation (RAG) powered Large Language Model application that helps you chat with your meetings to answer questions and generate insights. 

## Objectives
- User should be able to upload an audio/video meeting file along with a meeting `Topic`
- There can be multiple meeting topics. With each topic having a series of meetings.
- Use would then be able to choose a `topic` and chat with the meeting just and ask any question

## Initial Sketches

RAG Inference
- The user would select the meeting `Topic` and ask a question.
- Pinecone would retrieve relevant information and would feed the LLm with custom prompt, context, and the user query.
- We also plan to add a `Semantic Router` to route queries according to the user input.
- The LLm would then generate the result and answer the question.

![image](https://github.com/SartajBhuvaji/Resonate/assets/31826483/e4e01b5e-d29b-4591-af3a-f7594ac85a2c)

Data Store
- The below diagram shows how we plan to store data using `Pinecone` which is a popular Vector DB.
- User would upload meetings in audio/video format.
- We would use `AWS Transcribe` to diarize and transcribe the audio file into `timestamp, speaker, text` (this is simplified)
- We would embed the text data into vectors that would be uploaded to Pinecone serverless.

![image](https://github.com/SartajBhuvaji/Resonate/assets/31826483/a89fddc3-f020-4b9e-9904-ac2966f9b0e2)

Research
- We would try multiple `Vector embeddings` and also fine-tune `LLM Models` using `Microsoft DeepSpeed` on the custom dataset and compare the performance of these models.
![image](https://github.com/SartajBhuvaji/Resonate/assets/31826483/bd4559b3-780f-428e-ae13-a885008e858f)


Proposed UI
- Below is the sketch of proposed UI.
![image](https://github.com/SartajBhuvaji/Resonate/assets/31826483/b60ae38f-b727-4bc6-b94b-491336833981)
