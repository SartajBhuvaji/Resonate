<br />
<div align="center">
  <a href="https://github.com/SartajBhuvaji/Data-Science-Project/tree/main/">
    <img src="docs/logo.png" alt="logo" width="80" height="80">
  </a>

<h3 align="center">Resonate</h3>

    Data Science Capstone
  <p align="center">
    <br />
    <a href="https://github.com/SartajBhuvaji">Sartaj Bhuvaji</a>
    ·
    <a href="https://github.com/PrachiteeChouhan31">Prachitee Chouhan</a>
    ·
    <a href="https://github.com/madhuroopa">Madhuroopa Irukulla</a>
    ·
    <a href="https://github.com/jay-singhvi">Jay Singhvi</a>
  </p>
</div>


# Resonate: A Retrieval Augmented Framework For Meeting Insight Extraction
In the fast-paced professional realm, meetings serve as vital platforms for collaboration and decision-making. Yet, among the vast exchange of information, recollecting essential details often proves challenging, hindering overall productivity. Imagine a scenario where past discussions on User Interface design are essential but cumbersome to retrieve.

Our project aims to tackle this challenge by developing a solution to effortlessly extract pivotal insights from historical meetings. expeeLeveraging Retrieval Augmented Generation techniques, our proposed system enables users to seamlessly upload meeting records and pose queries for relevant information retrieval. One core component of the system is to group meetings based on their abstractive summaries. Several state-of-the-art clustering algorithms were extensively trained and evaluated. When users pose inquiries, our system will pinpoint the cluster most likely to contain relevant discussions. 

By utilizing the Pinecone vector store database, we retrieve pertinent conversations within a contextual window. The retrieved conversations and custom prompt are then processed through a Large Language Model (LLM) to generate precise responses. Our focus on system optimization involves exploring diverse encoders and LLM models, with fine-tuning to ensure rigorous evaluation and seamless integration. Through our approach, we transcend challenges in conversational meeting summarization, content discovery, and delivering a tailored, high-performance solution designed for user convenience. 

## Getting Started

### Running on Github Codespace

1. Create a Codespace with 4 cores.
2. Press Ctrl+C to cancel the automatic installation of requirements.txt, as it may not install the packages correctly.
3. Manually install required packages:

    ```bash
    pip install -r requirements.txt
    ```
4. Setting environment variables
    - Create a `/config/.env` file and fill in your environment variables.
    - Learn more about config options: README

5. Running the pre-requisits script:

    ```bash
    python init_one_time_utils/pinecone_sample_dataloader.py
    ```

6. Run the application:

    ```bash
    streamlit run app.py
    ```

### Running Locally

1. Clone the repository:

    ```bash
    git clone https://github.com/SartajBhuvaji/Resonate.git
    ```

2. Set up a virtual environment:

    ```bash
    python -m venv .venv
    ```

3. Activate the virtual environment:

    - On Windows:

    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```

    - On Unix or MacOS:

    ```bash
    source .venv/bin/activate
    ```

4. Install dependencies:

    ```bash
    pip install -r requirements.txt --upgrade
    ```

5. Setting environment variables:

    Create a `/config/.env` file and fill in your environment variables.

6. Running the pre-requisite script:

    ```bash
    python init_one_time_utils/pinecone_sample_dataloader.py
    ```

7. Run the application:

    ```bash
    streamlit run app.py
    ```

## Demo

https://github.com/SartajBhuvaji/Resonate/assets/31826483/b7f9ef2c-3839-47d5-94cf-c1d632c70f57

## Framework
![architecture_diagram](https://github.com/SartajBhuvaji/Resonate/assets/31826483/20140140-8c94-4952-9900-334bfe8b335a)
