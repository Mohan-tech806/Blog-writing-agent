# Blog Writing Agent

## Problem Statement
Writing a well-structured blog requires topic planning, research, content writing, image creation, and reviewing. Performing these tasks manually can take significant time.

The Blog Writing Agent automates this process using a multi-agent workflow built with LangGraph.

## Objectives
- Generate structured blogs from a user-provided topic.
- Support technical and non-technical blog styles.
- Research relevant information from the web.
- Include citations and source links near relevant claims.
- Generate and place images within the blog.
- Display the generated blog through a Streamlit interface.
- Provide options to download the blog.

## Tools and Technologies
- Python
- LangGraph
- Streamlit
- Google Gemini
- Tavily
- Cloudflare Workers AI
- Markdown

## Workflow
1. User enters a topic and selects the blog preferences.
2. The workflow plans the blog structure.
3. The research agent collects relevant information when required.
4. The writing agents generate the blog sections.
5. The generated sections are combined into a complete blog.
6. The image generation step creates and places images where required.
7. The final blog is displayed in the Streamlit interface.
8. The user can preview and download the generated content.

## Features

### Blog Generation
- Generates blogs based on user-provided topics.
- Supports technical and non-technical writing styles.
- Allows customization of the audience and tone.
- Creates a structured blog with headings and sections.

### Web Research
- Uses Tavily to retrieve relevant web information.
- Supports research-based blog generation.
- Includes citations and source links near relevant claims.
- Provides a Sources section at the end of the blog.

### Image Generation
- Generates images using Cloudflare Workers AI.
- Places generated images within the blog content.
- Displays images in the Markdown preview.

### Streamlit Interface
- Provides an interactive interface for blog generation.
- Displays the blog plan, research evidence, preview, images, and logs.
- Supports Markdown and ZIP downloads.
- Allows users to view previously generated blogs.




## Application Screenshot

<img width="862" height="482" alt="streamlit_screenshot" src="https://github.com/user-attachments/assets/5e295c6a-26cf-4dc8-bc95-cd92bbcd8a09" />



## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Blog-Writing-Agent
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not included, install the packages required by the backend and frontend.

### 3. Configure Environment Variables

Create a file named `.env` in the project root directory.

Add the API keys and credentials required by the backend:

```env
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
```

Use the exact environment variable names expected by `bwa_backend.py`. Add or remove entries based on the services enabled in your version of the project.

#### Important
- Do not upload the `.env` file to GitHub.
- Add `.env` to `.gitignore` to prevent accidental uploads.
- Do not include API keys in the README or Python source files.
- Anyone running the project must create their own `.env` file and provide their own API keys.
- If an API key is accidentally exposed, revoke or regenerate it.

## Running the Project

Run the Streamlit application:

```bash
streamlit run bwa_frontend.py
```

Open the local URL displayed in the terminal to use the application.

## Output
The generated blog content and related files are saved in the `blog_output/` directory.

The application provides:
- Generated blog in Markdown format
- Research evidence and source links
- Generated images
- Blog preview
- Download options

## Key Observation
The project demonstrates how a multi-agent workflow can coordinate planning, research, writing, and image generation to automate blog creation.
