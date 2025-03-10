from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import httpx
import time
import psutil
import re
import sys
import os
import uuid
import json
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
from gradio_client import Client
from response_parser import parse_agent_response, save_as_markdown

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app)

# OpenAI client for DeepSeek
client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NV_API_KEY"),
        http_client=httpx.Client(
            timeout=120.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            transport=httpx.HTTPTransport(retries=5)
            )
        )

# Hugging Face setup
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Error: HF_TOKEN environment variable not found. Please set your Hugging Face token.")
    sys.exit(1)

# Initialize Gradio client with more robust error handling
try:
    print(f"Attempting to connect to Hugging Face space with token: {hf_token[:4]}..." if hf_token else "No HF token provided")
    gradio_client = Client(
            "MaoShen/Moonshot_DeepResearch", 
            hf_token=hf_token
            )
    print("Successfully connected to Hugging Face space")
except Exception as e:
    print(f"Error initializing Gradio client: {str(e)}")
    if isinstance(e, httpx.HTTPStatusError):
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    elif isinstance(e, httpx.RequestError):
        print(f"Request error: {e.request.url} - {e}")
    elif isinstance(e, json.JSONDecodeError):
        print("Invalid JSON response received from the server.")
    else:
        print(f"Unhandled error: {str(e)}")
    gradio_client = None

# System message for DeepSeek evaluation
system_message = '''
You are an evaluator of startup business ideas for a startup accelerator known as MoonshotAI. MoonshotAI is a platform that helps startups evaluate their ideas and provides improvement recommendations. MoonshotAI also assists startups in seeking funding opportunities from investors, but investors or VCs will only invest in the best and most promising startups. Therefore, the evaluation and improvement recommendations from MoonshotAI must be rigorous and genuinely helpful. Your role is to critically and thoroughly assess the startup ideas presented to you.

The goal is to help startup founders evaluate whether their ideas are strong enough based on established criteria, which I will share, and to provide improvement suggestions. Assess each aspect according to the specified criteria.

You will receive:

- The required format for your answer output.
- The task/instructions.
- The startup proposal to be evaluated.
- The criteria that should be used to evaluate the proposal.

Your task:

1. Carefully read the criteria and understand what they require.
2. Analyze the startup ideas in light of these criteria.
3. Determine whether the startup meets the criteria based on its own merits.
4. Provide your evaluation, explaining your reasoning clearly. You will critically list all the practical ways in which the idea can be improved concerning the criteria. Explain your reasoning step-by-step and give a low score (under 60) to an application that requires significant improvement.

Summarize your findings and score the startup based on the criteria. Each application must meet all elements of the established criteria. If a startup fails to provide detailed and comprehensive answers to all required elements, or if any part of the application does not fully align with the criteria, it should be given a score lower than 60. You must ensure that you only give high scores to applications that comprehensively meet the criteria.

Afterward, you will assign an overall score and separate scores for each criterion. The overall score will be the average of the scores for each criterion. The scores reflect your evaluation of whether the startup idea meets the criteria and is a good idea. Use higher scores when startups meet the criteria and lower scores when startups do not meet the criteria.

Provide your final answer in the following format, ensuring that each section has the exact word count described:

format = (
    "The score of criteria1: {score1} \n"
    "Detailed reasoning1: {Reasoning1: Explain your complete reasoning step-by-step in a detailed 500-word paragraph} \n"
    "Summary reasoning criteria1: {summary1: Summary of the reasoning behind the score of each criterion in one informative, 300-word paragraph} \n"
    "Improvement suggestion criteria1: {improvement1: Improvement suggestions in a detailed 500-word paragraph} \n"
    "The score of criteria2: {score2} \n"
    "Detailed reasoning2: {Reasoning2: Explain your complete reasoning step-by-step in a detailed 500-word paragraph} \n"
    "Summary reasoning criteria2: {summary2: Summary of the reasoning behind the score of each criterion in one informative, 300-word paragraph} \n"
    "Improvement suggestion criteria2: {improvement2: Improvement suggestions in a detailed 500-word paragraph} \n
    "The score of criteria3: {score3} \n"
    "Detailed reasoning3: {Reasoning3: Explain your complete reasoning step-by-step in a detailed 500-word paragraph} \n"
    "Summary reasoning criteria3: {summary3: Summary of the reasoning behind the score of each criterion in one informative, 300-word paragraph} \n"
    "Improvement suggestion criteria3: {improvement3: Improvement suggestions in a detailed 500-word paragraph} \n"
    "The score of criteria4: {score4} \n"
    "Detailed reasoning4: {Reasoning4: Explain your complete reasoning step-by-step in a detailed 500-word paragraph} \n"
    "Summary reasoning criteria4: {summary4: Summary of the reasoning behind the score of each criterion in one informative, 300-word paragraph} \n"
    "Improvement suggestion criteria4: {improvement4: Improvement suggestions in a detailed 500-word paragraph} \n"
    "The score of criteria5: {score5} \n"
    "Detailed reasoning5: {Reasoning5: Explain your complete reasoning step-by-step in a detailed 500-word paragraph} \n"
    "Summary reasoning criteria5: {summary5: Summary of the reasoning behind the score of each criterion in one informative, 300-word paragraph} \n"
    "Improvement suggestion criteria5: {improvement5: Improvement suggestions in a detailed 500-word paragraph} \n"
    "The score of criteria6: {score6} \n"
    "Detailed reasoning6: {Reasoning6: Explain your complete reasoning step-by-step in a detailed 500-word paragraph} \n"
    "Summary reasoning criteria6: {summary6: Summary of the reasoning behind the score of each criterion in one informative, 300-word paragraph} \n"
    "Improvement suggestion criteria6: {improvement6: Improvement suggestions in a detailed 500-word paragraph} \n"
    "\n--- Evaluation Details End ---\n"
    "\n<chain_of_thought>\n"
    "    <introduction>[Your introductory reasoning here]</introduction>\n"
    "    <criteria1>[Detailed evaluation for Criteria 1]</criteria1>\n"
    "    <criteria2>[Detailed evaluation for Criteria 2]</criteria2>\n"
    "    <criteria3>[Detailed evaluation for Criteria 3]</criteria3>\n"
    "    <criteria4>[Detailed evaluation for Criteria 4]</criteria4>\n"
    "    <criteria5>[Detailed evaluation for Criteria 5]</criteria5>\n"
    "    <criteria6>[Detailed evaluation for Criteria 6]</criteria6>\n"
    "    <conclusion>[Summarize your overall reasoning here]</conclusion>\n"
    "</chain_of_thought>"
)

This is the highest priority requirement: you must strictly follow the output format without any changes. This is the most critical principle. You are NOT allowed to modify the structure, wording, or order of the format. Every line and section must appear exactly as written. Fill in the missing values without altering anything else; you cannot leave them blank.

Scoring Rubric:

Score how well the startup idea meets the criteria. Consider your certainty and doubts using this scale:

0: Completely sure the startup idea doesn’t meet the criterion.
10: Almost entirely sure the startup idea fails the criterion.
20: Mostly sure it doesn’t meet the criterion, though some aspects suggest it meets the criterion.
30: Leaning towards not meeting the criterion, but some aspects are still valuable, though not enough.
40: Close to an equal case for meeting or failing the criterion, but still leaning towards failure.
50: Exactly equal case for meeting or failing the criterion.
60: Leaning towards meeting the criterion, but still close to an equal case for meeting or failing.
70: Leaning towards meeting the criterion, with some reservations.
80: Mostly sure it meets the criterion, though some aspects suggest it could fail.
90: Almost entirely sure it meets the criterion.
100: Completely sure the startup idea meets the criterion.

Use this scale to express your level of confidence and opinion.

List of Criteria:

Criteria 1:
Criteria to assess: Is the application complete, appropriate, and intelligible?

Focus on the following aspects when evaluating the criteria:

1. The application must be entirely in English — this is a strict requirement.
2. If any part of the application is not in English, it should score below 60.
3. Control characters, special characters, formatting symbols, markup tags, and image references are acceptable.
4. Assess whether the application provides detailed responses to all required questions: there are 20 questions for each application, and a complete application should contain substantial answers to most of them.
5. It is acceptable to have "N/A" or an empty answer if a question is irrelevant or does not apply to the team or the product.
6. Examine each answer in detail to determine if the application fully addresses the required questions.
7. Immediately score below 60 if the application is not intelligible. Score 0 if it does not relate to business ideas and products, for example, if the product and solution remain unclear after reading the application.

Criteria 2:
Criteria to assess: Is the startup idea not just a concept but has been validated or tested?

There are four possibilities the solution can indicate:

- Ideation Stage (Pre-Seed Stage): Defining the problem-solution fit, conducting market research, prototyping, and validating the idea.
- Early Stage (Seed Stage): Developing a Minimum Viable Product (MVP) and preparing for initial launch.
- Growth Stage (Series A & B): Achieved product-market fit, scaling operations, and expanding customer adoption.
- Scale Stage (Series C and Beyond): Rapid scaling, market expansion, and organizational growth.

Focus on the following aspects when evaluating the criteria:

1. Identify the current stage of the product or startup (Concept, Prototype, Early Traction, Growth, or Scale).
2. If the product or startup is in the Prototype stage or beyond, provide evidence of prototyping and testing by users (e.g., user numbers, user feedback, geographical reach, or current operations). A mere claim of being in a particular stage is insufficient; facts are required.
3. Determine how many people the product currently serves and in which countries it operates.
4. Evaluate how well the team is positioned to deliver this solution effectively.

Important Considerations:

1. Startups in the Early Traction, Growth, or Scale stages should receive a score of more than 60, as they are generally beyond the prototype stage.
2. The goal is to confirm that the product is not merely a concept but has been tested and validated in the market.
3. Grade the application with a low score if no concrete product, service, or business model is being built or tested.
4. Be cautious with applications that claim to be in the Prototype stage or beyond but still lack a tangible product.

Criteria 3:
Criteria to assess: Is the market large enough, and is there sufficient demand?

Focus on the following aspects when evaluating the criteria:

1. The TAM (Total Addressable Market), SAM (Serviceable Available Market), and SOM (Serviceable Obtainable Market) should not be too small.
2. Use both top-down (e.g., from McKinsey, Gartner, Statista, IBISWorld) and bottom-up approaches (e.g., competitor revenue) to estimate market size.
3. Evaluate whether the startup meets market size expectations, especially if seeking funding support.
4. Assess whether the market and demand are rising using Google Trends and funding trends from sources like Crunchbase and CB Insights.

Criteria 4:
Criteria to assess: Can the startup outcompete others and stand out in the market?

Focus on the following aspects when evaluating the criteria:

1. Identify direct and indirect competitors.
2. Analyze the competitiveness of the market using tools like SWOT or Porter’s Five Forces model.
3. Determine if the startup idea is differentiated and superior to its competitors.
4. Assess whether the startup has a strong competitive moat (e.g., technological advantage, network effect, brand) or if it can be easily copied.

Criteria 5:
Criteria to assess: Does the startup have a business model to generate revenue or achieve financial sustainability?

Focus on the following aspects when evaluating the criteria:

1. Does the startup solve a real pain point for users?
2. Is it significantly better (or cheaper, faster, etc.) than existing alternatives?
3. Has the startup idea been tested with users, and are users willing to pay for the product or solution?
4. Does the startup have a scalable, clear, and recurring revenue stream? Can it become profitable?
5. Do customers demonstrate repeat engagement with the product?

Criteria 6:
Criteria to assess: Does the startup have a good team?

Focus on the following aspects when evaluating the criteria:

1. How many people are on the team, and how well-positioned are they to deliver the solution?
2. The startup team should cover all critical areas—product, technology, business, marketing, etc. Each member should excel in their field without overlap in responsibilities.
3. Evaluate whether the founders or core team members have experience in successfully building or scaling companies before. While this is a strong indicator of execution ability, the absence of such experience should not automatically disqualify the startup.
4. Determine whether the founders deeply understand the problem they are solving and if the team has a well-defined purpose and long-term goal or roadmap.
5. Assess how long the team has been working on the product or solution and if they can quickly turn ideas into a working product.
6. Evaluate whether the team listens to users and iterates based on feedback.
'''

RETRY_EXCEPTIONS = (
        httpx.RemoteProtocolError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.WriteTimeout,
        httpx.ProtocolError
        )

@retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        before_sleep=lambda _: print("Connection issue, retrying...")
        )

def evaluate_criteria(proposal):
    user_message = f"\n This is the solution that you are going to evaluate: \n {proposal} \n"
    try:
        time.sleep(max(0, 1.2 - (time.time() % 1)))
        completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-r1",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                    ],
                stream=True,
                temperature=0,
                timeout=httpx.Timeout(30.0, read=300.0)
                )
        response_text = []
        chunk_count = 0
        for chunk in completion:
            if chunk_count % 5 == 0 and psutil.virtual_memory().percent > 75:
                print("[WARN] Memory usage high, trimming output")
                response_text = response_text[-1000:]
            if chunk.choices[0].delta.content:
                response_text.append(chunk.choices[0].delta.content)
                chunk_count += 1
        final_response = "".join(response_text)
        print(final_response)
        return final_response
    except Exception as e:
        print(f"Failed request: {type(e).__name__} - {str(e)}")
        raise

def extract_key_elements_as_variables(text: str) -> dict:
    extracted_variables = {}
    evaluation_end_marker = "\n--- Evaluation Details End ---"

    # Split the evaluation details if the end marker is present
    evaluation_text = text.split(evaluation_end_marker, 1)[0] if evaluation_end_marker in text else text

    # Regular expression to match criteria scores, detailed reasoning, summaries, and improvement suggestions
    criteria_pattern = re.compile(
            r"The score of criteria(\d+):\s*(\d+)\s*\n?"
            r"Detailed reasoning\1:\s*(.*?)\s*\n?"
            r"Summary reasoning criteria\1:\s*(.*?)\s*\n?"
            r"Improvement suggestion criteria\1:\s*(.*?)(?=\nThe score of criteria|\Z)",
            re.DOTALL
            )

    for match in criteria_pattern.finditer(evaluation_text):
        num, score, detailed_reasoning, summary, improvement = match.groups()
        extracted_variables[f"score_criteria{num}"] = int(score)
        extracted_variables[f"detailed_reasoning_criteria{num}"] = detailed_reasoning.strip()
        extracted_variables[f"summary_reasoning_criteria{num}"] = summary.strip()
        extracted_variables[f"improvement_suggestion_criteria{num}"] = improvement.strip()

    # Extract the think_section and chain_of_thought separately
    think_section_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    extracted_variables["think_section"] = think_section_match.group(1).strip() if think_section_match else None

    chain_of_thought_match = re.search(r"<chain_of_thought>(.*?)</chain_of_thought>", text, re.DOTALL)
    extracted_variables["chain_of_thought"] = chain_of_thought_match.group(1).strip() if chain_of_thought_match else None

    return extracted_variables

def extract_think_parts(text: str) -> dict:
    think_parts = {}
    think_sections = ['introduction', 'criteria1', 'criteria2', 'criteria3', 'criteria4', 'criteria5', 'criteria6', 'conclusion']

    # Extract the raw chain_of_thought section from the text
    chain_of_thought_match = re.search(r"<chain_of_thought>(.*?)</chain_of_thought>", text, re.DOTALL)
    chain_of_thought = chain_of_thought_match.group(1).strip() if chain_of_thought_match else None
    think_parts['chain_of_thought'] = chain_of_thought

    # Loop through all defined think sections and extract their content
    for part in think_sections:
        pattern = re.compile(rf"<{part}>(.*?)</{part}>", re.DOTALL)
        match = pattern.search(chain_of_thought if chain_of_thought else '')
        think_parts[part] = match.group(1).strip() if match else None

    return think_parts

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    if 'proposal' not in data:
        return jsonify({"error": "Missing 'proposal' in request"}), 400

    final_response = evaluate_criteria(data['proposal'])
    extracted_data = extract_key_elements_as_variables(final_response)
    think_parts = extract_think_parts(final_response)
    extracted_data.update(think_parts)
    return jsonify(extracted_data)


@app.route('/research', methods=['POST'])
def research():
    start_time = time.time()
    data = request.json

    if 'proposal' not in data:
        return jsonify({"error": "Missing 'proposal' in request"}), 400

    if gradio_client is None:
        return jsonify({"error": "Gradio client not initialized. Check server logs for details."}), 503

    # Generate a unique session and message ID
    session_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    try:
        print(f"Processing research request for proposal: {data['proposal'][:50]}...")

        # First log the user message
        prompt = f"""
        There is a startup idea: "{data['proposal']}".

        Do you think this startup idea is highly novel? Please research this idea from three perspectives and generate a novelty score:

        1. Problem Uniqueness: Does this idea address an unmet or unrecognized need?
        2. Existing Solutions: Evaluate competitors (the most important factor), patent and intellectual property research, and relevant academic research.
        3. Differentiation: Assess the idea from the perspectives of technical innovation, business model innovation, market segmentation, and user experience.

        Your final answer should be as detailed and professional as possible, including a novelty score on a scale of 100 and a comprehensive report of over 5,000 words. The report should contain sections for the overview, problem uniqueness, existing solutions, differentiation, conclusion, and sources and references, with all sources displayed as hyperlinks. Ensure that the final content includes proper in-text citations with hyperlinks, making each citation a clickable link.
        """

        print("Logging user message...")
        try:
            log_result = gradio_client.predict(
                    text_input=prompt,
                    api_name="/log_user_message"
                    )
            print(f"Log result completed: {len(log_result) if isinstance(log_result, str) else 'Not a string'} characters")
        except Exception as e:
            print(f"Error during message logging: {str(e)}")
            return jsonify({"error": f"Error logging message: {str(e)}"}), 500

        # Then interact with agent using the logged message
        print("Interacting with agent...")
        try:
            result = gradio_client.predict(
                    messages=[{
                        "role": "user",
                        "content": log_result,
                        "metadata": {
                            "id": message_id,
                            "parent_id": session_id
                            }
                        }],
                    api_name="/interact_with_agent_1"
                    )
            print(f"Agent interaction completed: {type(result)}")
        except Exception as e:
            print(f"Error during agent interaction: {str(e)}")
            return jsonify({"error": f"Error during agent interaction: {str(e)}"}), 500

        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time

        # Parse the result using the response parser
        parsed_result = parse_agent_response(result)

        # Save the parsed result as markdown (optional)
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_report.md")
        save_as_markdown(parsed_result, report_path)
        print(f"Analysis report saved to: {report_path}")

        # Format response with both raw and parsed results
        response = {
                "execution_time": f"{execution_time:.2f} seconds",
                "raw_result": result,
                "parsed_result": parsed_result
                }

        # Save result to file for debugging
        try:
            with open('last_research_result.json', 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
            print("Saved result to last_research_result.json")
        except Exception as e:
            print(f"Error saving result: {str(e)}")

        return jsonify(response)

    except Exception as e:
        print(f"Unhandled error in research route: {str(e)}")
        return jsonify({"error": f"Error during research: {str(e)}"}), 500

def test_hf_connection():
    """Test the Hugging Face connection and print diagnostic information"""
    import requests
    try:
        print("Testing connection to Hugging Face...")
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        # Try to access the Hugging Face Hub API directly
        response = requests.get(
                "https://huggingface.co/api/spaces/MaoShen/Moonshot_DeepResearch",
                headers=headers,
                timeout=10
                )

        print(f"HF API Response Status: {response.status_code}")
        if response.status_code == 200:
            print("Successfully connected to Hugging Face API")
            if response.headers.get('content-type', '').startswith('application/json'):
                print(f"Response Content (first 1000 chars): {response.text[:1000]}")
            else:
                print(f"Response Content-Type: {response.headers.get('content-type')}")
        else:
            print(f"Failed to connect to Hugging Face API: {response.text}")
    except Exception as e:
        print(f"Error testing HF connection: {str(e)}")

if __name__ == "__main__":
    # Test the Hugging Face connection before starting the server
    test_hf_connection()

    port = int(os.getenv("PORT", 10000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
